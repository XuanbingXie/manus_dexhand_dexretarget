#!/usr/bin/env python3
import sys
import time
from pathlib import Path
import numpy as np
import tyro
from loguru import logger
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).parent.parent / "Manus_retarget"))
sys.path.insert(0, str(Path(__file__).parent.parent / "linker_hand_python_sdk"))
from manus_skeleton_parser import ManusSkeletonParser
from fingertip_ik import FingerIK
import socket

try:
    from LinkerHand.linker_hand_api import LinkerHandApi
    LINKER_AVAILABLE = True
except ImportError:
    LINKER_AVAILABLE = False
    logger.warning("LinkerHand SDK not available")


class ManusUDPReceiver:
    def __init__(self, port=5006):
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("", self.port))
        self.sock.settimeout(0.01)
        self.parser = ManusSkeletonParser()

    def close(self):
        self.sock.close()


def qpos_to_motor_positions(qpos_dict):
    required_joints = [
        "thumb_cmc_roll", "thumb_cmc_yaw", "thumb_cmc_pitch",
        "index_mcp_roll", "index_mcp_pitch",
        "middle_mcp_pitch",
        "ring_mcp_roll", "ring_mcp_pitch",
        "pinky_mcp_roll", "pinky_mcp_pitch",
    ]
    
    ctrl = np.zeros(10)
    ctrl[0] = qpos_dict.get("thumb", [0,0,0])[0]
    ctrl[1] = qpos_dict.get("thumb", [0,0,0])[1]
    ctrl[2] = qpos_dict.get("thumb", [0,0,0])[2]
    ctrl[3] = qpos_dict.get("index", [0,0])[0]
    ctrl[4] = qpos_dict.get("index", [0,0])[1]
    ctrl[5] = qpos_dict.get("middle", [0])[0]
    ctrl[6] = qpos_dict.get("ring", [0,0])[0]
    ctrl[7] = qpos_dict.get("ring", [0,0])[1]
    ctrl[8] = qpos_dict.get("pinky", [0,0])[0]
    ctrl[9] = qpos_dict.get("pinky", [0,0])[1]
    
    bounds = np.array([1.1339, 1.9189, 0.5146, 0.2181, 1.3607, 1.3607, 0.2181, 1.3607, 0.2181, 1.3607])
    ctrl_normalized = np.clip(ctrl / bounds, 0, 1)
    
    cmd = np.round(ctrl_normalized * 255).astype(int)
    cmd = cmd[[2, 1, 4, 5, 7, 9, 3, 6, 8, 0]]
    invert_mask = np.array([255, 255, 255, 255, 255, 255, 0, 0, 0, 255])
    cmd = np.abs(cmd - invert_mask)
    
    return cmd.tolist()


def main(
    udp_port: int = 5006,
    position_scale: float = 1.0,
    can_interface: str = "can0",
    hand_joint: str = "L10",
    dry_run: bool = True,
):
    linker_hand = None
    if not dry_run:
        if not LINKER_AVAILABLE:
            logger.error("LinkerHand SDK not available")
            return
            
        try:
            linker_hand = LinkerHandApi(
                hand_type="right",
                hand_joint=hand_joint,
                can=can_interface
            )
            logger.info("Linker Hand initialized")
            linker_hand.set_speed(speed=[255, 255, 255, 255, 255])
            linker_hand.set_torque(torque=[200, 200, 200, 200, 200])
        except Exception as e:
            logger.error(f"Failed to initialize: {e}")
            return
    else:
        logger.info("Dry run mode")

    model_path = Path(__file__).parent.parent / "anytwist/model/robot/l10_right.xml"
    ik_solver = FingerIK(str(model_path))
    
    manus_receiver = ManusUDPReceiver(port=udp_port)
    ref_rot_fixed = R.from_euler('y', -90, degrees=True)
    
    manus_fingertip_indices = {
        "thumb": 4,
        "index": 9,
        "middle": 14,
        "ring": 19,
        "pinky": 24
    }
    
    last_qpos = {
        "thumb": np.array([0.3, 0.5, 0.2]),
        "index": np.array([0.0, 0.3]),
        "middle": np.array([0.3]),
        "ring": np.array([0.0, 0.3]),
        "pinky": np.array([0.0, 0.3])
    }
    
    frame_count = 0
    last_data_time = time.time()
    fps_counter = []
    fps_start_time = time.time()
    last_print_time = time.time()

    try:
        while True:
            frame_count += 1
            joint_pos = None

            try:
                data_raw, _ = manus_receiver.sock.recvfrom(4096)
                nodes = manus_receiver.parser.parse_udp_data(data_raw)
                if len(nodes) >= 25:
                    joint_pos = np.zeros((25, 3), dtype=np.float32)
                    parent_quat = nodes[0]['rotation']
                    parent_pos = nodes[0]['position']
                    parent_rot = R.from_quat(parent_quat)
                    transform_rot = ref_rot_fixed * parent_rot.inv()

                    for i, node in enumerate(nodes[:25]):
                        joint_pos[i] = transform_rot.apply(node['position'] - parent_pos)
            except socket.timeout:
                pass
            except Exception as e:
                logger.error(f"Error: {e}")

            if joint_pos is not None:
                last_data_time = time.time()
                
                try:
                    target_positions = {}
                    for finger_name, idx in manus_fingertip_indices.items():
                        target_positions[finger_name] = joint_pos[idx] * position_scale
                    
                    results = ik_solver.solve_all_fingers(target_positions, last_qpos)
                    
                    qpos_dict = {}
                    for finger_name, result in results.items():
                        if result["success"]:
                            last_qpos[finger_name] = result["qpos"]
                            qpos_dict[finger_name] = result["qpos"]
                    
                    motor_positions = qpos_to_motor_positions(qpos_dict)
                    
                    if time.time() - last_print_time >= 10.0:
                        logger.info(f"Motor: {motor_positions}")
                        last_print_time = time.time()
                    
                    if not dry_run and linker_hand is not None:
                        linker_hand.finger_move(pose=motor_positions)
                    
                    fps_counter.append(time.time())
                    
                except Exception as e:
                    logger.error(f"IK failed: {e}")
                    import traceback
                    traceback.print_exc()

            if time.time() - fps_start_time >= 10.0:
                if fps_counter:
                    fps = len(fps_counter) / (time.time() - fps_start_time)
                    logger.info(f"FPS: {fps:.1f} Hz")
                    fps_counter = []
                    fps_start_time = time.time()

    except KeyboardInterrupt:
        logger.info("Stopping")
    finally:
        manus_receiver.close()


if __name__ == "__main__":
    tyro.cli(main)
