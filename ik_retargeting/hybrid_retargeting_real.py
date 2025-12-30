#!/usr/bin/env python3
import sys
import time
from pathlib import Path
import numpy as np
import tyro
from loguru import logger
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).parent.parent / "dex-retargeting" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "Manus_retarget"))
sys.path.insert(0, str(Path(__file__).parent.parent / "linker_hand_python_sdk"))

from dex_retargeting.constants import HandType
from dex_retargeting.retargeting_config import RetargetingConfig
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


def qpos_to_motor_positions(qpos_dict, joint_names):
    required_joints = [
        "thumb_cmc_roll", "thumb_cmc_yaw", "thumb_cmc_pitch",
        "index_mcp_roll", "index_mcp_pitch",
        "middle_mcp_pitch",
        "ring_mcp_roll", "ring_mcp_pitch",
        "pinky_mcp_roll", "pinky_mcp_pitch",
    ]
    
    ctrl = np.zeros(10)
    
    for i, joint_name in enumerate(joint_names):
        if joint_name in required_joints:
            ctrl_idx = required_joints.index(joint_name)
            ctrl[ctrl_idx] = qpos_dict.get(joint_name, 0.0)
    
    bounds = np.array([1.1339, 1.9189, 0.5146, 0.2181, 1.3607, 1.3607, 0.2181, 1.3607, 0.2181, 1.3607])
    ctrl_normalized = np.clip(ctrl / bounds, 0, 1)
    
    cmd = np.round(ctrl_normalized * 255).astype(int)
    cmd = cmd[[2, 1, 4, 5, 7, 9, 3, 6, 8, 0]]
    invert_mask = np.array([255, 255, 255, 255, 255, 255, 0, 0, 0, 255])
    cmd = np.abs(cmd - invert_mask)
    
    return cmd.tolist()


def main(
    hand_type: HandType = HandType.right,
    udp_port: int = 5006,
    thumb_position_scale: float = 0.47,
    four_finger_scaling: tuple = (0.7, 0.7, 0.85, 1.2),
    four_finger_offset: tuple = (-0.00, -0.0, -0.0, -0.00),
    use_dexpilot: bool = True,
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
            hand_type_str = "right" if hand_type == HandType.right else "left"
            linker_hand = LinkerHandApi(
                hand_type=hand_type_str,
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

    hand_type_str = "right" if hand_type == HandType.right else "left"
    config_dir = Path(__file__).parent.parent / "dex-retargeting/src/dex_retargeting/configs/teleop"
    
    if use_dexpilot:
        config_path = config_dir / f"linker_hand_{hand_type_str}_dexpilot.yml"
        logger.info("Using DexPilot mode for four fingers")
    else:
        config_path = config_dir / f"linker_hand_{hand_type_str}_manus.yml"
        logger.info("Using Vector mode for four fingers")

    urdf_dir = Path(__file__).parent.parent / "dex-retargeting/assets/dex-urdf/robots/hands"
    RetargetingConfig.set_default_urdf_dir(str(urdf_dir))

    config = RetargetingConfig.load_from_file(str(config_path))
    retargeting = config.build()
    logger.info("Retargeting initialized for four fingers")

    model_path = Path(__file__).parent.parent / "anytwist/model/robot/l10_right.xml"
    ik_solver = FingerIK(str(model_path))
    logger.info("IK solver initialized for thumb only")

    manus_receiver = ManusUDPReceiver(port=udp_port)
    ref_rot_fixed = R.from_euler('y', -90, degrees=True)
    
    last_thumb_qpos = np.array([0.3, 0.5, 0.2])
    
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
                    thumb_target = joint_pos[4] * thumb_position_scale
                    thumb_qpos, thumb_success = ik_solver.solve_ik("thumb", thumb_target, last_thumb_qpos)
                    
                    if thumb_success:
                        last_thumb_qpos = thumb_qpos
                    
                    if use_dexpilot:
                        fingertip_indices = [4, 9, 14, 19, 24]
                        fingertip_pos = joint_pos[fingertip_indices, :]
                        wrist_pos = joint_pos[0, :]
                        ref_vectors = []
                        for i in range(len(fingertip_indices)):
                            for j in range(i + 1, len(fingertip_indices)):
                                ref_vectors.append(fingertip_pos[j] - fingertip_pos[i])
                        for i in range(len(fingertip_indices)):
                            ref_vectors.append(fingertip_pos[i] - wrist_pos)
                        
                        ref_value = np.array(ref_vectors, dtype=np.float32)
                        qpos = retargeting.retarget(ref_value)
                    else:
                        indices = retargeting.optimizer.target_link_human_indices
                        origin_indices = indices[0, :]
                        task_indices = indices[1, :]
                        ref_value = joint_pos[task_indices, :] - joint_pos[origin_indices, :]
                        qpos = retargeting.retarget(ref_value)
                    
                    joint_to_finger = {
                        "index_mcp_roll": 0, "index_mcp_pitch": 0, "index_pip": 0, "index_dip": 0,
                        "middle_mcp_pitch": 1, "middle_pip": 1, "middle_dip": 1,
                        "ring_mcp_roll": 2, "ring_mcp_pitch": 2, "ring_pip": 2, "ring_dip": 2,
                        "pinky_mcp_roll": 3, "pinky_mcp_pitch": 3, "pinky_pip": 3, "pinky_dip": 3,
                    }
                    
                    qpos_dict = {}
                    
                    qpos_dict["thumb_cmc_roll"] = thumb_qpos[0]
                    qpos_dict["thumb_cmc_yaw"] = thumb_qpos[1] * 3.2
                    qpos_dict["thumb_cmc_pitch"] = thumb_qpos[2] * 0.8
                    
                    for i, joint_name in enumerate(retargeting.joint_names):
                        if joint_name.startswith("thumb"):
                            continue
                        
                        value = qpos[i]
                        
                        if joint_name in joint_to_finger:
                            finger_idx = joint_to_finger[joint_name]
                            value = value * four_finger_scaling[finger_idx]
                        
                        if joint_name == "index_mcp_roll":
                            value = -value
                        
                        if joint_name in joint_to_finger:
                            finger_idx = joint_to_finger[joint_name]
                            value = value - four_finger_offset[finger_idx]
                        
                        qpos_dict[joint_name] = value
                    
                    motor_positions = qpos_to_motor_positions(qpos_dict, retargeting.joint_names)
                    
                    if time.time() - last_print_time >= 10.0:
                        logger.info(f"Motor: {motor_positions}")
                        last_print_time = time.time()
                    
                    if not dry_run and linker_hand is not None:
                        linker_hand.finger_move(pose=motor_positions)
                    
                    fps_counter.append(time.time())
                    
                except Exception as e:
                    logger.error(f"Retargeting failed: {e}")
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
