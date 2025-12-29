#!/usr/bin/env python3
import sys
import time
from pathlib import Path
import numpy as np
import mujoco
import mujoco.viewer
import tyro
from loguru import logger
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).parent.parent / "Manus_retarget"))
from manus_skeleton_parser import ManusSkeletonParser
from fingertip_ik import FingerIK
import socket


class ManusUDPReceiver:
    def __init__(self, port=5006):
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("", self.port))
        self.sock.settimeout(0.01)
        self.parser = ManusSkeletonParser()

    def close(self):
        self.sock.close()


def main(
    udp_port: int = 5006,
    position_scale: float = 1.0,
):
    model_path = Path(__file__).parent.parent / "anytwist/model/robot/l10_right.xml"
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        return

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    
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

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
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
                    
                    for finger_name, result in results.items():
                        if result["success"]:
                            last_qpos[finger_name] = result["qpos"]
                            
                            for joint_name, value in zip(ik_solver.finger_joints[finger_name], result["qpos"]):
                                joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                                data.qpos[joint_id] = value
                    
                    fps_counter.append(time.time())
                    
                except Exception as e:
                    logger.error(f"IK failed: {e}")
            else:
                if time.time() - last_data_time > 5.0:
                    logger.warning("No data")
                    last_data_time = time.time()

            mujoco.mj_forward(model, data)
            viewer.sync()

            if time.time() - fps_start_time >= 1.0:
                if fps_counter:
                    fps = len(fps_counter) / (time.time() - fps_start_time)
                    logger.info(f"FPS: {fps:.1f}")
                    fps_counter = []
                    fps_start_time = time.time()

    manus_receiver.close()


if __name__ == "__main__":
    tyro.cli(main)
