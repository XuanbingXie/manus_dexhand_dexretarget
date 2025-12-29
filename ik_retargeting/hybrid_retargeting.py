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

sys.path.insert(0, str(Path(__file__).parent.parent / "dex-retargeting" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "Manus_retarget"))

from dex_retargeting.constants import HandType
from dex_retargeting.retargeting_config import RetargetingConfig
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
    hand_type: HandType = HandType.right,
    udp_port: int = 5006,
    thumb_position_scale: float = 1.0,
    four_finger_scaling: tuple = (1.0, 0.9, 1.0, 1.0),
    use_dexpilot: bool = False,
):
    model_path = Path(__file__).parent.parent / "anytwist/model/robot/l10_right.xml"
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        return

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

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

    ik_solver = FingerIK(str(model_path))
    logger.info("IK solver initialized for thumb only")

    manus_receiver = ManusUDPReceiver(port=udp_port)
    ref_rot_fixed = R.from_euler('y', -90, degrees=True)
    
    last_thumb_qpos = np.array([0.3, 0.5, 0.2])
    
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
                    thumb_target = joint_pos[4] * thumb_position_scale
                    thumb_qpos, thumb_success = ik_solver.solve_ik("thumb", thumb_target, last_thumb_qpos)
                    
                    if thumb_success:
                        last_thumb_qpos = thumb_qpos
                        for joint_name, value in zip(ik_solver.finger_joints["thumb"], thumb_qpos):
                            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                            data.qpos[joint_id] = value
                    
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
                    
                    for i, joint_name in enumerate(retargeting.joint_names):
                        if joint_name.startswith("thumb"):
                            continue
                        
                        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                        if joint_id >= 0:
                            value = qpos[i]
                            
                            if joint_name in joint_to_finger:
                                finger_idx = joint_to_finger[joint_name]
                                value = value * four_finger_scaling[finger_idx]
                            
                            if joint_name == "index_mcp_roll":
                                value = -value
                            
                            data.qpos[joint_id] = value
                    
                    fps_counter.append(time.time())
                    
                except Exception as e:
                    logger.error(f"Retargeting failed: {e}")
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
