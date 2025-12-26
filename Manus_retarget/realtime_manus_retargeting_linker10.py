#!/usr/bin/env python3
"""Manus手套到Gaia机械手的实时Retargeting"""
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

from dex_retargeting.constants import RetargetingType, HandType
from dex_retargeting.retargeting_config import RetargetingConfig
from manus_skeleton_parser import ManusSkeletonParser
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
    finger_scaling: tuple = (1.3, 0.9, 0.9, 1.0, 1.0),  # 参考真机的缩放系数
    use_dexpilot: bool = False,
):
    """Manus到Gaia手实时Retargeting
    
    Args:
        hand_type: 左手或右手
        udp_port: UDP端口
        finger_scaling: 5个手指的缩放因子 (thumb, index, middle, ring, pinky)
        finger_offset: 5个手指的偏移量 (thumb, index, middle, ring, pinky)
        use_dexpilot: 是否使用DexPilot模式（更简单但可能精度略低）
    """
    model_path = Path(__file__).parent.parent / "anytwist/model/robot/l10_right.xml"
    if not model_path.exists():
        logger.error(f"MuJoCo model not found: {model_path}")
        return

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    hand_type_str = "right" if hand_type == HandType.right else "left"
    config_dir = Path(__file__).parent.parent / "dex-retargeting/src/dex_retargeting/configs/teleop"
    
    if use_dexpilot:
        config_path = config_dir / f"linker_hand_{hand_type_str}_dexpilot.yml"
        logger.info("Using DexPilot mode")
    else:
        config_path = config_dir / f"linker_hand_{hand_type_str}_manus.yml"
        logger.info("Using Vector mode")

    urdf_dir = Path(__file__).parent.parent / "dex-retargeting/assets/dex-urdf/robots/hands"
    RetargetingConfig.set_default_urdf_dir(str(urdf_dir))

    config = RetargetingConfig.load_from_file(str(config_path))
    retargeting = config.build()
    logger.info("Retargeting initialized")

    manus_receiver = ManusUDPReceiver(port=udp_port)
    scaling_vector = np.array(list(finger_scaling), dtype=np.float32)

    ref_rot_fixed = R.from_euler('y', -90, degrees=True)

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
                    if use_dexpilot:
                        # DexPilot模式：需要提供所有手指对之间的向量
                        # 对于5个手指，需要10个向量：手指间的连接(10个) + 手指到手腕的连接(5个) = 15个
                        # 但实际上DexPilot只用前10个
                        fingertip_indices = [4, 9, 14, 19, 24]  # thumb, index, middle, ring, pinky tips
                        fingertip_pos = joint_pos[fingertip_indices, :]
                        wrist_pos = joint_pos[0, :]  # 手腕位置
                        
                        # 生成所有手指对之间的向量 + 手指到手腕的向量
                        ref_vectors = []
                        # 手指之间的连接
                        for i in range(len(fingertip_indices)):
                            for j in range(i + 1, len(fingertip_indices)):
                                ref_vectors.append(fingertip_pos[j] - fingertip_pos[i])
                        # 手指到手腕的连接
                        for i in range(len(fingertip_indices)):
                            ref_vectors.append(fingertip_pos[i] - wrist_pos)
                        
                        ref_value = np.array(ref_vectors, dtype=np.float32)
                        logger.debug(f"DexPilot ref_value shape: {ref_value.shape}")
                        qpos = retargeting.retarget(ref_value)
                    else:
                        # Vector模式：使用向量差
                        indices = retargeting.optimizer.target_link_human_indices
                        origin_indices = indices[0, :]
                        task_indices = indices[1, :]
                        ref_value = joint_pos[task_indices, :] - joint_pos[origin_indices, :]
                        # 不使用 scaling_vector，让配置文件中的 scaling_factor 来处理
                        qpos = retargeting.retarget(ref_value)

                    # 设置独立关节
                    independent_joints = [
                        "thumb_cmc_roll", "thumb_cmc_yaw", "thumb_cmc_pitch",
                        "index_mcp_roll", "index_mcp_pitch",
                        "middle_mcp_pitch",
                        "ring_mcp_roll", "ring_mcp_pitch",
                        "pinky_mcp_roll", "pinky_mcp_pitch"
                    ]
                    
                    logger.debug(f"retargeting.joint_names: {retargeting.joint_names}")
                    logger.debug(f"qpos shape: {qpos.shape}")
                    
                    # 直接使用优化器输出的所有关节值，并应用每个手指的缩放
                    # 定义每个关节对应的手指索引 (0=thumb, 1=index, 2=middle, 3=ring, 4=pinky)
                    joint_to_finger = {
                        "thumb_cmc_roll": 0, "thumb_cmc_yaw": 0, "thumb_cmc_pitch": 0, "thumb_mcp": 0, "thumb_ip": 0,
                        "index_mcp_roll": 1, "index_mcp_pitch": 1, "index_pip": 1, "index_dip": 1,
                        "middle_mcp_pitch": 2, "middle_pip": 2, "middle_dip": 2,
                        "ring_mcp_roll": 3, "ring_mcp_pitch": 3, "ring_pip": 3, "ring_dip": 3,
                        "pinky_mcp_roll": 4, "pinky_mcp_pitch": 4, "pinky_pip": 4, "pinky_dip": 4,
                    }
                    
                    for i, joint_name in enumerate(retargeting.joint_names):
                        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                        if joint_id >= 0:
                            value = qpos[i]
                            
                            # 应用手指缩放
                            if joint_name in joint_to_finger:
                                finger_idx = joint_to_finger[joint_name]
                                value = value * finger_scaling[finger_idx]
                            
                            # 食指分指方向需要取反
                            if joint_name == "index_mcp_roll":
                                value = -value
                            
                            data.qpos[joint_id] = value

                    fps_counter.append(time.time())
                except Exception as e:
                    logger.error(f"Retargeting failed: {e}")
            else:
                if time.time() - last_data_time > 5.0:
                    logger.warning("No data received for 5 seconds")
                    last_data_time = time.time()

            mujoco.mj_forward(model, data)
            viewer.sync()

            if time.time() - fps_start_time >= 1.0:
                if fps_counter:
                    fps = len(fps_counter) / (time.time() - fps_start_time)
                    logger.info(f"FPS: {fps:.1f} | Frame: {frame_count}")
                    fps_counter = []
                    fps_start_time = time.time()

    manus_receiver.close()


if __name__ == "__main__":
    tyro.cli(main)
