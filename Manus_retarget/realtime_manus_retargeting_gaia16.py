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

from dex_retargeting.constants import (
    RobotName,
    RetargetingType,
    HandType,
)
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
    
    def receive(self):
        try:
            data, addr = self.sock.recvfrom(4096)
            nodes = self.parser.parse_udp_data(data)
            
            if len(nodes) < 25:
                return None
            
            keypoints = self.parser.to_mediapipe_format(nodes)
            return keypoints
            
        except socket.timeout:
            return None
        except Exception as e:
            logger.error(f"Failed to receive data: {e}")
            return None
    
    def close(self):
        self.sock.close()


def main(
    retargeting_type: RetargetingType = RetargetingType.vector,
    hand_type: HandType = HandType.right,
    udp_port: int = 5006,
    use_manus_direct: bool = True,  
):
    model_path = Path(__file__).parent.parent / "anytwist/model/robot/gaia_hand16_right.xml"
    if not model_path.exists():
        logger.error(f"MuJoCo model not found: {model_path}")
        return
    
    try:
        logger.info(f"Loading MuJoCo model: {model_path}")
        model = mujoco.MjModel.from_xml_path(str(model_path))
        data = mujoco.MjData(model)
    except Exception as e:
        logger.error(f"Failed to load MuJoCo model: {e}")
        return
    
    hand_type_str = "right" if hand_type == HandType.right else "left"
    
    config_dir = Path(__file__).parent.parent / "dex-retargeting/src/dex_retargeting/configs/teleop"
    
    if use_manus_direct:
        config_path = config_dir / f"right_gaia16_hand_{hand_type_str}_manus.yml"
        logger.info("Using Manus direct mapping (25 nodes)")
    else:
        if retargeting_type == RetargetingType.dexpilot:
            config_path = config_dir / f"right_gaia16_hand_{hand_type_str}_dexpilot.yml"
        else:
            config_path = config_dir / f"right_gaia16_hand_{hand_type_str}.yml"
        logger.info("Using MediaPipe format (21 nodes)")
    
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        logger.info(f"Available configs in {config_dir}:")
        if config_dir.exists():
            for f in config_dir.glob("*gaia16*.yml"):
                logger.info(f"  - {f.name}")
        return
    
    urdf_dir = Path(__file__).parent.parent / "dex-retargeting/assets/dex-urdf/robots/hands"
    RetargetingConfig.set_default_urdf_dir(str(urdf_dir))
    logger.info(f"Loading retargeting config from {config_path}")
    
    try:
        config = RetargetingConfig.load_from_file(str(config_path))
        retargeting = config.build()
        logger.info("dex-retargeting initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize dex-retargeting: {e}")
        return
    
    manus_receiver = ManusUDPReceiver(port=udp_port)
    
    logger.info(f"Waiting for Manus data on UDP port {udp_port}...")
    
    frame_count = 0
    last_data_time = time.time()
    fps_counter = []
    fps_start_time = time.time()
    retarget_times = []
    
    # 存储初始手腕旋转，用于补偿
    initial_wrist_rot = None
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            frame_count += 1
            
            if use_manus_direct:
                try:
                    data_raw, addr = manus_receiver.sock.recvfrom(4096)
                    nodes = manus_receiver.parser.parse_udp_data(data_raw)
                    if len(nodes) >= 25:
                        # 获取手腕旋转
                        wrist_quat = nodes[0]['rotation']  # [x, y, z, w]
                        current_wrist_rot = R.from_quat(wrist_quat)
                        
                        # 第一帧：存储初始手腕旋转
                        if initial_wrist_rot is None:
                            initial_wrist_rot = current_wrist_rot
                            logger.info("初始手腕旋转已捕获")
                        
                        # 提取位置
                        joint_pos = np.zeros((25, 3), dtype=np.float32)
                        for i, node in enumerate(nodes[:25]):
                            joint_pos[i] = node['position']
                    else:
                        joint_pos = None
                        current_wrist_rot = None
                except socket.timeout:
                    joint_pos = None
                    current_wrist_rot = None
                except Exception as e:
                    logger.error(f"Error parsing Manus data: {e}")
                    joint_pos = None
                    current_wrist_rot = None
            else:
                joint_pos = manus_receiver.receive()
                current_wrist_rot = None
            
            if joint_pos is not None:
                last_data_time = time.time()
                
                try:
                    retargeting_type_str = retargeting.optimizer.retargeting_type
                    indices = retargeting.optimizer.target_link_human_indices
                    
                    if retargeting_type_str == "POSITION":
                        ref_value = joint_pos[indices, :]
                    else:
                        origin_indices = indices[0, :]
                        task_indices = indices[1, :]
                        ref_value = joint_pos[task_indices, :] - joint_pos[origin_indices, :]
                        
                        # 手腕旋转补偿：将向量转换到初始手腕坐标系
                        if use_manus_direct and current_wrist_rot is not None and initial_wrist_rot is not None:
                            rot_compensation = initial_wrist_rot * current_wrist_rot.inv()
                            ref_value = rot_compensation.apply(ref_value)
                    
                    start_time = time.perf_counter()
                    qpos = retargeting.retarget(ref_value)
                    retarget_time = time.perf_counter() - start_time
                    retarget_times.append(retarget_time)

                    retargeting_joint_names = retargeting.joint_names
                    
                    for i, joint_name in enumerate(retargeting_joint_names):
                        try:
                            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                            data.qpos[joint_id] = qpos[i]
                        except:
                            pass
                    
                    fps_counter.append(time.time())
                    
                except Exception as e:
                    logger.error(f"Retargeting failed: {e}")
            else:
                if time.time() - last_data_time > 5.0:
                    logger.warning("No data received for 5 seconds. Check Manus connection.")
                    last_data_time = time.time()
            
            mujoco.mj_forward(model, data)
            viewer.sync()
            
            if time.time() - fps_start_time >= 1.0:
                if fps_counter:
                    fps = len(fps_counter) / (time.time() - fps_start_time)
                    if retarget_times:
                        avg_retarget_time = np.mean(retarget_times) * 1000
                        max_retarget_time = np.max(retarget_times) * 1000
                        logger.info(f"FPS: {fps:.1f} | Frame: {frame_count} | Retarget: {avg_retarget_time:.2f}ms (max: {max_retarget_time:.2f}ms)")
                        retarget_times = []
                    
                    fps_counter = []
                    fps_start_time = time.time()
    
    manus_receiver.close()
    logger.info("Retargeting stopped")


if __name__ == "__main__":
    tyro.cli(main)
