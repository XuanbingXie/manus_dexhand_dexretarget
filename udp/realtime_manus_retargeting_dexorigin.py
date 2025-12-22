#!/usr/bin/env python3
"""
实时 Manus 手套到机器人手的重定向可视化
使用 MuJoCo 进行可视化，使用 qpos 直接控制关节位置
"""
import sys
import time
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer
import tyro
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent / "dex-retargeting" / "src"))

from dex_retargeting.constants import (
    RobotName,
    RetargetingType,
    HandType,
    get_default_config_path,
)
from dex_retargeting.retargeting_config import RetargetingConfig
from manus_skeleton_parser import ManusSkeletonParser
import socket


class ManusUDPReceiver:
    """接收并解析 Manus UDP 数据"""
    
    def __init__(self, port=5006):
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("", self.port))
        self.sock.settimeout(0.01)
        self.parser = ManusSkeletonParser()
        logger.info(f"UDP receiver initialized on port {self.port}")
    
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


def get_mujoco_model_path(robot_name: RobotName) -> str:
    model_paths = {
        RobotName.shadow: "anytwist/model/robot/shadowhand_right.xml",
    }
    
    if robot_name not in model_paths:
        available_robots = list(model_paths.keys())
        raise ValueError(f"MuJoCo model not available for robot: {robot_name}. Available: {available_robots}")
    
    model_path = Path(__file__).parent.parent / model_paths[robot_name]
    if not model_path.exists():
        raise FileNotFoundError(f"MuJoCo model not found: {model_path}")
    
    return str(model_path)


def main(
    robot_name: RobotName,
    retargeting_type: RetargetingType,
    hand_type: HandType,
    udp_port: int = 5006,
):
    """
    从 Manus 手套接收数据并实时重定向到机器人手，使用 MuJoCo 可视化。

    Args:
        robot_name: 机器人标识符（shadow）
        retargeting_type: 重定向类型（vector, position, dexpilot）
        hand_type: 手的类型（right, left）
        udp_port: UDP 端口号，默认 5006
    """
    
    try:
        model_path = get_mujoco_model_path(robot_name)
        logger.info(f"Loading MuJoCo model: {model_path}")
        model = mujoco.MjModel.from_xml_path(model_path)
        data = mujoco.MjData(model)
    except Exception as e:
        logger.error(f"Failed to load MuJoCo model: {e}")
        return

    config_path = get_default_config_path(robot_name, retargeting_type, hand_type)
    robot_dir = (
        Path(__file__).absolute().parent.parent / "dex-retargeting" / "assets" / "dex-urdf" / "robots" / "hands"
    )
    
    RetargetingConfig.set_default_urdf_dir(str(robot_dir))
    logger.info(f"Loading retargeting config from {config_path}")
    
    try:
        config = RetargetingConfig.load_from_file(config_path)
        retargeting = config.build()
        logger.info("dex-retargeting initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize dex-retargeting: {e}")
        return
    
    manus_receiver = ManusUDPReceiver(port=udp_port)
    
    logger.info("Starting real-time retargeting loop")
    logger.info(f"Waiting for Manus data on UDP port {udp_port}...")
    logger.info("Make sure SDKClient_Linux is running and sending data")
    
    frame_count = 0
    last_data_time = time.time()
    fps_counter = []
    fps_start_time = time.time()
    retarget_times = []
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            frame_count += 1
            
            joint_pos = manus_receiver.receive()
            
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
                    
                    start_time = time.perf_counter()
                    qpos = retargeting.retarget(ref_value)
                    retarget_time = time.perf_counter() - start_time
                    retarget_times.append(retarget_time)

                    retargeting_joint_names = retargeting.joint_names
                    
                    for i, ret_joint_name in enumerate(retargeting_joint_names):
                        if ret_joint_name.startswith('WRJ'):
                            continue
                            
                        if robot_name == RobotName.shadow:
                            joint_name = f"rh_{ret_joint_name}"
                        else:
                            joint_name = ret_joint_name
                        
                        try:
                            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                            data.qpos[joint_id] = qpos[i]
                        except:
                            pass
                    
                    try:
                        wrist_joints = ['rh_WRJ1', 'rh_WRJ2']
                        for wrist_joint in wrist_joints:
                            try:
                                wrist_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, wrist_joint)
                                data.qpos[wrist_id] = 0.0
                            except:
                                pass
                    except:
                        pass
                    
                    fps_counter.append(time.time())
                    
                except Exception as e:
                    logger.error(f"Retargeting failed: {e}")
            else:
                if time.time() - last_data_time > 5.0:
                    logger.warning("No data received for 5 seconds. Check Manus connection.")
                    last_data_time = time.time()
            
            mujoco.mj_step(model, data)
            
            try:
                wrist_joint_ids = [0, 1]
                for wrist_id in wrist_joint_ids:
                    data.qpos[wrist_id] = 0.0
                    data.qvel[wrist_id] = 0.0
            except:
                pass
            
            viewer.sync()
            
            if time.time() - fps_start_time >= 1.0:
                if fps_counter:
                    fps = len(fps_counter) / (time.time() - fps_start_time)
                    if retarget_times:
                        avg_retarget_time = np.mean(retarget_times) * 1000
                        max_retarget_time = np.max(retarget_times) * 1000
                        retarget_times = []
                    else:
                        logger.info(f"FPS: {fps:.1f} | Frame: {frame_count} | No retargeting data")
                    
                    fps_counter = []
                    fps_start_time = time.time()
    
    manus_receiver.close()
    logger.info("Retargeting stopped")


if __name__ == "__main__":
    tyro.cli(main)