#!/usr/bin/env python3
import sys
import time
from pathlib import Path

import numpy as np
import tyro
from loguru import logger
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).parent.parent / "dex-retargeting" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "linker_hand_python_sdk"))

from dex_retargeting.constants import RetargetingType, HandType
from dex_retargeting.retargeting_config import RetargetingConfig
from manus_skeleton_parser import ManusSkeletonParser
import socket

try:
    from LinkerHand.linker_hand_api import LinkerHandApi
    LINKER_AVAILABLE = True
except ImportError:
    LINKER_AVAILABLE = False
    logger.warning("LinkerHand SDK not available, running in dry-run mode only")


class ManusUDPReceiver:
    def __init__(self, port=5006):
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("", self.port))
        self.sock.settimeout(0.01)
        self.parser = ManusSkeletonParser()

    def close(self):
        self.sock.close()


def qpos_to_motor_positions(qpos, joint_names):
    required_joints = [
        "thumb_cmc_roll",      # 0
        "thumb_cmc_yaw",       # 1
        "thumb_cmc_pitch",     # 2
        "index_mcp_roll",      # 3
        "index_mcp_pitch",     # 4
        "middle_mcp_pitch",    # 5
        "ring_mcp_roll",       # 6
        "ring_mcp_pitch",      # 7
        "pinky_mcp_roll",      # 8
        "pinky_mcp_pitch",     # 9
    ]
    
    ctrl = np.zeros(10)
    for i, joint_name in enumerate(joint_names):
        if joint_name in required_joints:
            ctrl_idx = required_joints.index(joint_name)
            ctrl[ctrl_idx] = qpos[i]
    
    bounds = np.array([1.1339, 1.9189, 0.5146, 0.2181, 1.3607, 1.3607, 0.2181, 1.3607, 0.2181, 1.3607])
    ctrl_normalized = np.clip(ctrl / bounds, 0, 1)
    
    cmd = np.round(ctrl_normalized * 255).astype(int)
    cmd = cmd[[2, 1, 4, 5, 7, 9, 3, 6, 8, 0]]
    invert_mask = np.array([0, 255, 255, 255, 255, 255, 0, 0, 0, 0])
    cmd = np.abs(cmd - invert_mask)
    
    return cmd.tolist()


def main(
    hand_type: HandType = HandType.right,
    udp_port: int = 5006,
    finger_scaling: tuple = (1.0, 1.0, 0.8, 1.0, 1.2),
    finger_offset: tuple = (0.02, -0.01, -0.0, -0.0, -0.08),
    use_dexpilot: bool = False,
    can_interface: str = "can0",
    hand_joint: str = "L10",
    dry_run: bool = True,  
    test_finger: str = "all", 
):

    finger_to_motors = {
        "thumb": [0, 1, 9],      # 拇指: thumb_cmc_pitch, thumb_cmc_yaw, thumb_cmc_roll
        "index": [2, 6],         # 食指: index_mcp_pitch, index_mcp_roll
        "middle": [3],           # 中指: middle_mcp_pitch
        "ring": [4, 7],          # 无名指: ring_mcp_pitch, ring_mcp_roll
        "pinky": [5, 8],         # 小指: pinky_mcp_pitch, pinky_mcp_roll
    }
    
    linker_hand = None
    if not dry_run:
        if not LINKER_AVAILABLE:
            logger.error("LinkerHand SDK not available! Cannot control real hand.")
            return
            
        hand_type_str = "right" if hand_type == HandType.right else "left"
        logger.info(f"Initializing Linker Hand: {hand_type_str} {hand_joint} on {can_interface}")
        
        try:
            linker_hand = LinkerHandApi(
                hand_type=hand_type_str,
                hand_joint=hand_joint,
                can=can_interface
            )
            logger.info("Linker Hand initialized successfully")
            
            # 设置最大速度和力矩以减少延迟
            linker_hand.set_speed(speed=[255, 255, 255, 255, 255])  
            linker_hand.set_torque(torque=[200, 200, 200, 200, 200])  
            logger.info("Set motor speed to MAX (255) and torque to 200")
            
        except Exception as e:
            logger.error(f"Failed to initialize Linker Hand: {e}")
            return
    else:
        logger.info("=" * 80)
        logger.info("DRY RUN MODE: Will only print control data, not send to real hand")
        logger.info("To control real hand, use: --no-dry-run")
        logger.info("=" * 80)

    # 初始化 retargeting
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
    offset_vector = np.array(list(finger_offset), dtype=np.float32)
    logger.info(f"Scaling vector: {scaling_vector}")
    logger.info(f"Offset vector: {offset_vector}")

    ref_rot_fixed = R.from_euler('y', -90, degrees=True)

    frame_count = 0
    last_data_time = time.time()
    fps_counter = []
    fps_start_time = time.time()
    last_print_time = time.time()

    logger.info("Starting main loop...")
    
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
                logger.error(f"Error receiving data: {e}")

            if joint_pos is not None:
                last_data_time = time.time()
                try:
                    if use_dexpilot:
                        # DexPilot模式
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
                        # Vector模式
                        indices = retargeting.optimizer.target_link_human_indices
                        origin_indices = indices[0, :]
                        task_indices = indices[1, :]
                        ref_value = joint_pos[task_indices, :] - joint_pos[origin_indices, :]
                        qpos = retargeting.retarget(ref_value)

                    joint_to_finger = {
                        "thumb_cmc_roll": 0, "thumb_cmc_yaw": 0, "thumb_cmc_pitch": 0, "thumb_mcp": 0, "thumb_ip": 0,
                        "index_mcp_roll": 1, "index_mcp_pitch": 1, "index_pip": 1, "index_dip": 1,
                        "middle_mcp_pitch": 2, "middle_pip": 2, "middle_dip": 2,
                        "ring_mcp_roll": 3, "ring_mcp_pitch": 3, "ring_pip": 3, "ring_dip": 3,
                        "pinky_mcp_roll": 4, "pinky_mcp_pitch": 4, "pinky_pip": 4, "pinky_dip": 4,
                    }
                    
                    invert_joints = {
                        "thumb_cmc_roll": True,
                        "index_mcp_roll": True,
                    }
                    
                    scaled_qpos = qpos.copy()
                    for i, joint_name in enumerate(retargeting.joint_names):
                        value = qpos[i]
                        
                        if joint_name in joint_to_finger:
                            finger_idx = joint_to_finger[joint_name]
                            value = value * finger_scaling[finger_idx]
                        
                        if joint_name in invert_joints and invert_joints[joint_name]:
                            value = -value
                        
                        if joint_name in joint_to_finger:
                            finger_idx = joint_to_finger[joint_name]
                            value = value - finger_offset[finger_idx]
                        
                        scaled_qpos[i] = value
                    
                    motor_positions = qpos_to_motor_positions(scaled_qpos, retargeting.joint_names)
                    
                    if test_finger != "all":
                        filtered_positions = [255] * 10
                        for motor_idx in finger_to_motors[test_finger]:
                            filtered_positions[motor_idx] = motor_positions[motor_idx]
                        motor_positions = filtered_positions
                    
                    if time.time() - last_print_time >= 5.0:  # 每5秒打印一次
                        logger.info("=" * 80)
                        logger.info(f"Frame: {frame_count}")
                        if test_finger != "all":
                            logger.info(f"测试手指: {test_finger} (Motors: {finger_to_motors[test_finger]})")
                        
                        logger.info(f"Motor positions (0-255): {motor_positions}")
                        logger.info("=" * 80)
                        last_print_time = time.time()
                    
                    if not dry_run and linker_hand is not None:
                        linker_hand.finger_move(pose=motor_positions)

                    fps_counter.append(time.time())
                except Exception as e:
                    logger.error(f"Retargeting failed: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                if time.time() - last_data_time > 5.0:
                    logger.warning("No data received for 5 seconds")
                    last_data_time = time.time()

            # FPS统计
            if time.time() - fps_start_time >= 5.0:
                if fps_counter:
                    fps = len(fps_counter) / (time.time() - fps_start_time)
                    logger.info(f"Control FPS: {fps:.1f} Hz | Total frames: {frame_count}")
                    fps_counter = []
                    fps_start_time = time.time()
            
            # 移除 sleep，让循环尽可能快
            # time.sleep(0.001)  

    except KeyboardInterrupt:
        logger.info("Stopping...")
    finally:
        manus_receiver.close()
        logger.info("Closed")


if __name__ == "__main__":
    tyro.cli(main)
