#!/usr/bin/env python3
"""
混合控制：拇指用简单映射，四指用 vector retargeting
"""
import socket
import struct
import numpy as np
from scipy.spatial.transform import Rotation as R
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "dex-retargeting/src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "linker_hand_python_sdk"))

from dex_retargeting.constants import HandType
from dex_retargeting.retargeting_config import RetargetingConfig

try:
    from LinkerHand.linker_hand_api import LinkerHandApi
    LINKER_AVAILABLE = True
except ImportError:
    LINKER_AVAILABLE = False
    print("Warning: LinkerHand SDK not available")


UDP_ADDR = "0.0.0.0"
UDP_PORT = 9000

HEADER_FMT = "<III"
HEADER_SIZE = 12
NUM_SENSORS = 5
SENSOR_FLOATS = 7
WRIST_FLOATS = 4


def parse_packet(data):
    """解析 Manus Endpoint 数据包"""
    seq, left_id, right_id = struct.unpack_from(HEADER_FMT, data, 0)
    floats = np.frombuffer(data, dtype=np.float32, offset=HEADER_SIZE)
    
    offset = 0
    left_sensors = floats[offset:offset + NUM_SENSORS * SENSOR_FLOATS].reshape(NUM_SENSORS, SENSOR_FLOATS)
    offset += NUM_SENSORS * SENSOR_FLOATS
    left_wrist = floats[offset:offset + WRIST_FLOATS]
    offset += WRIST_FLOATS
    
    right_sensors = floats[offset:offset + NUM_SENSORS * SENSOR_FLOATS].reshape(NUM_SENSORS, SENSOR_FLOATS)
    offset += NUM_SENSORS * SENSOR_FLOATS
    right_wrist = floats[offset:offset + WRIST_FLOATS]
    
    return {
        'seq': seq,
        'left_id': left_id,
        'right_id': right_id,
        'left_sensors': left_sensors if left_id != 0 else None,
        'left_wrist': left_wrist if left_id != 0 else None,
        'right_sensors': right_sensors if right_id != 0 else None,
        'right_wrist': right_wrist if right_id != 0 else None,
    }


def create_four_finger_config(hand_type, scaling_factor):
    """创建只包含四指的 retargeting 配置"""
    config_content = f"""retargeting:
  type: vector
  urdf_path: o6_{hand_type}/linkerhand_o6_{hand_type}.urdf

  # 只优化四个手指的 MCP
  target_joint_names: [
    "index_mcp_pitch",
    "middle_mcp_pitch",
    "ring_mcp_pitch",
    "pinky_mcp_pitch"
  ]

  target_origin_link_names: [
    "hand_base_link", "hand_base_link", "hand_base_link", "hand_base_link"
  ]
  target_task_link_names: [
    "index_tip", "middle_tip", "ring_tip", "pinky_tip"
  ]
  
  scaling_factor: {scaling_factor}

  target_link_human_indices: [
    [0, 0, 0, 0],
    [1, 2, 3, 4]
  ]

  normal_delta: 0.005
  huber_delta: 0.02
  low_pass_alpha: 0.3
  ignore_mimic_joint: true
"""
    return config_content


def qpos_to_o6_motors(thumb_motor_0, thumb_motor_1, qpos_dict):
    """
    将关节角度转换为 O6 电机位置
    拇指电机直接传入，四指从 qpos_dict 计算
    """
    motors = [thumb_motor_0, thumb_motor_1]
    
    # 四指关节
    finger_joints = [
        "index_mcp_pitch",
        "middle_mcp_pitch",
        "ring_mcp_pitch",
        "pinky_mcp_pitch",
    ]
    
    for joint_name in finger_joints:
        angle = qpos_dict.get(joint_name, 0.0)
        # 归一化到 0-1
        normalized = np.clip(angle / 1.60, 0, 1)
        motor_val = int(normalized * 255)
        motors.append(motor_val)
    
    motors[2] = 255 - motors[2]
    # motors[3] = 255 - motors[3]
    
    return motors


def main(
    hand_type="left",
    can_interface="can0",
    dry_run=False,
    scaling_factor=1.0,
    index_scale=1.0,
    middle_scale=1.0,
    ring_scale=1.0,
    pinky_scale=1.0,
):
    """
    主函数
    
    Args:
        hand_type: "left" 或 "right"
        can_interface: CAN接口名称
        dry_run: 测试模式
        scaling_factor: 整体缩放因子
        index_scale: 食指缩放
        middle_scale: 中指缩放
        ring_scale: 无名指缩放
        pinky_scale: 小指缩放
    """
    linker_hand = None
    if not dry_run:
        if not LINKER_AVAILABLE:
            print("Error: LinkerHand SDK not available")
            return
        
        try:
            linker_hand = LinkerHandApi(
                hand_type=hand_type,
                hand_joint="O6",
                can=can_interface
            )
            print(f"O6 {hand_type} hand initialized on {can_interface}")
            linker_hand.set_speed(speed=[255] * 6)
            linker_hand.set_torque(torque=[200] * 6)
        except Exception as e:
            print(f"Failed to initialize O6 hand: {e}")
            return
    else:
        print("Dry run mode")
    
    # 创建四指 retargeting 配置
    config_dir = Path(__file__).parent.parent / "dex-retargeting/src/dex_retargeting/configs/teleop"
    config_path = config_dir / f"o6_four_fingers_{hand_type}.yml"
    
    config_content = create_four_finger_config(hand_type, scaling_factor)
    config_path.write_text(config_content)
    print(f"Created four-finger config: {config_path}")
    
    urdf_dir = Path(__file__).parent.parent / "dex-retargeting/assets/dex-urdf/robots/hands"
    RetargetingConfig.set_default_urdf_dir(str(urdf_dir.absolute()))
    
    config = RetargetingConfig.load_from_file(str(config_path))
    retargeting = config.build()
    
    print(f"Loaded vector retargeting for 4 fingers")
    
    # 坐标系转换
    ref_rot_fixed = R.from_euler('y', -90, degrees=True)
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_ADDR, UDP_PORT))
    sock.settimeout(0.01)
    
    print(f"Listening on {UDP_ADDR}:{UDP_PORT}")
    print("Hybrid mode: Thumb=simple, 4-fingers=vector retargeting")
    print("Press Ctrl+C to stop")
    
    last_print = time.time()
    frame_count = 0
    
    try:
        while True:
            try:
                data, _ = sock.recvfrom(2048)
                msg = parse_packet(data)
                
                sensors = msg['left_sensors'] if hand_type == "left" else msg['right_sensors']
                wrist_quat = msg['left_wrist'] if hand_type == "left" else msg['right_wrist']
                
                if sensors is not None and wrist_quat is not None:
                    thumb_pos = sensors[0, :3]
                    thumb_len = np.linalg.norm(thumb_pos)
                    thumb_yaw = int(np.clip((0.12 - thumb_len) / 0.03 * 255, 0, 255))
                    thumb_pitch = int(np.clip((0.12 - thumb_len) / 0.03 * 255, 0, 255))
                    thumb_yaw = 255 - thumb_yaw
                    thumb_pitch = 255 - thumb_pitch
                    
                    # === 四指：vector retargeting ===
                    four_finger_pos = sensors[1:5, :3]  # shape (4, 3)
                    
                    if np.linalg.norm(wrist_quat) < 0.01:
                        continue
                    
                    wrist_rot = R.from_quat([wrist_quat[1], wrist_quat[2], wrist_quat[3], wrist_quat[0]])
                    transform_rot = ref_rot_fixed * wrist_rot.inv()
                    transformed_pos = np.array([transform_rot.apply(pos) for pos in four_finger_pos])
                    
                    ref_value = transformed_pos * scaling_factor
                    
                    # 关节顺序（从 URDF）：thumb_cmc_yaw, thumb_cmc_pitch, thumb_ip, 
                    #                      index_mcp_pitch, index_dip, 
                    #                      middle_mcp_pitch, middle_dip,
                    #                      ring_mcp_pitch, ring_dip,
                    #                      pinky_mcp_pitch, pinky_dip
                    # 目标关节（4个）：index_mcp_pitch, middle_mcp_pitch, ring_mcp_pitch, pinky_mcp_pitch
                    # 固定关节（7个）：thumb_cmc_yaw, thumb_cmc_pitch, thumb_ip, index_dip, middle_dip, ring_dip, pinky_dip
                    thumb_yaw_rad = (thumb_yaw / 255.0) * 1.3  
                    thumb_pitch_rad = (thumb_pitch / 255.0) * 0.58
                    thumb_ip_rad = thumb_pitch_rad * 2.29  
                    fixed_qpos = np.array([
                        thumb_yaw_rad, thumb_pitch_rad, thumb_ip_rad,  
                        0.0, 0.0, 0.0, 0.0  # 四指的 DIP 关节（会被 mimic 关系覆盖）
                    ])
                    
                    qpos = retargeting.retarget(ref_value, fixed_qpos=fixed_qpos)
                    
                    active_joint_names = retargeting.optimizer.robot.dof_joint_names
                    qpos_dict = {}
                    scales = {
                        "index_mcp_pitch": index_scale,
                        "middle_mcp_pitch": middle_scale,
                        "ring_mcp_pitch": ring_scale,
                        "pinky_mcp_pitch": pinky_scale,
                    }
                    
                    for i, joint_name in enumerate(active_joint_names):
                        value = qpos[i] * scales.get(joint_name, 1.0)
                        qpos_dict[joint_name] = value
                    
                    motors = qpos_to_o6_motors(thumb_yaw, thumb_pitch, qpos_dict)
                    
                    if not dry_run and linker_hand is not None:
                        linker_hand.finger_move(pose=motors)
                    
                    frame_count += 1
                    if time.time() - last_print >= 2.0:
                        print(f"Frame {frame_count} | Motors: {motors}")
                        last_print = time.time()
                
            except socket.timeout:
                pass
            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()
                
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        sock.close()
        if linker_hand is not None:
            print("Closing hand connection")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--hand", choices=["left", "right"], default="left")
    parser.add_argument("--can", default="can0")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--scaling", type=float, default=1.0, help="整体缩放因子")
    parser.add_argument("--index-scale", type=float, default=1.0)
    parser.add_argument("--middle-scale", type=float, default=1.0)
    parser.add_argument("--ring-scale", type=float, default=1.0)
    parser.add_argument("--pinky-scale", type=float, default=1.0)
    args = parser.parse_args()
    
    main(
        hand_type=args.hand,
        can_interface=args.can,
        dry_run=args.dry_run,
        scaling_factor=args.scaling,
        index_scale=args.index_scale,
        middle_scale=args.middle_scale,
        ring_scale=args.ring_scale,
        pinky_scale=args.pinky_scale,
    )
