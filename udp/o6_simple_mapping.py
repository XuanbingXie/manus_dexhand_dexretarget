#!/usr/bin/env python3
"""
简单直接的映射：从传感器位置长度计算手指弯曲
位置长度越短 = 手指越弯曲
"""
import socket
import struct
import numpy as np
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "linker_hand_python_sdk"))

try:
    from LinkerHand.linker_hand_api import LinkerHandApi
    LINKER_AVAILABLE = True
except ImportError:
    LINKER_AVAILABLE = False
    print("Warning: LinkerHand SDK not available")


UDP_ADDR = "127.0.0.1"
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
        'right_sensors': right_sensors if right_id != 0 else None,
    }


def sensors_to_motors_simple(sensors):
    """
    简单映射：位置长度 → 弯曲角度 → 电机指令
    
    观察到的位置长度范围：0.09 - 0.12 米
    - 长度大（~0.12）= 手指伸直
    - 长度小（~0.09）= 手指弯曲
    """
    motors = []
    
    # 拇指（两个电机）
    thumb_pos = sensors[0, :3]
    thumb_len = np.linalg.norm(thumb_pos)
    # 映射：0.09-0.12 → 0-255
    thumb_yaw = int(np.clip((0.12 - thumb_len) / 0.03 * 255, 0, 255))
    thumb_pitch = int(np.clip((0.12 - thumb_len) / 0.03 * 255, 0, 255))
    motors.append(thumb_yaw)
    motors.append(thumb_pitch)
    
    # 其他四个手指
    for i in range(1, 5):
        pos = sensors[i, :3]
        length = np.linalg.norm(pos)
        # 映射：0.09-0.12 → 0-255
        motor_val = int(np.clip((0.12 - length) / 0.03 * 255, 0, 255))
        motors.append(motor_val)
    
    # 反转方向
    motors[0] = 255 - motors[0]
    motors[1] = 255 - motors[1]
    motors[2] = 255 - motors[2]
    motors[3] = 255 - motors[3]
    
    return motors


def main(hand_type="right", can_interface="can1", dry_run=False):
    """主函数"""
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
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_ADDR, UDP_PORT))
    sock.settimeout(0.01)
    
    print(f"Listening on {UDP_ADDR}:{UDP_PORT}")
    print("Simple mapping: position length → motor command")
    print("Press Ctrl+C to stop")
    
    last_print = time.time()
    frame_count = 0
    
    try:
        while True:
            try:
                data, _ = sock.recvfrom(2048)
                msg = parse_packet(data)
                
                sensors = msg['left_sensors'] if hand_type == "left" else msg['right_sensors']
                
                if sensors is not None:
                    # 简单映射
                    motors = sensors_to_motors_simple(sensors)
                    
                    # 发送指令
                    if not dry_run and linker_hand is not None:
                        linker_hand.finger_move(pose=motors)
                    
                    # 定期打印
                    frame_count += 1
                    if time.time() - last_print >= 1.0:
                        lengths = [np.linalg.norm(sensors[i, :3]) for i in range(5)]
                        print(f"Frame {frame_count} | Lengths: {[f'{l:.3f}' for l in lengths]} | Motors: {motors}")
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
    args = parser.parse_args()
    
    main(hand_type=args.hand, can_interface=args.can, dry_run=args.dry_run)
