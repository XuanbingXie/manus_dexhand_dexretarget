#!/usr/bin/env python3
"""
调试脚本：理解手腕旋转如何影响向量
"""
import socket
import struct
import numpy as np
from scipy.spatial.transform import Rotation as R

def parse_manus_data(data):
    num_floats = len(data) // 4
    values = struct.unpack(f"!{num_floats}f", data)
    
    node_count = int(values[0])
    nodes = []
    
    idx = 1
    for i in range(node_count):
        if idx + 8 > len(values):
            break
        node = {
            'id': int(values[idx]),
            'position': np.array([values[idx+1], values[idx+2], values[idx+3]]),
            'rotation': np.array([values[idx+4], values[idx+5], values[idx+6], values[idx+7]])
        }
        nodes.append(node)
        idx += 8
    
    return nodes

def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("", 5006))
    sock.settimeout(0.1)
    
    print("等待Manus数据...")
    print("请保持手指姿势不变，只旋转手腕\n")
    
    frame_count = 0
    initial_wrist_rot = None
    
    while True:
        try:
            data, addr = sock.recvfrom(4096)
            nodes = parse_manus_data(data)
            
            if len(nodes) < 25:
                continue
            
            frame_count += 1
            
            # 每30帧打印一次
            if frame_count % 30 != 0:
                continue
            
            # 手腕信息
            wrist_pos = nodes[0]['position']
            wrist_quat = nodes[0]['rotation']
            wrist_rot = R.from_quat(wrist_quat)
            
            if initial_wrist_rot is None:
                initial_wrist_rot = wrist_rot
                print("=== 初始手腕旋转已捕获 ===\n")
            
            # 计算手腕旋转角度变化
            rot_diff = wrist_rot * initial_wrist_rot.inv()
            euler_diff = rot_diff.as_euler('xyz', degrees=True)
            
            # 选择几个关键点：食指尖(9)、中指尖(14)、小指尖(24)
            index_tip = nodes[9]['position']
            middle_tip = nodes[14]['position']
            pinky_tip = nodes[24]['position']
            
            # 计算向量（世界坐标系）
            vec_index = index_tip - wrist_pos
            vec_middle = middle_tip - wrist_pos
            vec_pinky = pinky_tip - wrist_pos
            
            # 计算向量长度
            len_index = np.linalg.norm(vec_index)
            len_middle = np.linalg.norm(vec_middle)
            len_pinky = np.linalg.norm(vec_pinky)
            
            # 将向量转换到初始手腕坐标系
            rot_compensation = initial_wrist_rot * wrist_rot.inv()
            vec_index_local = rot_compensation.apply(vec_index)
            vec_middle_local = rot_compensation.apply(vec_middle)
            vec_pinky_local = rot_compensation.apply(vec_pinky)
            
            print(f"=== Frame {frame_count} ===")
            print(f"手腕旋转变化 (度): X={euler_diff[0]:.1f}, Y={euler_diff[1]:.1f}, Z={euler_diff[2]:.1f}")
            print(f"\n世界坐标系向量:")
            print(f"  食指: {vec_index}, 长度: {len_index:.4f}")
            print(f"  中指: {vec_middle}, 长度: {len_middle:.4f}")
            print(f"  小指: {vec_pinky}, 长度: {len_pinky:.4f}")
            print(f"\n补偿后向量(初始手腕坐标系):")
            print(f"  食指: {vec_index_local}")
            print(f"  中指: {vec_middle_local}")
            print(f"  小指: {vec_pinky_local}")
            print()
            
        except socket.timeout:
            continue
        except KeyboardInterrupt:
            print("\n停止")
            break
    
    sock.close()

if __name__ == "__main__":
    main()
