#!/usr/bin/env python3
"""
Manus Skeleton Parser
解析从 Manus SDK 通过 UDP 发送的 25 节点骨骼数据
"""
import struct
import numpy as np
from typing import List, Dict, Tuple


class ManusSkeletonParser:
    """解析 Manus 手套的 25 节点骨骼数据"""
    
    def __init__(self):
        # Manus 25 节点的映射
        self.node_mapping = {
            'wrist': 0,
            'thumb': [1, 2, 3, 4],      # CMC, MCP, IP, Tip
            'index': [5, 6, 7, 8, 9],   # Metacarpal, Proximal, Intermediate, Distal, Tip
            'middle': [10, 11, 12, 13, 14],
            'ring': [15, 16, 17, 18, 19],
            'pinky': [20, 21, 22, 23, 24]
        }
        
        # MediaPipe 21 关键点索引
        # 0: 手腕
        # 1-4: 拇指 (CMC, MCP, IP, Tip)
        # 5-8: 食指 (MCP, PIP, DIP, Tip)
        # 9-12: 中指
        # 13-16: 无名指
        # 17-20: 小指
        
        # Manus 节点到 MediaPipe 的映射
        self.manus_to_mediapipe = {
            0: 0,   # 手腕
            # 拇指
            1: 1,   # CMC
            2: 2,   # MCP
            3: 3,   # IP
            4: 4,   # Tip
            # 食指 (跳过 Metacarpal node 5)
            6: 5,   # Proximal -> MCP
            7: 6,   # Intermediate -> PIP
            8: 7,   # Distal -> DIP
            9: 8,   # Tip
            # 中指
            11: 9,
            12: 10,
            13: 11,
            14: 12,
            # 无名指
            16: 13,
            17: 14,
            18: 15,
            19: 16,
            # 小指
            21: 17,
            22: 18,
            23: 19,
            24: 20
        }
    
    def parse_udp_data(self, data: bytes) -> List[Dict]:
        """
        解析 UDP 数据包
        
        格式: [node_count, node0_id, pos_x, pos_y, pos_z, rot_x, rot_y, rot_z, rot_w, node1_id, ...]
        
        Args:
            data: UDP 接收的字节数据
            
        Returns:
            节点列表，每个节点包含 id, position, rotation
        """
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
                'position': np.array([values[idx+1], values[idx+2], values[idx+3]], dtype=np.float32),
                'rotation': np.array([values[idx+4], values[idx+5], values[idx+6], values[idx+7]], dtype=np.float32)
            }
            nodes.append(node)
            idx += 8
        
        return nodes
    
    def to_mediapipe_format(self, nodes: List[Dict]) -> np.ndarray:
        """
        转换为 MediaPipe 21 关键点格式
        
        Args:
            nodes: Manus 节点列表
            
        Returns:
            shape (21, 3) 的关键点位置数组
        """
        keypoints = np.zeros((21, 3), dtype=np.float32)
        
        # 设置固定的手腕位置作为参考点（不使用 node 0）
        keypoints[0] = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # 手腕固定在原点
        
        for manus_idx, mediapipe_idx in self.manus_to_mediapipe.items():
            # 跳过手腕节点（node 0）
            if manus_idx == 0:
                continue
                
            if manus_idx < len(nodes):
                keypoints[mediapipe_idx] = nodes[manus_idx]['position']
        
        return keypoints
    
    def build_reference_vectors(self, keypoints: np.ndarray) -> np.ndarray:
        """
        构建用于 dex-retargeting 的参考向量
        
        对于 Vector Retargeting，需要从 origin 到 task 的向量
        
        Args:
            keypoints: MediaPipe 格式的关键点 (21, 3)
            
        Returns:
            参考向量数组，用于 retargeting
        """
        palm = keypoints[0]  
        
        vectors = []
        
        tip_indices = [4, 8, 12, 16, 20]  # 拇指、食指、中指、无名指、小指尖
        for idx in tip_indices:
            vectors.append(keypoints[idx] - palm)

        middle_indices = [2, 6, 10, 14, 18]  
        for idx in middle_indices:
            vectors.append(keypoints[idx] - palm)
        
        return np.array(vectors, dtype=np.float32)
    
    def get_finger_joint_angles(self, nodes: List[Dict]) -> Dict[str, float]:
        """
        从节点旋转计算手指关节角度（简化版本）
        
        Args:
            nodes: Manus 节点列表
            
        Returns:
            关节角度字典
        """
        angles = {}
        
        for finger_name, node_indices in self.node_mapping.items():
            if finger_name == 'wrist':
                continue
                
            for i, node_idx in enumerate(node_indices):
                if node_idx < len(nodes):
                    quat = nodes[node_idx]['rotation']
                    # 简单地使用四元数的某个分量作为角度（不准确）
                    # 实际应该计算相对旋转
                    angles[f"{finger_name}_joint_{i}"] = quat[1]  # x 分量
        
        return angles
    
    def visualize_skeleton(self, nodes: List[Dict]) -> str:
        lines = ["Manus Skeleton (25 nodes):"]
        lines.append(f"  Wrist (0): pos={nodes[0]['position']}")
        
        for finger_name, node_indices in self.node_mapping.items():
            if finger_name == 'wrist':
                continue
            lines.append(f"  {finger_name.capitalize()}:")
            joint_names = ['Metacarpal', 'Proximal', 'Intermediate', 'Distal', 'Tip']
            if finger_name == 'thumb':
                joint_names = ['CMC', 'MCP', 'IP', 'Tip']
            
            for i, node_idx in enumerate(node_indices):
                if node_idx < len(nodes):
                    pos = nodes[node_idx]['position']
                    lines.append(f"    {joint_names[i] if i < len(joint_names) else 'Unknown'} ({node_idx}): pos={pos}")
        
        return "\n".join(lines)


def test_parser():
    import socket
    
    parser = ManusSkeletonParser()
    
    # 创建 UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("", 5006))
    sock.settimeout(5.0)
    
    print("等待 Manus 数据...")
    
    try:
        data, addr = sock.recvfrom(4096)
        print(f"接收到数据: {len(data)} 字节")
        
        nodes = parser.parse_udp_data(data)
        print(f"\n解析到 {len(nodes)} 个节点")

        print("\n" + parser.visualize_skeleton(nodes))
        
        keypoints = parser.to_mediapipe_format(nodes)
        print(f"\nMediaPipe 关键点形状: {keypoints.shape}")
        print(f"手腕位置: {keypoints[0]}")
        print(f"中指尖位置: {keypoints[12]}")
        
        ref_vectors = parser.build_reference_vectors(keypoints)
        print(f"\n参考向量形状: {ref_vectors.shape}")
        print(f"到中指尖的向量: {ref_vectors[2]}")
        
    except socket.timeout:
        print("超时：未接收到数据")
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        sock.close()


if __name__ == "__main__":
    test_parser()
