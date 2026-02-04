#!/usr/bin/env python3
"""
O6 手实时 Manus 手套 Retargeting 控制脚本

O6 手特点：
- 只有 6 个主动关节（2 个拇指 + 4 个手指 MCP）
- 其他关节通过 mimic 约束自动跟随
- 结构简单，控制直接
"""
import sys
import time
from pathlib import Path
import numpy as np
import tyro
from loguru import logger
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).parent.parent / "dex-retargeting" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "Manus_retarget"))
sys.path.insert(0, str(Path(__file__).parent.parent / "linker_hand_python_sdk"))

from dex_retargeting.constants import HandType
from dex_retargeting.retargeting_config import RetargetingConfig
from manus_skeleton_parser import ManusSkeletonParser
import socket

try:
    from LinkerHand.linker_hand_api import LinkerHandApi
    LINKER_AVAILABLE = True
except ImportError:
    LINKER_AVAILABLE = False
    logger.warning("LinkerHand SDK not available")


class ManusUDPReceiver:
    """接收 Manus 手套的 UDP 数据"""
    def __init__(self, port=5006):
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("0.0.0.0", self.port))
        self.sock.settimeout(0.01)
        self.parser = ManusSkeletonParser()

    def close(self):
        self.sock.close()


def qpos_to_o6_motor_positions(qpos_dict, debug=False):
    """
    将关节角度转换为 O6 电机位置
    
    O6 的 6 个主动关节映射：
    0: thumb_cmc_yaw
    1: thumb_cmc_pitch
    2: index_mcp_pitch
    3: middle_mcp_pitch
    4: ring_mcp_pitch
    5: pinky_mcp_pitch
    """
    # O6 的 6 个主动关节
    required_joints = [
        "thumb_cmc_yaw",
        "thumb_cmc_pitch",
        "index_mcp_pitch",
        "middle_mcp_pitch",
        "ring_mcp_pitch",
        "pinky_mcp_pitch",
    ]
    
    ctrl = np.zeros(6)
    
    for i, joint_name in enumerate(required_joints):
        ctrl[i] = qpos_dict.get(joint_name, 0.0)
    
    bounds = np.array([
        1.3,    # thumb_cmc_yaw: 0 to 1.3
        0.58,   # thumb_cmc_pitch: 0 to 0.58
        1.60,   # index_mcp_pitch: 0 to 1.60
        1.60,   # middle_mcp_pitch: 0 to 1.60
        1.60,   # ring_mcp_pitch: 0 to 1.60
        1.60,   # pinky_mcp_pitch: 0 to 1.60
    ])
    
    # 归一化到 0-1
    ctrl_normalized = np.clip(ctrl / bounds, 0, 1)
    
    # 转换为 0-255 的电机指令
    cmd = np.round(ctrl_normalized * 255).astype(int)
    
    # 根据实际电机方向调整（可能需要根据实际情况调整）
    invert_mask = np.array([255, 0, 255, 255, 255, 0])
    cmd = np.abs(cmd - invert_mask)
    
    return cmd.tolist()


def main(
    hand_type: HandType = HandType.left,
    udp_port: int = 5006,
    use_dexpilot: bool = False,
    can_interface: str = "can1",
    hand_joint: str = "O6",
    dry_run: bool = False,
    scaling_factor: float = 1.0,
    thumb_scale: float = 1.0,
    index_scale: float = 1.0,
    middle_scale: float = 2.0,
    ring_scale: float = 2.0,
    pinky_scale: float = 5.0,
):
    """
    O6 手实时 Retargeting 主函数
    
    Args:
        hand_type: 左手或右手
        udp_port: Manus 手套 UDP 端口
        use_dexpilot: 使用 DexPilot 还是 Vector retargeting
        can_interface: CAN 接口名称
        hand_joint: 手的型号（O6）
        dry_run: 测试模式，不发送实际指令
        scaling_factor: 整体缩放因子
        thumb_scale: 拇指缩放因子
        index_scale: 食指缩放因子
        middle_scale: 中指缩放因子
        ring_scale: 无名指缩放因子
        pinky_scale: 小指缩放因子
    """
    # 初始化 Linker Hand API
    linker_hand = None
    if not dry_run:
        if not LINKER_AVAILABLE:
            logger.error("LinkerHand SDK not available")
            return
            
        try:
            hand_type_str = "right" if hand_type == HandType.right else "left"
            linker_hand = LinkerHandApi(
                hand_type=hand_type_str,
                hand_joint=hand_joint,
                can=can_interface
            )
            logger.info(f"O6 {hand_type_str} hand initialized on {can_interface}")
            
            # 设置速度和扭矩（O6 只有 6 个电机）
            linker_hand.set_speed(speed=[255, 255, 255, 255, 255, 255])
            linker_hand.set_torque(torque=[200, 200, 200, 200, 200, 200])
        except Exception as e:
            logger.error(f"Failed to initialize O6 hand: {e}")
            return
    else:
        logger.info("Dry run mode - no actual motor commands will be sent")

    # 加载 O6 retargeting 配置
    hand_type_str = "right" if hand_type == HandType.right else "left"
    config_dir = Path(__file__).parent.parent / "dex-retargeting/src/dex_retargeting/configs/teleop"
    
    if use_dexpilot:
        config_path = config_dir / f"o6_hand_{hand_type_str}_dexpilot.yml"
        logger.info("Using DexPilot retargeting mode")
    else:
        config_path = config_dir / f"o6_hand_{hand_type_str}.yml"
        logger.info("Using Vector retargeting mode")

    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        return

    # 设置 URDF 目录
    urdf_dir = Path(__file__).parent.parent / "dex-retargeting/assets/dex-urdf/robots/hands"
    RetargetingConfig.set_default_urdf_dir(str(urdf_dir.absolute()))

    # 加载配置并创建 retargeting 对象
    config = RetargetingConfig.load_from_file(str(config_path))
    retargeting = config.build()
    logger.info(f"O6 retargeting initialized with {len(retargeting.joint_names)} joints")
    logger.info(f"Joint names: {retargeting.joint_names}")

    # 初始化 Manus UDP 接收器
    manus_receiver = ManusUDPReceiver(port=udp_port)
    logger.info(f"Listening for Manus data on UDP port {udp_port}")
    
    # 参考旋转（用于坐标系转换）
    ref_rot_fixed = R.from_euler('y', -90, degrees=True)
    
    # 性能统计
    frame_count = 0
    fps_counter = []
    fps_start_time = time.time()
    last_print_time = time.time()
    last_data_time = time.time()

    try:
        logger.info("Starting retargeting loop... Press Ctrl+C to stop")
        
        while True:
            frame_count += 1
            joint_pos = None

            # 接收 Manus 数据
            try:
                data_raw, _ = manus_receiver.sock.recvfrom(4096)
                nodes = manus_receiver.parser.parse_udp_data(data_raw)
                
                if len(nodes) >= 25:
                    joint_pos = np.zeros((25, 3), dtype=np.float32)
                    
                    # 获取手腕作为参考点
                    parent_quat = nodes[0]['rotation']
                    parent_pos = nodes[0]['position']
                    parent_rot = R.from_quat(parent_quat)
                    transform_rot = ref_rot_fixed * parent_rot.inv()

                    # 转换所有关节位置到相对坐标系
                    for i, node in enumerate(nodes[:25]):
                        joint_pos[i] = transform_rot.apply(node['position'] - parent_pos)
                        
            except socket.timeout:
                # 超时是正常的，继续循环
                pass
            except Exception as e:
                logger.error(f"Error receiving Manus data: {e}")

            # 如果收到数据，进行 retargeting
            if joint_pos is not None:
                last_data_time = time.time()
                
                try:
                    # 根据 retargeting 类型准备输入数据
                    if use_dexpilot:
                        # DexPilot: 使用指尖之间的向量
                        fingertip_indices = [4, 8, 12, 16, 20]  # 5 个指尖
                        fingertip_pos = joint_pos[fingertip_indices, :]
                        wrist_pos = joint_pos[0, :]
                        
                        ref_vectors = []
                        for i in range(len(fingertip_indices)):
                            for j in range(i + 1, len(fingertip_indices)):
                                ref_vectors.append(fingertip_pos[j] - fingertip_pos[i])
                        # 指尖到手腕的向量
                        for i in range(len(fingertip_indices)):
                            ref_vectors.append(fingertip_pos[i] - wrist_pos)
                        
                        ref_value = np.array(ref_vectors, dtype=np.float32)
                    else:
                        # Vector: 使用手腕到指尖的向量
                        indices = retargeting.optimizer.target_link_human_indices
                        origin_indices = indices[0, :]  # 都是 0（手腕）
                        task_indices = indices[1, :]    # [4, 8, 12, 16, 20]（5 个指尖）
                        ref_value = joint_pos[task_indices, :] - joint_pos[origin_indices, :]
                    
                    # 应用整体缩放
                    ref_value = ref_value * scaling_factor
                    
                    # 执行 retargeting
                    qpos = retargeting.retarget(ref_value)
                    
                    # 构建关节角度字典 - 只使用主动关节，忽略 mimic 关节
                    qpos_dict = {}
                    for i, joint_name in enumerate(retargeting.joint_names):
                        # 跳过 mimic 关节（dip, ip）
                        if joint_name.endswith("_dip") or joint_name.endswith("_ip"):
                            continue
                            
                        value = qpos[i]
                        
                        # 应用每个手指的独立缩放
                        if joint_name.startswith("thumb"):
                            value = value * thumb_scale
                        elif joint_name.startswith("index"):
                            value = value * index_scale
                        elif joint_name.startswith("middle"):
                            value = value * middle_scale
                        elif joint_name.startswith("ring"):
                            value = value * ring_scale
                        elif joint_name.startswith("pinky"):
                            value = value * pinky_scale
                        
                        qpos_dict[joint_name] = value
                    
                    # 转换为电机位置
                    motor_positions = qpos_to_o6_motor_positions(qpos_dict)
                    
                    # 定期打印调试信息
                    if time.time() - last_print_time >= 2.0:
                        logger.info("=" * 60)
                        logger.info("Joint angles (rad):")
                        for joint_name in qpos_dict.keys():
                            logger.info(f"  {joint_name}: {qpos_dict[joint_name]:.3f}")
                        logger.info(f"Motor positions (0-255): {motor_positions}")
                        logger.info(f"  [0]thumb_yaw: {motor_positions[0]}, [1]thumb_pitch: {motor_positions[1]}")
                        logger.info(f"  [2]index: {motor_positions[2]}, [3]middle: {motor_positions[3]}")
                        logger.info(f"  [4]ring: {motor_positions[4]}, [5]pinky: {motor_positions[5]}")
                        logger.info("=" * 60)
                        last_print_time = time.time()
                    
                    # 发送电机指令
                    if not dry_run and linker_hand is not None:
                        linker_hand.finger_move(pose=motor_positions)
                    
                    # 统计 FPS
                    fps_counter.append(time.time())
                    
                except Exception as e:
                    logger.error(f"Retargeting failed: {e}")
                    import traceback
                    traceback.print_exc()

            # 检查数据超时
            if time.time() - last_data_time > 2.0:
                if frame_count % 100 == 0:
                    logger.warning("No Manus data received for 2 seconds")

            # 计算并显示 FPS
            if time.time() - fps_start_time >= 10.0:
                if fps_counter:
                    fps = len(fps_counter) / (time.time() - fps_start_time)
                    logger.info(f"Retargeting FPS: {fps:.1f} Hz")
                    fps_counter = []
                    fps_start_time = time.time()

    except KeyboardInterrupt:
        logger.info("Stopping retargeting...")
    finally:
        # 清理资源
        manus_receiver.close()
        if linker_hand is not None:
            logger.info("Closing O6 hand connection")
        logger.info("Done")


if __name__ == "__main__":
    tyro.cli(main)

