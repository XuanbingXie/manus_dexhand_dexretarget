#!/usr/bin/env python3
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
    
    ctrl_normalized = np.clip(ctrl / bounds, 0, 1)
    cmd = np.round(ctrl_normalized * 255).astype(int)
    
    invert_mask = np.array([255, 255, 255, 255, 0, 0])
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
    thumb_scale: float = 1.4,
    index_scale: float = 2.0,
    middle_scale: float = 3.0,
    ring_scale: float = 2.0,
    pinky_scale: float = 5.0,
):
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
            linker_hand.set_speed(speed=[255, 255, 255, 255, 255, 255])
            linker_hand.set_torque(torque=[200, 200, 200, 200, 200, 200])
        except Exception as e:
            logger.error(f"Failed to initialize O6 hand: {e}")
            return
    else:
        logger.info("Dry run mode - no actual motor commands will be sent")

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

    urdf_dir = Path(__file__).parent.parent / "dex-retargeting/assets/dex-urdf/robots/hands"
    RetargetingConfig.set_default_urdf_dir(str(urdf_dir.absolute()))

    config = RetargetingConfig.load_from_file(str(config_path))
    retargeting = config.build()
    
    active_joint_names = retargeting.optimizer.robot.dof_joint_names
    logger.info(f"O6 retargeting initialized with {len(active_joint_names)} active joints")
    logger.info(f"Active joint names: {active_joint_names}")

    manus_receiver = ManusUDPReceiver(port=udp_port)
    logger.info(f"Listening for Manus data on UDP port {udp_port}")
    
    ref_rot_fixed = R.from_euler('y', -90, degrees=True)

    frame_count = 0
    fps_counter = []
    fps_start_time = time.time()
    last_print_time = time.time()
    last_data_time = time.time()

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
                logger.error(f"Error receiving Manus data: {e}")

            if joint_pos is not None:
                last_data_time = time.time()
                
                try:
                    if use_dexpilot:
                        fingertip_indices = [4, 8, 12, 16, 20]  # 5 个指尖
                        fingertip_pos = joint_pos[fingertip_indices, :]
                        wrist_pos = joint_pos[0, :]
                        
                        ref_vectors = []
                        for i in range(len(fingertip_indices)):
                            for j in range(i + 1, len(fingertip_indices)):
                                ref_vectors.append(fingertip_pos[j] - fingertip_pos[i])
                        for i in range(len(fingertip_indices)):
                            ref_vectors.append(fingertip_pos[i] - wrist_pos)
                        
                        ref_value = np.array(ref_vectors, dtype=np.float32)
                    else:
                        indices = retargeting.optimizer.target_link_human_indices
                        origin_indices = indices[0, :]  
                        task_indices = indices[1, :]   
                        ref_value = joint_pos[task_indices, :] - joint_pos[origin_indices, :]
                        
                    ref_value = ref_value * scaling_factor
                    qpos = retargeting.retarget(ref_value)
                    
                    active_joint_names = retargeting.optimizer.robot.dof_joint_names
                    qpos_dict = {}
                    for i, joint_name in enumerate(active_joint_names):
                        value = qpos[i]

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
                    
                    if len(nodes) >= 25:
                        ring_quat = nodes[16]['rotation']  # [x, y, z, w]
                        from scipy.spatial.transform import Rotation as R_scipy
                        ring_rot = R_scipy.from_quat(ring_quat)
                        ring_euler = ring_rot.as_euler('xyz', degrees=False)
                        ring_angle = abs(ring_euler[1])  # pitch 角度
                        qpos_dict['ring_mcp_pitch'] = np.clip(ring_angle * ring_scale, 0, 1.60)
                        
                        pinky_quat = nodes[21]['rotation']
                        pinky_rot = R_scipy.from_quat(pinky_quat)
                        pinky_euler = pinky_rot.as_euler('xyz', degrees=False)
                        pinky_angle = abs(pinky_euler[1])  # pitch 角度
                        qpos_dict['pinky_mcp_pitch'] = np.clip(pinky_angle * pinky_scale, 0, 1.60)
                    
                    motor_positions = qpos_to_o6_motor_positions(qpos_dict)
                    
                    if time.time() - last_print_time >= 3.0:
                        logger.info("Joints(rad): thumb_y={:.2f} thumb_p={:.2f} | idx={:.2f} mid={:.2f} ring={:.2f} pinky={:.2f}".format(
                            qpos_dict.get('thumb_cmc_yaw', 0), qpos_dict.get('thumb_cmc_pitch', 0),
                            qpos_dict.get('index_mcp_pitch', 0), qpos_dict.get('middle_mcp_pitch', 0),
                            qpos_dict.get('ring_mcp_pitch', 0), qpos_dict.get('pinky_mcp_pitch', 0)))
                        logger.info(f"Motors: {motor_positions}")
                        last_print_time = time.time()
                    
                    if not dry_run and linker_hand is not None:
                        linker_hand.finger_move(pose=motor_positions)
                    
                    fps_counter.append(time.time())
                    
                except Exception as e:
                    logger.error(f"Retargeting failed: {e}")
                    import traceback
                    traceback.print_exc()

            if time.time() - last_data_time > 2.0:
                if frame_count % 100 == 0:
                    logger.warning("No Manus data received for 2 seconds")

            if time.time() - fps_start_time >= 10.0:
                if fps_counter:
                    fps = len(fps_counter) / (time.time() - fps_start_time)
                    logger.info(f"Retargeting FPS: {fps:.1f} Hz")
                    fps_counter = []
                    fps_start_time = time.time()

    except KeyboardInterrupt:
        logger.info("Stopping retargeting...")
    finally:
        manus_receiver.close()
        if linker_hand is not None:
            logger.info("Closing O6 hand connection")
        logger.info("Done")


if __name__ == "__main__":
    tyro.cli(main)

