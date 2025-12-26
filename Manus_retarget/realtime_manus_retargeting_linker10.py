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
    finger_scaling: tuple = (1.0, 1.2, 1.2, 1.2, 1.5),
):
    """Manus到Gaia手实时Retargeting"""
    model_path = Path(__file__).parent.parent / "anytwist/model/robot/l10_right.xml"
    if not model_path.exists():
        logger.error(f"MuJoCo model not found: {model_path}")
        return

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    hand_type_str = "right" if hand_type == HandType.right else "left"
    config_dir = Path(__file__).parent.parent / "dex-retargeting/src/dex_retargeting/configs/teleop"
    config_path = config_dir / f"linker_hand_{hand_type_str}_manus.yml"

    urdf_dir = Path(__file__).parent.parent / "dex-retargeting/assets/dex-urdf/robots/hands"
    RetargetingConfig.set_default_urdf_dir(str(urdf_dir))

    config = RetargetingConfig.load_from_file(str(config_path))
    retargeting = config.build()
    logger.info("Retargeting initialized")

    manus_receiver = ManusUDPReceiver(port=udp_port)
    scaling_vector = np.array(list(finger_scaling) + list(finger_scaling), dtype=np.float32)

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
                    indices = retargeting.optimizer.target_link_human_indices
                    origin_indices = indices[0, :]
                    task_indices = indices[1, :]
                    ref_value = joint_pos[task_indices, :] - joint_pos[origin_indices, :]
                    ref_value = ref_value * scaling_vector[:, np.newaxis]
                    qpos = retargeting.retarget(ref_value)

                    for i, joint_name in enumerate(retargeting.joint_names):
                        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                        if joint_id >= 0:
                            data.qpos[joint_id] = qpos[i]

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
