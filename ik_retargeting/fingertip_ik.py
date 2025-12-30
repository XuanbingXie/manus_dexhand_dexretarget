#!/usr/bin/env python3
import numpy as np
import mujoco
from scipy.optimize import minimize
from pathlib import Path


class FingerIK:
    def __init__(self, model_path, low_pass_alpha=0.3):
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self.low_pass_alpha = low_pass_alpha
        
        self.fingertip_sites = {
            "thumb": "thumb_distal",
            "index": "index_distal", 
            "middle": "middle_distal",
            "ring": "ring_distal",
            "pinky": "pinky_distal"
        }
        
        self.finger_joints = {
            "thumb": ["thumb_cmc_roll", "thumb_cmc_yaw", "thumb_cmc_pitch"],
            "index": ["index_mcp_roll", "index_mcp_pitch"],
            "middle": ["middle_mcp_pitch"],
            "ring": ["ring_mcp_roll", "ring_mcp_pitch"],
            "pinky": ["pinky_mcp_roll", "pinky_mcp_pitch"]
        }
        
        self.joint_limits = {}
        for finger, joints in self.finger_joints.items():
            self.joint_limits[finger] = []
            for joint_name in joints:
                joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                jnt_range = self.model.jnt_range[joint_id]
                self.joint_limits[finger].append((jnt_range[0], jnt_range[1]))
        
        self.collision_pairs = self._build_collision_pairs()
        
        # 存储上一帧的解，用于平滑
        self.prev_solutions = {}
    
    def _build_collision_pairs(self):
        pairs = []
        geom_names = []
        for i in range(self.model.ngeom):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i)
            if name and "distal" in name:
                geom_names.append((i, name))
        
        for i, (id1, name1) in enumerate(geom_names):
            for id2, name2 in geom_names[i+1:]:
                pairs.append((id1, id2))
        
        return pairs
    
    def get_fingertip_position(self, finger_name):
        geom_name = self.fingertip_sites[finger_name]
        geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        return self.data.geom_xpos[geom_id].copy()
    
    def set_finger_joints(self, finger_name, joint_values):
        for joint_name, value in zip(self.finger_joints[finger_name], joint_values):
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            self.data.qpos[joint_id] = value
        mujoco.mj_forward(self.model, self.data)
    
    def check_collision(self):
        mujoco.mj_forward(self.model, self.data)
        
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            if contact.dist < -0.001:
                return True
        return False
    
    def solve_ik(self, finger_name, target_pos, initial_guess=None, weight_collision=10.0):
        if initial_guess is None:
            initial_guess = np.zeros(len(self.finger_joints[finger_name]))
        
        bounds = self.joint_limits[finger_name]
        
        def objective(q):
            self.set_finger_joints(finger_name, q)
            current_pos = self.get_fingertip_position(finger_name)
            
            pos_error = np.linalg.norm(current_pos - target_pos)
            
            collision_penalty = 0.0
            if self.check_collision():
                collision_penalty = weight_collision
            
            return pos_error + collision_penalty
        
        result = minimize(
            objective,
            initial_guess,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 100, 'ftol': 1e-4}
        )
        
        return result.x, result.success
    
    def solve_all_fingers(self, target_positions, initial_qpos=None):
        if initial_qpos is None:
            initial_qpos = {}
            for finger in self.finger_joints.keys():
                initial_qpos[finger] = np.zeros(len(self.finger_joints[finger]))
        
        results = {}
        for finger_name, target_pos in target_positions.items():
            qpos, success = self.solve_ik(finger_name, target_pos, initial_qpos.get(finger_name))
            results[finger_name] = {
                "qpos": qpos,
                "success": success
            }
        
        return results
