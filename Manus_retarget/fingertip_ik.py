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
    
    def solve_ik_jacobian(self, finger_name, target_pos, initial_guess=None, max_iter=50, tolerance=1e-3, damping=0.1):
        if initial_guess is None:
            initial_guess = np.zeros(len(self.finger_joints[finger_name]))
        
        q = initial_guess.copy()
        bounds = self.joint_limits[finger_name]
        
        for iteration in range(max_iter):
            self.set_finger_joints(finger_name, q)
            current_pos = self.get_fingertip_position(finger_name)
            
            error = target_pos - current_pos
            error_norm = np.linalg.norm(error)
            
            if error_norm < tolerance:
                return q, True
            
            epsilon = 1e-6
            jacobian = np.zeros((3, len(q)))
            
            for i in range(len(q)):
                q_plus = q.copy()
                q_plus[i] += epsilon
                
                q_plus[i] = np.clip(q_plus[i], bounds[i][0], bounds[i][1])
                
                self.set_finger_joints(finger_name, q_plus)
                pos_plus = self.get_fingertip_position(finger_name)
                
                jacobian[:, i] = (pos_plus - current_pos) / epsilon
            
            JtJ = jacobian.T @ jacobian
            damping_matrix = damping * np.eye(len(q))
            
            try:
                delta_q = np.linalg.solve(JtJ + damping_matrix, jacobian.T @ error)
            except np.linalg.LinAlgError:
                delta_q = np.linalg.pinv(jacobian) @ error
            
            alpha = 1.0
            q_new = q + alpha * delta_q
            for i in range(len(q_new)):
                q_new[i] = np.clip(q_new[i], bounds[i][0], bounds[i][1])
            
            q = q_new
        self.set_finger_joints(finger_name, q)
        final_pos = self.get_fingertip_position(finger_name)
        final_error = np.linalg.norm(target_pos - final_pos)
        
        return q, final_error < tolerance * 2
    
    def solve_ik(self, finger_name, target_pos, initial_guess=None, weight_collision=10.0, weight_regularization=0.001):
        if initial_guess is None:
            initial_guess = np.zeros(len(self.finger_joints[finger_name]))
        
        q_jacobian, success_jacobian = self.solve_ik_jacobian(finger_name, target_pos, initial_guess)
        
        if success_jacobian:
            return q_jacobian, True
        
        bounds = self.joint_limits[finger_name]
        
        def objective(q):
            self.set_finger_joints(finger_name, q)
            current_pos = self.get_fingertip_position(finger_name)
            
            pos_error = np.linalg.norm(current_pos - target_pos)
            
            collision_penalty = 0.0
            if self.check_collision():
                collision_penalty = weight_collision
            
            regularization = weight_regularization * np.sum((q - initial_guess) ** 2)
            
            return pos_error + collision_penalty + regularization
        
        best_result = None
        best_error = float('inf')
        
        result = minimize(
            objective,
            initial_guess,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 100, 'ftol': 1e-5}
        )
        
        if result.fun < best_error:
            best_error = result.fun
            best_result = result

        if best_error > 0.01:
            mid_guess = np.array([(b[0] + b[1]) / 2 for b in bounds])
            result2 = minimize(
                objective,
                mid_guess,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-5}
            )
            
            if result2.fun < best_error:
                best_error = result2.fun
                best_result = result2
        
        return best_result.x, best_result.success
    
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
