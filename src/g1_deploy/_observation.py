import numpy as np
import joblib

from g1_deploy.utils.math import (
    quat_apply,
    quat_conjugate,
    quat_inv,
    quat_mul,
    quat_rotate_inverse,
    yaw_quat,
    subtract_frame_transforms
)
from typing import List, Optional,Any
from pathlib import Path
from g1_deploy.base import Articulation, RobotInterface


class RefMotion:
    def __init__(self, motion_file):
        motion_file = Path(__file__).parent.parent / "checkpoints" / motion_file 
        with open(motion_file, "rb") as f:
            motions = joblib.load(f)
            motion_name = list(motions.keys())[0]
            print("Loading motion: ", motion_name)
            motion = motions[motion_name]
            self.fps = motion["fps"]
            self.joint_pos = motion["joint_pos"]        # (T, num_joints)
            self.joint_vel = motion["joint_vel"]        # (T, num_joints)
            self.body_pos_w = motion["body_pos_w"]    # (T, num_bodies, 3)
            self.body_quat_w = motion["body_quat_w"]    # (T, num_bodies, 4)
            self.body_lin_vel_w = motion["body_lin_vel_w"]    # (T, num_bodies, 3)
            self.body_ang_vel_w = motion["body_ang_vel_w"]    # (T, num_bodies, 3)
            self.root_pos_w = self.body_pos_w[:, 0]    # (T, 3)
            self.root_quat_w = self.body_quat_w[:, 0]    # (T, 4)

        self.motion_length = self.joint_pos.shape[0]

    def enter(self, root_pos_w, root_quat_w):
        robot_yaw_q = yaw_quat(root_quat_w)
        ref_yaw_q = yaw_quat(self.root_quat_w[0])
        delta_yaw_q = quat_mul(robot_yaw_q, quat_inv(ref_yaw_q))

        ref_start_pos = self.body_pos_w[0, 0].copy()
        
        rel_pos = self.body_pos_w - ref_start_pos
        self.body_pos_w = root_pos_w + quat_apply(delta_yaw_q, rel_pos)
        
        T, N, _ = self.body_quat_w.shape
        self.body_quat_w = quat_mul(
            np.tile(delta_yaw_q, (T, N, 1)), 
            self.body_quat_w
        )
        
        self.body_lin_vel_w = quat_apply(delta_yaw_q, self.body_lin_vel_w)
        self.body_ang_vel_w = quat_apply(delta_yaw_q, self.body_ang_vel_w)

        self.root_pos_w = self.body_pos_w[:, 0]    # (T, 3)
        self.root_quat_w = self.body_quat_w[:, 0]    # (T, 4)


class ref_root_quat_w(Observation):
    def compute(self):
        ref_motion = self.articulation.ref_motion
        t = min(self.articulation.t, ref_motion.motion_length)
        return ref_motion.root_quat_w[t]

class ref_kp_pos_b(Observation):
    def __init__(self, articulation: Articulation, body_names: List[str], command: Optional[Any] = None):
        super().__init__(articulation)
        self.body_indices = self.articulation.find_bodies(body_names)[0]
    
    def compute(self):
        root_quat_w = self.articulation.root_quat_w
        body_quat_w = self.articulation.body_quat_w[self.body_indices]
        root_quat_conj = quat_conjugate(root_quat_w)
        # Expand root_quat_conj to match body_quat_w shape: (4,) -> (n_bodies, 4)
        root_quat_conj = np.tile(root_quat_conj, (body_quat_w.shape[0], 1))
        body_ori_b = quat_mul(root_quat_conj, body_quat_w)
        return body_ori_b.flatten()

class ref_root_quat(Observation):
    def compute(self):
        t = min(self.articulation.t, self.articulation.ref_motion.motion_length - 1)
        # Use get_aligned_body_state to get aligned reference body states
        _, aligned_body_quat_w = self.articulation.get_aligned_body_state(t)
        # Root body is at index 0 (pelvis)
        ref_root_quat_w = aligned_body_quat_w[0]
        root_quat_w = self.articulation.root_quat_w
        quat_error = quat_mul(quat_inv(root_quat_w), ref_root_quat_w)
        return quat_error.reshape(-1)

class ref_anchor_pos(Observation):
    def __init__(self, articulation: Articulation, anchor_body_name: str):
        super().__init__(articulation)
        self.anchor_body_index = articulation.find_bodies(anchor_body_name)[0][0]

    def compute(self):
        t = min(self.articulation.t, self.articulation.ref_motion.motion_length - 1)
        ref_anchor_pos = self.articulation.ref_motion.body_pos_w[t, self.anchor_body_index]
        ref_anchor_quat = self.articulation.ref_motion.body_quat_w[t, self.anchor_body_index]
        anchor_pos = self.articulation.body_pos_w[self.anchor_body_index]
        anchor_quat = self.articulation.body_quat_w[self.anchor_body_index]
        pos, _ = subtract_frame_transforms(anchor_pos, anchor_quat, ref_anchor_pos, ref_anchor_quat)
        return pos.reshape(-1)

class ref_anchor_quat(Observation):
    def __init__(self, articulation: Articulation, anchor_body_name: str):
        super().__init__(articulation)
        self.anchor_body_index = articulation.find_bodies(anchor_body_name)[0][0]

    def compute(self):
        t = min(self.articulation.t, self.articulation.ref_motion.motion_length - 1)
        ref_anchor_pos = self.articulation.ref_motion.body_pos_w[t, self.anchor_body_index]
        ref_anchor_quat = self.articulation.ref_motion.body_quat_w[t, self.anchor_body_index]
        anchor_pos = self.articulation.body_pos_w[self.anchor_body_index]
        anchor_quat = self.articulation.body_quat_w[self.anchor_body_index]
        _, quat = subtract_frame_transforms(anchor_pos, anchor_quat, ref_anchor_pos, ref_anchor_quat)
        return quat.reshape(-1)

class ref_kp_pos_gap(Observation):
    def __init__(self, articulation: Articulation, body_names: List[str]):
        super().__init__(articulation)
        self.body_indices = self.articulation.find_bodies(body_names)[0]

    def compute(self):
        t = min(self.articulation.t, self.articulation.ref_motion.motion_length - 1)
        # Use get_aligned_body_state to get aligned reference body states
        aligned_body_pos_w, aligned_body_quat_w = self.articulation.get_aligned_body_state(t)
        ref_kp_pos = aligned_body_pos_w[self.body_indices]
        ref_kp_quat = aligned_body_quat_w[self.body_indices]

        body_kp_pos = self.articulation.body_pos_w[self.body_indices]
        body_kp_quat = self.articulation.body_quat_w[self.body_indices]

        pos, _ = subtract_frame_transforms(body_kp_pos, body_kp_quat, ref_kp_pos, ref_kp_quat)
        return pos.flatten()

