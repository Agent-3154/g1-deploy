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
    def __init__(self, articulation: Articulation, motion_file):
        self.articulation = articulation
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

    def get_aligned_body_state(self, t: int, anchor_body_index: int):
        """Get aligned body state from reference motion.
        
        Args:
            t: Current time step index.
            anchor_body_name: Name of the anchor body for alignment. Default is "torso_link".
            
        Returns:
            aligned_body_pos_w: Aligned body positions in world frame. Shape (num_bodies, 3).
            aligned_body_quat_w: Aligned body quaternions in world frame. Shape (num_bodies, 4).
        """
        t = min(t, self.motion_length - 1)
        
        # Get reference body states from motion
        ref_body_pos_w = self.body_pos_w[t]  # (num_bodies, 3)
        ref_body_quat_w = self.body_quat_w[t]  # (num_bodies, 4)
        
        num_bodies = ref_body_pos_w.shape[0]
        # Get anchor body states
        ref_anchor_pos_w = ref_body_pos_w[anchor_body_index]  # (3,)
        ref_anchor_quat_w = ref_body_quat_w[anchor_body_index]  # (4,)
        
        robot_anchor_pos_w = self.articulation.body_pos_w[anchor_body_index]  # (3,)
        robot_anchor_quat_w = self.articulation.body_quat_w[anchor_body_index]  # (4,)
        
        ref_anchor_pos_w_repeat = np.tile(ref_anchor_pos_w, (num_bodies, 1))  # (num_bodies, 3)
        robot_anchor_pos_w_repeat = np.tile(robot_anchor_pos_w, (num_bodies, 1))  # (num_bodies, 3)

        delta_pos_w = robot_anchor_pos_w_repeat.copy()  # (num_bodies, 3)
        delta_pos_w[:, 2] = ref_anchor_pos_w_repeat[:, 2]

        delta_quat = quat_mul(robot_anchor_quat_w, quat_inv(ref_anchor_quat_w))  # (4,)
        delta_yaw_quat = yaw_quat(delta_quat)  # (4,)
        delta_ori_w = np.tile(delta_yaw_quat, (num_bodies, 1))  # (num_bodies, 4)
        
        aligned_body_quat_w = quat_mul(delta_ori_w, ref_body_quat_w)  # (num_bodies, 4)
        aligned_body_pos_w = delta_pos_w + quat_apply(delta_ori_w, ref_body_pos_w - ref_anchor_pos_w_repeat)  # (num_bodies, 3)
        return aligned_body_pos_w, aligned_body_quat_w