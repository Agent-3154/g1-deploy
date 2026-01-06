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

class Observation:
    
    registry = {}

    def __init__(
        self,
        articulation: Articulation,
        command: Optional[Any] = None,
    ):
        self.articulation = articulation
        self.command = command
    
    def __init_subclass__(cls):
        cls_name = cls.__name__
        cls.registry[cls_name] = cls

    def __call__(self):
        return self.compute()
    
    def compute(self):
        raise NotImplementedError


class root_quat_w(Observation):
    def compute(self):
        return self.articulation.root_quat_w

class root_angvel_b(Observation):
    def compute(self):
        root_quat_w = self.articulation.root_quat_w
        root_ang_vel_w = self.articulation.root_ang_vel_w
        return quat_rotate_inverse(root_quat_w, root_ang_vel_w)

class projected_gravity_b(Observation):
    def compute(self):
        return np.asarray(self.articulation.data.projected_gravity)

class gravity_multistep(Observation):
    def __init__(self, articulation: Articulation, steps: int, command: Optional[Any] = None):
        super().__init__(articulation, command)
        self.steps = steps
        self.gravity_buffer = np.zeros((steps, 3))
    
    def compute(self):
        gravity = np.asarray(self.articulation.data.projected_gravity)
        self.gravity_buffer = np.roll(self.gravity_buffer, 1, axis=0)
        self.gravity_buffer[0] = gravity
        return self.gravity_buffer.reshape(-1)


class joint_pos(Observation):
    def compute(self):
        return self.articulation.joint_pos
    
class joint_vel(Observation):
    def compute(self):
        return self.articulation.joint_vel


class joint_pos_multistep(Observation):
    def __init__(self, articulation: Articulation, joint_names: str, steps: int, command: Optional[Any] = None):
        super().__init__(articulation, command)
        self.steps = steps
        self.joint_ids = self.articulation.find_joints(joint_names)[0]
        self.joint_pos_buffer = np.zeros((steps, len(self.joint_ids)))
    
    def compute(self):
        joint_pos = np.asarray(self.articulation.joint_pos)[self.joint_ids]
        self.joint_pos_buffer = np.roll(self.joint_pos_buffer, 1, axis=0)
        self.joint_pos_buffer[0] = joint_pos
        return self.joint_pos_buffer.reshape(-1)


class joint_vel_multistep(Observation):
    def __init__(self, articulation: Articulation, joint_names: str, steps: int, command: Optional[Any] = None):
        super().__init__(articulation)
        self.steps = steps
        self.joint_ids = self.articulation.find_joints(joint_names)[0]
        self.joint_vel_buffer = np.zeros((steps, len(self.joint_ids)))
    
    def compute(self):
        joint_vel = np.asarray(self.articulation.joint_vel)[self.joint_ids]
        self.joint_vel_buffer = np.roll(self.joint_vel_buffer, 1, axis=0)
        self.joint_vel_buffer[0] = joint_vel
        return self.joint_vel_buffer.reshape(-1)


class prev_actions(Observation):
    def __init__(self, articulation: Articulation, steps: int, command: Optional[Any] = None):
        super().__init__(articulation)
        self.steps = steps
    
    def compute(self):
        return self.articulation.action_buf[:self.steps].reshape(-1)


class root_linvel_b(Observation):
    def compute(self):
        return quat_rotate_inverse(self.articulation.root_quat_w, self.articulation.root_lin_vel_w)


class body_height(Observation):
    def __init__(self, articulation: Articulation, body_names: str, command: Optional[Any] = None):
        super().__init__(articulation)
        self.body_ids = self.articulation.find_bodies(body_names)[0]
    
    def compute(self):
        return self.articulation.body_pos_w[self.body_ids, 2]


class body_pos_b(Observation):
    def __init__(self, articulation: Articulation, body_names: List[str], command: Optional[Any] = None):
        super().__init__(articulation)
        self.body_indices = self.articulation.find_bodies(body_names)[0]

    def compute(self):
        root_pos_w = self.articulation.root_pos_w
        root_quat_w = self.articulation.root_quat_w
        body_pos_w = self.articulation.body_pos_w[self.body_indices]
        body_pos_b = quat_rotate_inverse(root_quat_w, body_pos_w - root_pos_w)
        return body_pos_b.flatten()

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


class command(Observation):
    def compute(self):
        return np.array([0.5, 0., 0., 0.3])
