import numpy as np
from typing import Optional, Any, List, Union
from g1_deploy.base import Observation, Articulation
from g1_deploy.utils.motion import MotionDataset, MotionData
from g1_deploy.utils.math import (
    quat_rotate_inverse,
    yaw_quat,
    quat_mul,
    quat_inv,
    quat_conjugate,
    subtract_frame_transforms,
    matrix_from_quat,
)
# from track.track import command


class motion_command:
    def __init__(
        self,
        motion_path: str,
        future_steps: List[int],
        joint_names: List[str],
        body_names: List[str],
        root_body_name: str = "pelvis",
    ):
        self.motion_dataset = MotionDataset.create_from_path(motion_path)
        self.future_steps = future_steps
        self.motion_ids = np.array([0])
        self.motion_length = self.motion_dataset.num_steps
        self.t = 0

        self.joint_indices = [self.motion_dataset.joint_names.index(name) for name in joint_names]
        self.body_indices = [self.motion_dataset.body_names.index(name) for name in body_names]
        self.root_body_idx = self.motion_dataset.body_names.index(root_body_name)

        self.n_future_steps = len(self.future_steps)
        self.n_bodies = len(self.body_indices)
    
    def update(self) -> None:
        self.t += 1
        if self.t == self.motion_length:
            self.t = 0
        motion_data: MotionData = self.motion_dataset.get_slice(self.motion_ids, self.t, self.future_steps)
        self.ref_joint_pos_future = motion_data.joint_pos[:, :, self.joint_indices]
        self.ref_joint_vel_future = motion_data.joint_vel[:, :, self.joint_indices]
        self.ref_body_pos_future_w = motion_data.body_pos_w[:, :, self.body_indices]
        self.ref_body_lin_vel_future_w = motion_data.body_lin_vel_w[:, :, self.body_indices]
        self.ref_body_quat_future_w = motion_data.body_quat_w[:, :, self.body_indices]
        self.ref_body_ang_vel_future_w = motion_data.body_ang_vel_w[:, :, self.body_indices]
        self.ref_root_pos_w = motion_data.body_pos_w[:, [0], [self.root_body_idx], :]
        self.ref_root_quat_w = motion_data.body_quat_w[:, [0], [self.root_body_idx], :]


class ref_joint_pos_future(Observation):
    command: motion_command
    def compute(self) -> np.ndarray:
        return self.command.ref_joint_pos_future.reshape(-1)


class ref_joint_vel_future(Observation):
    command: motion_command
    def compute(self) -> np.ndarray:
        return self.command.ref_joint_vel_future.reshape(-1)


class ref_body_pos_future_local(Observation):
    """
    Reference body position in motion root frame
    """
    command: motion_command
    def update(self) -> None:
        ref_body_pos_future_w = self.command.ref_body_pos_future_w
        ref_root_pos_w: np.ndarray = self.command.ref_root_pos_w # [batch, 1, 1, 3]
        ref_root_quat_w: np.ndarray = self.command.ref_root_quat_w  # [batch, 1, 1, 4]

        # Expand dimensions to match ref_body_pos_future_w
        ref_root_pos_w = np.tile(ref_root_pos_w, (1, self.command.n_future_steps, self.command.n_bodies, 1))  # [batch, future_steps, n_bodies, 3]
        ref_root_quat_w = np.tile(ref_root_quat_w, (1, self.command.n_future_steps, self.command.n_bodies, 1))  # [batch, future_steps, n_bodies, 4]

        ref_root_pos_w[..., 2] = 0.0
        ref_root_quat_w = yaw_quat(ref_root_quat_w)

        ref_body_pos_future_local = quat_rotate_inverse(ref_root_quat_w, ref_body_pos_future_w - ref_root_pos_w)
        self.ref_body_pos_future_local = ref_body_pos_future_local
    
    def compute(self):
        return self.ref_body_pos_future_local.reshape(-1)


class ref_motion_phase(Observation):
    command: motion_command
    def __init__(self, articulation: Articulation, motion_duration_second: float, command: motion_command):
        super().__init__(articulation, command)
        self.motion_steps = int(motion_duration_second * 50)
    
    def compute(self) -> np.ndarray:
        ref_motion_phase = (self.command.t % self.motion_steps) / self.motion_steps
        return ref_motion_phase.reshape(-1)

from g1_deploy.commands import RefMotion

class ref_root_pos(Observation):
    def __init__(self, articulation: Articulation, command: RefMotion):
        super().__init__(articulation, command)

    def compute(self):
        t = min(self.articulation.t, self.command.motion_length - 1)
        
        ref_root_pos_w = self.command.root_pos_w[t]
        ref_root_quat_w = self.command.root_quat_w[t]
        root_pos_w = self.articulation.root_pos_w
        root_quat_w = self.articulation.root_quat_w
        pos, quat = subtract_frame_transforms(root_pos_w, root_quat_w, ref_root_pos_w, ref_root_quat_w)
        return pos.reshape(-1)

class ref_root_quat(Observation):
    def __init__(self, articulation: Articulation, command: RefMotion):
        super().__init__(articulation, command)

    def compute(self):
        t = min(self.articulation.t, self.command.motion_length - 1)
        
        ref_root_pos_w = self.command.root_pos_w[t]
        ref_root_quat_w = self.command.root_quat_w[t]
        root_pos_w = self.articulation.root_pos_w
        root_quat_w = self.articulation.root_quat_w
        pos, quat = subtract_frame_transforms(root_pos_w, root_quat_w, ref_root_pos_w, ref_root_quat_w)
        mat = matrix_from_quat(quat)
        return mat[..., :2].reshape(-1)

class ref_kp_pos_gap(Observation):
    def __init__(self, articulation: Articulation, command: RefMotion, body_names: List[str]):
        super().__init__(articulation, command)
        self.body_indices = self.articulation.find_bodies(body_names)[0]

    def compute(self):
        t = min(self.articulation.t, self.command.motion_length - 1)
        
        ref_kp_pos = self.command.body_pos_w[t, self.body_indices]
        ref_kp_quat = self.command.body_quat_w[t, self.body_indices]
        body_kp_pos = self.articulation.body_pos_w[self.body_indices]
        body_kp_quat = self.articulation.body_quat_w[self.body_indices]
        pos, _ = subtract_frame_transforms(body_kp_pos, body_kp_quat, ref_kp_pos, ref_kp_quat)
        return pos.reshape(-1)

class ref_qpos(Observation):
    def __init__(self, articulation: Articulation, command: RefMotion, joint_names: List[str] = ".*"):
        super().__init__(articulation, command)
        self.joint_indices = self.articulation.find_joints(joint_names)[0]

    def compute(self):
        t = min(self.articulation.t, self.command.motion_length - 1)
        ref_qpos = self.command.joint_pos[t, self.joint_indices]
        return ref_qpos.reshape(-1)