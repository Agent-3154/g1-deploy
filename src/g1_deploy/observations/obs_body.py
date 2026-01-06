import numpy as np
from typing import Optional, Any, List, Union
from g1_deploy.base import Observation, Articulation

from g1_deploy.utils.math import quat_rotate_inverse, quat_mul, quat_conjugate


class command(Observation):
    def compute(self):
        return np.array([0.5, 0.0, 0.0, 0.3], dtype=np.float32)


class root_quat_w(Observation):
    def compute(self):
        return self.articulation.root_quat_w


class root_linvel_b(Observation):
    def compute(self):
        return self.articulation.root_lin_vel_b


class root_angvel_b(Observation):
    def __init__(
        self,
        articulation: Articulation,
        steps: Union[int, List[int]] = 1,
        command: Optional[Any] = None,
    ):
        super().__init__(articulation, command)
        if isinstance(steps, int):
            self.steps = slice(None)
            self.root_ang_vel_buffer = np.zeros((steps, 3))
        else:
            self.steps = steps
            self.root_ang_vel_buffer = np.zeros((max(steps), 3))
    
    def compute(self):
        root_ang_vel = np.asarray(self.articulation.root_ang_vel_b)
        self.root_ang_vel_buffer = np.roll(self.root_ang_vel_buffer, 1, axis=0)
        self.root_ang_vel_buffer[0] = root_ang_vel
        return self.root_ang_vel_buffer[self.steps].reshape(-1)


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

class body_ori_b(Observation):
    def __init__(self, articulation: Articulation, body_names: List[str], command: Optional[Any] = None):
        super().__init__(articulation)
        self.body_indices = self.articulation.find_bodies(body_names)[0]
    
    def compute(self):
        root_quat_w = self.articulation.root_quat_w
        body_quat_w = self.articulation.body_quat_w[self.body_indices]
        
        root_quat_conj = quat_conjugate(root_quat_w)
        root_quat_conj = np.tile(root_quat_conj, (body_quat_w.shape[0], 1))
        body_ori_b = quat_mul(root_quat_conj, body_quat_w)
        return body_ori_b.flatten()


class projected_gravity(Observation):
    def __init__(
        self,
        articulation: Articulation,
        steps: Union[int, List[int]] = 1,
        command: Optional[Any] = None,
    ):
        super().__init__(articulation, command)
        if isinstance(steps, int):
            self.steps = slice(None)
            self.projected_gravity_buffer = np.zeros((steps, 3))
        else:
            self.steps = steps
            self.projected_gravity_buffer = np.zeros((max(steps), 3))
    
    def compute(self):
        projected_gravity = np.asarray(self.articulation.data.projected_gravity)
        self.projected_gravity_buffer = np.roll(self.projected_gravity_buffer, 1, axis=0)
        self.projected_gravity_buffer[0] = projected_gravity
        return self.projected_gravity_buffer[self.steps].reshape(-1)


class prev_actions(Observation):
    def __init__(
        self,
        articulation: Articulation,
        steps: int,
        command: Optional[Any] = None
    ):
        super().__init__(articulation)
        self.steps = steps
    
    def compute(self):
        return self.articulation.action_buf[:self.steps].reshape(-1)


class body_height(Observation):
    def __init__(self, articulation: Articulation, body_names: str, command: Optional[Any] = None):
        super().__init__(articulation)
        self.body_ids = self.articulation.find_bodies(body_names)[0]
    
    def compute(self):
        return self.articulation.body_pos_w[self.body_ids, 2]

