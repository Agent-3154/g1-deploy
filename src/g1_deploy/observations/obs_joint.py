import numpy as np
from typing import Optional, Any, Union, List
from g1_deploy.base import Observation, Articulation


class joint_pos(Observation):
    def compute(self):
        return self.articulation.joint_pos
    
class joint_vel(Observation):
    def compute(self):
        return self.articulation.joint_vel


class joint_pos_multistep(Observation):
    def __init__(
        self,
        articulation: Articulation,
        joint_names: Optional[str] = ".*",
        steps: Union[int, List[int]] = 1,
        command: Optional[Any] = None,
    ):
        super().__init__(articulation, command)
        self.joint_ids = self.articulation.find_joints(joint_names)[0]
        if isinstance(steps, int):
            self.steps = slice(None)
            self.joint_pos_buffer = np.zeros((steps, len(self.joint_ids)))
        else:
            self.steps = steps
            self.joint_pos_buffer = np.zeros((max(steps), len(self.joint_ids)))
    
    def compute(self):
        joint_pos = np.asarray(self.articulation.joint_pos)[self.joint_ids]
        self.joint_pos_buffer = np.roll(self.joint_pos_buffer, 1, axis=0)
        self.joint_pos_buffer[0] = joint_pos
        return self.joint_pos_buffer[self.steps].reshape(-1)


class joint_vel_multistep(Observation):
    def __init__(
        self,
        articulation: Articulation,
        joint_names: Optional[str] = ".*",
        steps: Union[int, List[int]] = 1,
        command: Optional[Any] = None,
    ):
        super().__init__(articulation)
        self.joint_ids = self.articulation.find_joints(joint_names)[0]
        if isinstance(steps, int):
            self.steps = slice(None)
            self.joint_vel_buffer = np.zeros((steps, len(self.joint_ids)))
        else:
            self.steps = steps
            self.joint_vel_buffer = np.zeros((max(steps), len(self.joint_ids)))
    
    def compute(self):
        joint_vel = np.asarray(self.articulation.joint_vel)[self.joint_ids]
        self.joint_vel_buffer = np.roll(self.joint_vel_buffer, 1, axis=0)
        self.joint_vel_buffer[0] = joint_vel
        return self.joint_vel_buffer[self.steps].reshape(-1)

