import numpy as np
from typing import Union, List, NamedTuple

from g1_deploy import G1HardwareInterface, G1MujocoInterface
from g1_deploy.utils.string import (
    resolve_matching_names,
    resolve_matching_names_values,
)
from g1_deploy.utils.constants import (
    BODY_NAMES_ISAAC,
    BODY_NAMES_MUJOCO,
    JOINT_NAMES_ISAAC,
    JOINT_NAMES_MUJOCO
)

RobotInterface = Union[G1HardwareInterface, G1MujocoInterface]


class Indexing(NamedTuple):
    mujoco2isaac: List[int]
    isaac2mujoco: List[int]


class Articulation:
    def __init__(
        self,
        robot: RobotInterface,
        action_scaling: dict | np.ndarray,
        default_joint_pos: dict | np.ndarray,
        stiffness: dict | np.ndarray,
        damping: dict | np.ndarray,
    ):
        self.robot = robot
        
        if isinstance(default_joint_pos, dict):
            self.default_joint_pos = np.zeros(29, dtype=np.float32)
            ids, _, default_joint_pos = resolve_matching_names_values(default_joint_pos, JOINT_NAMES_ISAAC)
            self.default_joint_pos[ids] = np.array(default_joint_pos, dtype=np.float32)
        else:
            self.default_joint_pos = np.array(default_joint_pos, dtype=np.float32)
        
        if isinstance(stiffness, dict):
            self.joint_stiffness = np.zeros(29, dtype=np.float32)
            ids, _, stiffness = resolve_matching_names_values(stiffness, JOINT_NAMES_ISAAC)
            self.joint_stiffness[ids] = np.array(stiffness, dtype=np.float32)
        else:
            self.joint_stiffness = np.array(stiffness, dtype=np.float32)
        
        if isinstance(damping, dict):
            self.joint_damping = np.zeros(29, dtype=np.float32)
            ids, _, damping = resolve_matching_names_values(damping, JOINT_NAMES_ISAAC)
            self.joint_damping[ids] = np.array(damping, dtype=np.float32)
        else:
            self.joint_damping = np.array(damping, dtype=np.float32)

        self.joint_indexing = Indexing(
            mujoco2isaac = [JOINT_NAMES_MUJOCO.index(name) for name in JOINT_NAMES_ISAAC],
            isaac2mujoco = [JOINT_NAMES_ISAAC.index(name) for name in JOINT_NAMES_MUJOCO]
        )
        self.body_indexing = Indexing(
            mujoco2isaac = [BODY_NAMES_MUJOCO.index(name) for name in BODY_NAMES_ISAAC],
            isaac2mujoco = [BODY_NAMES_ISAAC.index(name) for name in BODY_NAMES_MUJOCO]
        )
        
        self.robot.set_joint_stiffness(self.joint_stiffness[self.joint_indexing.isaac2mujoco])
        self.robot.set_joint_damping(self.joint_damping[self.joint_indexing.isaac2mujoco])
        
        self.action_joint_ids, self.action_joint_names, self.action_scaling = resolve_matching_names_values(action_scaling, JOINT_NAMES_ISAAC)
        self.action_scaling = np.array(self.action_scaling, dtype=np.float32)
        self.action_dim = len(self.action_joint_ids)
        self.action_buf = np.zeros((4, self.action_dim), dtype=np.float32)
        self.applied_action = np.zeros(self.action_dim, dtype=np.float32)

        self.joint_position_target = np.zeros(29)

        # self.ref_motion = RefMotion(motion_file = "sfu_29dof.pkl")
        # self.ref_motion.enter(self.root_pos_w, self.root_quat_w)
        self.t = 0
    
    def find_joints(self, joint_names: str):
        return resolve_matching_names(joint_names, JOINT_NAMES_ISAAC)
    
    def find_bodies(self, body_names: str):
        return resolve_matching_names(body_names, BODY_NAMES_ISAAC)
    
    @property
    def data(self):
        return self.robot.get_data()

    def process_action(self, action: np.ndarray):
        action = action.reshape(self.action_dim)
        self.action_buf[1:] = self.action_buf[:-1]
        self.action_buf[0] = action
        
    def apply_action(self, alpha: float = 0.8):
        self.applied_action = self.applied_action * (1 - alpha) + self.action_buf[0] * alpha
        joint_position_target = self.default_joint_pos.copy()
        joint_position_target[self.action_joint_ids] += self.applied_action * self.action_scaling
        
        self.robot.write_joint_position_target(joint_position_target[self.joint_indexing.isaac2mujoco])
        self.t += 1

    def reset(self):
        self.robot.reset(self.default_joint_pos[self.joint_indexing.isaac2mujoco])
        # self.robot.write_joint_position_target(self.default_joint_pos[self.joint_indexing.isaac2mujoco])
        self.t = 0

    @property
    def root_pos_w(self):
        return np.asarray(self.data.root_pos_w)

    @property
    def root_quat_w(self):
        return np.asarray(self.data.quaternion)

    @property
    def root_lin_vel_w(self):
        return np.asarray(self.data.root_lin_vel_w)

    @property
    def root_ang_vel_w(self):
        return np.asarray(self.data.root_ang_vel_w)

    @property
    def joint_pos(self):
        return np.asarray(self.data.q)[self.joint_indexing.mujoco2isaac]

    @property
    def joint_vel(self):
        return np.asarray(self.data.dq)[self.joint_indexing.mujoco2isaac]

    @property
    def body_pos_w(self):
        return np.asarray(self.data.body_positions)[self.body_indexing.mujoco2isaac]

    @property
    def body_quat_w(self):
        return np.asarray(self.data.body_quaternions)[self.body_indexing.mujoco2isaac]

