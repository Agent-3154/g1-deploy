import numpy as np
import joblib
from utils import *
from typing import List
from pathlib import Path
from typing import NamedTuple

class Indexing(NamedTuple):
    mujoco2isaac: List[int]
    isaac2mujoco: List[int]


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

    # def enter(self, root_pos_w, root_quat_w):
    #     robot_yaw_q = yaw_quat(root_quat_w)
    #     ref_yaw_q = yaw_quat(self.root_quat_w[0])
    #     delta_yaw_q = quat_mul(robot_yaw_q, quat_inv(ref_yaw_q))

    #     ref_start_pos = self.body_pos_w[0, 0].copy()
        
    #     rel_pos = self.body_pos_w - ref_start_pos
    #     self.body_pos_w = root_pos_w + quat_apply(delta_yaw_q, rel_pos)
        
    #     T, N, _ = self.body_quat_w.shape
    #     self.body_quat_w = quat_mul(
    #         np.tile(delta_yaw_q, (T, N, 1)), 
    #         self.body_quat_w
    #     )
        
    #     self.body_lin_vel_w = quat_apply(delta_yaw_q, self.body_lin_vel_w)
    #     self.body_ang_vel_w = quat_apply(delta_yaw_q, self.body_ang_vel_w)

    #     self.root_pos_w = self.body_pos_w[:, 0]    # (T, 3)
    #     self.root_quat_w = self.body_quat_w[:, 0]    # (T, 4)

class Articulation:
    def __init__(
        self,
        robot,
        action_scaling: dict,
        default_joint_pos: dict,
        stiffness: dict,
        damping: dict,
    ):
        self.robot = robot
        
        self.default_joint_pos = np.zeros(29, dtype=np.float32)
        self.joint_stiffness = np.zeros(29, dtype=np.float32)
        self.joint_damping = np.zeros(29, dtype=np.float32)

        ids, _, default_joint_pos = resolve_matching_names_values(default_joint_pos, JOINT_NAMES_ISAAC)
        self.default_joint_pos[ids] = np.array(default_joint_pos, dtype=np.float32)
        ids, _, joint_stiffness = resolve_matching_names_values(stiffness, JOINT_NAMES_ISAAC)
        self.joint_stiffness[ids] = np.array(joint_stiffness, dtype=np.float32)
        ids, _, joint_damping = resolve_matching_names_values(damping, JOINT_NAMES_ISAAC)
        self.joint_damping[ids] = np.array(joint_damping, dtype=np.float32)

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

        # self.action_joint_ids, _ = resolve_matching_names(
        #     [   'left_hip_pitch_joint', 
        #         'right_hip_pitch_joint', 
        #         'waist_yaw_joint', 
        #         'left_hip_roll_joint', 
        #         'right_hip_roll_joint', 
        #         'left_hip_yaw_joint', 
        #         'right_hip_yaw_joint', 
        #         'left_knee_joint', 
        #         'right_knee_joint', 
        #         'left_shoulder_pitch_joint', 
        #         'right_shoulder_pitch_joint', 
        #         'left_ankle_pitch_joint', 
        #         'right_ankle_pitch_joint', 
        #         'left_shoulder_roll_joint', 
        #         'right_shoulder_roll_joint', 
        #         'left_ankle_roll_joint', 
        #         'right_ankle_roll_joint', 
        #         'left_shoulder_yaw_joint', 
        #         'right_shoulder_yaw_joint', 
        #         'left_elbow_joint', 
        #         'right_elbow_joint', 
        #         'left_wrist_roll_joint', 
        #         'right_wrist_roll_joint', 
        #         'left_wrist_pitch_joint', 
        #         'right_wrist_pitch_joint', 
        #         'left_wrist_yaw_joint', 
        #         'right_wrist_yaw_joint'
        #     ],
        #     joint_names_isaac
        # )
        
        self.action_joint_ids, self.action_joint_names, self.action_scaling = resolve_matching_names_values(action_scaling, JOINT_NAMES_ISAAC)
        self.action_scaling = np.array(self.action_scaling, dtype=np.float32)
        self.action_dim = len(self.action_joint_ids)
        self.action_buf = np.zeros((4, self.action_dim), dtype=np.float32)

        self.joint_position_target = np.zeros(29)

        self.ref_motion = RefMotion(motion_file = "sfu_29dof.pkl")
        self.t = 0
    
    def find_joints(self, joint_names: str):
        return resolve_matching_names(joint_names, JOINT_NAMES_ISAAC)
    
    def find_bodies(self, body_names: str):
        return resolve_matching_names(body_names, BODY_NAMES_ISAAC)
    
    @property
    def data(self):
        return self.robot.get_data()

    def apply_action(self, action: np.ndarray):
        self.action_buf[1:] = self.action_buf[:-1]
        self.action_buf[0] = action.reshape(self.action_dim)
        joint_position_target = self.default_joint_pos.copy()
        joint_position_target[self.action_joint_ids] += self.action_buf[0] * self.action_scaling
        
        self.robot.write_joint_position_target(joint_position_target[self.joint_indexing.isaac2mujoco])
        self.t += 1

    def reset(self):
        self.robot.write_joint_position_target(self.default_joint_pos[self.joint_indexing.isaac2mujoco])
        self.t = 0

    @property
    def root_pos_w(self):
        return np.asarray(self.data.position)

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

class Observation:
    
    registry = {}

    def __init__(
        self,
        articulation: Articulation
    ):
        self.articulation = articulation
    
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
    def __init__(self, asset: Articulation, steps: int):
        super().__init__(asset)
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
    def __init__(self, asset: Articulation, joint_names: str, steps: int):
        super().__init__(asset)
        self.steps = steps
        self.joint_ids = self.articulation.find_joints(joint_names)[0]
        self.joint_pos_buffer = np.zeros((steps, len(self.joint_ids)))
    
    def compute(self):
        joint_pos = np.asarray(self.articulation.joint_pos)[self.joint_ids]
        self.joint_pos_buffer = np.roll(self.joint_pos_buffer, 1, axis=0)
        self.joint_pos_buffer[0] = joint_pos
        return self.joint_pos_buffer.reshape(-1)


class joint_vel_multistep(Observation):
    def __init__(self, asset: Articulation, joint_names: str, steps: int):
        super().__init__(asset)
        self.steps = steps
        self.joint_ids = self.articulation.find_joints(joint_names)[0]
        self.joint_vel_buffer = np.zeros((steps, len(self.joint_ids)))
    
    def compute(self):
        joint_vel = np.asarray(self.articulation.joint_vel)[self.joint_ids]
        self.joint_vel_buffer = np.roll(self.joint_vel_buffer, 1, axis=0)
        self.joint_vel_buffer[0] = joint_vel
        return self.joint_vel_buffer.reshape(-1)


class prev_actions(Observation):
    def __init__(self, articulation: Articulation, steps: int):
        super().__init__(articulation)
        self.steps = steps
    
    def compute(self):
        return self.articulation.action_buf[:self.steps].reshape(-1)


class root_linvel_b(Observation):
    def compute(self):
        return quat_rotate_inverse(self.articulation.root_quat_w, self.articulation.root_lin_vel_w)


class body_height(Observation):
    def __init__(self, articulation: Articulation, body_names: str):
        super().__init__(articulation)
        self.body_ids = self.articulation.find_bodies(body_names)[0]
    
    def compute(self):
        return self.articulation.body_pos_w[self.body_ids, 2]


class body_pos_b(Observation):
    def __init__(self, articulation:Articulation):
        super().__init__(articulation)
        body_names = [  "left_hip_pitch_link", "right_hip_pitch_link", 
                        "left_knee_link", "right_knee_link", 
                        "left_ankle_roll_link", "right_ankle_roll_link", 
                        "left_shoulder_roll_link", "right_shoulder_roll_link", 
                        "left_elbow_link", "right_elbow_link", 
                        "left_wrist_yaw_link", "right_wrist_yaw_link"]
        self.body_indices, self.body_names = resolve_matching_names(body_names, BODY_NAMES_ISAAC)

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
    def __init__(self, articulation):
        super().__init__(articulation)
        self.ref_motion = self.articulation.ref_motion
        body_names =  [
            "pelvis",
            "left_hip_pitch_link", "right_hip_pitch_link", 
            "left_knee_link", "right_knee_link", 
            "left_ankle_roll_link", "right_ankle_roll_link", 
            "left_shoulder_roll_link", "right_shoulder_roll_link", 
            "left_elbow_link", "right_elbow_link", 
            "left_wrist_yaw_link", "right_wrist_yaw_link"
        ]
        self.body_indices, self.body_names = resolve_matching_names(body_names, BODY_NAMES_ISAAC)


    def compute(self):
        t = min(self.articulation.t, self.ref_motion.motion_length)
        ref_body_pos_w = self.ref_motion.body_pos_w[t][self.body_indices]
        ref_body_quat_w = self.ref_motion.body_quat_w[t][self.body_indices]

        body_pos_w = self.articulation.body_pos_w[self.body_indices]
        body_quat_w = self.articulation.body_quat_w[self.body_indices]

        pos, _ = subtract_frame_transforms(body_pos_w, body_quat_w, ref_body_pos_w, ref_body_quat_w)
        return pos.flatten()


class command(Observation):
    def compute(self):
        return np.array([0.5, 0., 0., 0.3])


############## LocoMode Obs ##############

class loco_ang_vel(Observation):
    def compute(self):
        return self.articulation.root_ang_vel_w

class loco_gravity_orientation(Observation):
    def compute(self):
        def get_gravity_orientation(quaternion):
            qw, qx, qy, qz = quaternion
            gravity_orientation = np.zeros(3)
            gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
            gravity_orientation[1] = -2 * (qz * qy + qw * qx)
            gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)
            return gravity_orientation
        root_quat_w = self.articulation.root_quat_w
        return get_gravity_orientation(root_quat_w)

class loco_cmd(Observation):
    def compute(self):
        return np.array([0.5, 0., 0.])

class loco_joint_pos(Observation):
    def compute(self):
        return self.articulation.joint_pos - self.articulation.default_joint_pos
    
class loco_joint_vel(Observation):
    def compute(self):
        return self.articulation.joint_vel

class loco_prev_actions(Observation):
    def compute(self):
        return self.articulation.action_buf[0].reshape(-1)

