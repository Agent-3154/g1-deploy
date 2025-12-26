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
        asset_meta: dict,
    ):
        self.robot = robot
        
        self.action_buf = np.zeros((1, 27), dtype=np.float32)
        self.asset_meta = asset_meta
        
        self.default_joint_pos = np.array(asset_meta["default_joint_pos"]).copy()
        self.joint_stiffness = np.array(asset_meta["stiffness"]).copy()
        self.joint_damping = np.array(asset_meta["damping"]).copy()

        self.joint_indexing = Indexing(
            mujoco2isaac = [joint_names_mujoco.index(name) for name in joint_names_isaac],
            isaac2mujoco = [joint_names_isaac.index(name) for name in joint_names_mujoco]
        )
        self.body_indexing = Indexing(
            mujoco2isaac = [body_names_mujoco.index(name) for name in body_names_isaac],
            isaac2mujoco = [body_names_isaac.index(name) for name in body_names_mujoco]
        )
        
        self.robot.set_joint_stiffness(self.joint_stiffness[self.joint_indexing.isaac2mujoco])
        self.robot.set_joint_damping(self.joint_damping[self.joint_indexing.isaac2mujoco])

        self.joint_position_target = np.zeros(29)

        self.ref_motion = RefMotion(motion_file = "sfu_29dof.pkl")
    
    @property
    def data(self):
        return self.robot.get_data()

    def apply_action(self, action: np.ndarray):
        self.action_buf[0] = action.reshape(27)
        joint_position_target = self.default_joint_pos + self.action_buf[0] * 0.5
        self.robot.write_joint_position_target(joint_position_target[self.joint_indexing.isaac2mujoco])

    def reset(self):
        self.robot.write_joint_position_target(self.default_joint_pos[self.joint_indexing.isaac2mujoco])

    @property
    def root_pos_w(self):
        return self.data.position

    @property
    def root_quat_w(self):
        return self.data.quaternion

    @property
    def root_lin_vel_w(self):
        return self.data.velocity

    @property
    def root_ang_vel_w(self):
        return self.data.omega

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
        return quat_rotate_inverse(self.articulation.root_quat_w, np.array([0., 0., -1.]))

class joint_pos(Observation):
    def compute(self):
        return self.articulation.joint_pos
    
class joint_vel(Observation):
    def compute(self):
        return self.articulation.joint_vel

class prev_actions(Observation):
    def compute(self):
        return self.articulation.action_buf.reshape(-1)

class body_pos_b(Observation):
    def __init__(self, articulation:Articulation):
        super().__init__(articulation)
        body_names = [  "left_hip_pitch_link", "right_hip_pitch_link", 
                        "left_knee_link", "right_knee_link", 
                        "left_ankle_roll_link", "right_ankle_roll_link", 
                        "left_shoulder_roll_link", "right_shoulder_roll_link", 
                        "left_elbow_link", "right_elbow_link", 
                        "left_wrist_yaw_link", "right_wrist_yaw_link"]
        self.body_indices, self.body_names = resolve_matching_names(body_names, body_names_isaac)

    def compute(self):
        root_pos_w = self.articulation.root_pos_w
        root_quat_w = self.articulation.root_quat_w
        body_pos_w = self.articulation.body_pos_w[self.body_indices]
        body_pos_b = quat_rotate_inverse(root_quat_w, body_pos_w - root_pos_w)
        return body_pos_b.flatten()

class ref_root_quat_w(Observation):
    def compute(self, t):
        ref_motion = self.articulation.ref_motion
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
                    ],
        self.body_indices, self.body_names = resolve_matching_names(body_names, body_names_isaac)


    def compute(self, t):
        ref_body_pos_w = self.ref_motion.body_pos_w[t][self.body_indices]
        ref_body_quat_w = self.ref_motion.body_quat_w[t][self.body_indices]

        body_pos_w = self.articulation.body_pos_w[self.body_indices]
        body_quat_w = self.articulation.body_quat_w[self.body_indices]

        pos, _ = subtract_frame_transforms(body_pos_w, body_quat_w, ref_body_pos_w, ref_body_quat_w)
        return pos.flatten()