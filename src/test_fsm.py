import sys
import time
import mujoco
import mujoco.viewer
import numpy as np
import argparse
import trimesh
import rerun as rr
import itertools
import json
import onnxruntime as ort
import yaml
import torch
from pathlib import Path
from collections import OrderedDict
from observation import Observation, Articulation
from policy import FSM, SkillA, SkillB, TrackMode
from timerfd import Timer
from utils import extract_meshes


# Add the build directory to Python path to import the compiled module
build_dir = Path(__file__).parent.parent / "build"
sys.path.insert(0, str(build_dir))

# Import the g1_interface module
import g1_interface

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hardware", action="store_true", help="Use hardware interface")
    return parser.parse_args()

# Example usage
if __name__ == "__main__":
    # Create a G1Interface instance
    # Replace "eth0" with your actual network interface name
    args = parse_args()
    # onnx_path = Path(__file__).parent.parent / "checkpoints" / "motion.onnx"
    # onnx_module = ONNXModule(onnx_path)
    # print(onnx_module)
    # config_path = Path(__file__).parent.parent / "cfg" / "config.yaml"

    hardware = args.hardware
    mjcf_path = Path(__file__).parent.parent / "mjcf" / "g1.xml"
    if hardware:
        robot_interface = g1_interface.G1HarwareInterface("enp58s0")
        robot_interface.load_mjcf(str(mjcf_path))
        mjModel = mujoco.MjModel.from_xml_path(str(mjcf_path))
    else:
        scene_path = Path(__file__).parent.parent / "mjcf" / "g1_with_floor.xml"
        robot_interface = g1_interface.G1MujocoInterface(str(scene_path))
        robot_interface.run_async()
        mjModel = mujoco.MjModel.from_xml_path(str(scene_path))
        print(f"timestep: {robot_interface.get_timestep()}")

    mjData = mujoco.MjData(mjModel)
    mujoco.mj_forward(mjModel, mjData)
    meshes = extract_meshes(mjModel)
    print(len(meshes))
    
    rr.init("g1", recording_id="g1")
    rr.connect_grpc("rerun+http://192.168.3.23:9876/proxy")
    rr.set_time("step", timestamp=0.0)
    for body_name, mesh in meshes.items():
        rr.log(
            f"robot/{body_name}",
            rr.Mesh3D(
                vertex_positions=mesh.vertices,
                triangle_indices=mesh.faces,
                vertex_normals=mesh.vertex_normals,
            ),
        )

    asset_meta_path = Path(__file__).parent.parent / "checkpoints" / "asset_meta.json"
    with open(asset_meta_path, "r") as f:
        asset_meta = json.load(f)
    # config_path = Path(__file__).parent.parent / "cfg" / "loco.yaml"
    config_path = Path(__file__).parent.parent / "cfg" / "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    mujoco_joint_names = [mjModel.joint(i).name for i in range(mjModel.njnt)]
    mujoco_joint_names.remove("floating_base_joint")
    robot = Articulation(
        robot_interface,
        config["action"]["action_scaling"],
        asset_meta["default_joint_pos"],
        asset_meta["stiffness"],
        asset_meta["damping"],
    )

    fsm = FSM(
        policies = {
            "track": TrackMode(
                "track",
                robot,
                Path(__file__).parent.parent / "cfg" / "config.yaml",
                Path(__file__).parent.parent / "checkpoints" / "motion.onnx"
            )
        },
        start_policy_name = "track"
    )

    '''to be replaced with mujoco viewer key callback'''
    import sys
    import select
    import termios
    import tty

    def get_key():
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            [i, _, _] = select.select([sys.stdin], [], [], 0.01)
            if i:
                key = sys.stdin.read(1)
            else:
                key = None
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return key

    timer = Timer(0.02)
    # viewer = mujoco.viewer.launch_passive(mjModel, mjData)
    for i in itertools.count():
        data = robot.data
        mjData.qpos[0:3] = data.root_pos_w
        mjData.qpos[3:7] = data.quaternion
        mjData.qpos[7:] = data.q
        mjData.qvel[6:] = data.dq
        mujoco.mj_forward(mjModel, mjData)

        key = get_key()
        if key == 'a':
            fsm.set_next_policy("sA")
        elif key == 'b':
            fsm.set_next_policy("sB")
        elif key == 't':
            fsm.set_next_policy("track")

        action = fsm.run()
        robot.apply_action(action)
        
        body_names = [
            "left_hip_pitch_link", "right_hip_pitch_link", 
            "left_knee_link", "right_knee_link", 
            "left_ankle_roll_link", "right_ankle_roll_link", 
            "left_shoulder_roll_link", "right_shoulder_roll_link", 
            "left_elbow_link", "right_elbow_link", 
            "left_wrist_yaw_link", "right_wrist_yaw_link"
        ]

        body_indices = robot.find_bodies(body_names)[0]

        rr.set_time("step", timestamp=i)
        xpos = np.asarray(data.body_positions)
        xquat = np.asarray(data.body_quaternions)[:, [1, 2, 3, 0]]
        for ii, (body_name, mesh) in enumerate(meshes.items()):
            if ii in body_indices:
                rr.log(
                    f"robot/{body_name}_position",
                    rr.Points3D(positions=xpos[ii], colors=[255, 0, 0], radii=0.01)
                )
            rr.log(
                f"robot/{body_name}",
                rr.Transform3D(translation=xpos[ii], quaternion=xquat[ii])
            )

        if i % 500 == 0:
            robot.reset()

        # viewer.sync()
        timer.sleep()

