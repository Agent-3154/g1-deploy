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
from policy import FSM, SkillA, SkillB
from timerfd import Timer


# Add the build directory to Python path to import the compiled module
build_dir = Path(__file__).parent.parent / "build"
sys.path.insert(0, str(build_dir))

# Import the g1_interface module
import g1_interface

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hardware", action="store_true", help="Use hardware interface")
    return parser.parse_args()


def extract_meshes(model: mujoco.MjModel) -> OrderedDict[str, trimesh.Trimesh]:
    meshes = OrderedDict()
    for i in range(model.nbody):
        body = model.body(i)
        geomadr = body.geomadr[0]
        geomnum = body.geomnum[0]
        body_meshes = []
        for geomid in range(geomadr, geomadr + geomnum):
            geom = model.geom(geomid)
            if geom.type == mujoco.mjtGeom.mjGEOM_MESH and geom.contype[0] == 0:
                mesh = model.mesh(geom.dataid[0])
                faceadr = mesh.faceadr.item()
                facenum = mesh.facenum.item()
                vertadr = mesh.vertadr.item()
                vertnum = mesh.vertnum.item()
                mesh = trimesh.Trimesh(
                    vertices=model.mesh_vert[vertadr:vertadr + vertnum],
                    faces=model.mesh_face[faceadr:faceadr + facenum]
                )
                transform = trimesh.transformations.concatenate_matrices(
                    trimesh.transformations.translation_matrix(geom.pos),
                    trimesh.transformations.quaternion_matrix(geom.quat)
                )
                mesh.apply_transform(transform)
                body_meshes.append(mesh)
        if len(body_meshes) > 0:
            body_mesh = trimesh.util.concatenate(body_meshes)
            body_mesh.merge_vertices()
            meshes[body.name] = body_mesh
    return meshes


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

    asset_meta_path = Path(__file__).parent.parent / "checkpoints" / "asset_meta.json"
    with open(asset_meta_path, "r") as f:
        asset_meta = json.load(f)
    config_path = Path(__file__).parent.parent / "cfg" / "loco.yaml"
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
            "sA": SkillA(
                "sA", 
                robot, 
                Path(__file__).parent.parent / "cfg" / "loco.yaml", 
                Path(__file__).parent.parent / "checkpoints" / "policy-12-28_17-02.onnx"
            ),
            "sB": SkillB(
                "sB", 
                robot, 
                Path(__file__).parent.parent / "cfg" / "loco.yaml", 
                Path(__file__).parent.parent / "checkpoints" / "policy-12-28_17-02.onnx"
            )
        },
        start_policy_name = "sA"
    )
    
    def key_callback(keycode):
        try:
            key = chr(keycode).upper()
            if key == 'A':
                print("Key A pressed: switching to SkillA")
                fsm.set_next_policy("sA")
            elif key == 'B':
                print("Key B pressed: switching to SkillB")
                fsm.set_next_policy("sB")
        except:
            pass

    viewer = mujoco.viewer.launch_passive(mjModel, mjData, key_callback=key_callback)

    '''to be replaced with mujoco viewer key callback'''
    import sys
    import select
    import termios
    import tty

    # def get_key():
    #     fd = sys.stdin.fileno()
    #     old_settings = termios.tcgetattr(fd)
    #     try:
    #         tty.setraw(fd)
    #         [i, _, _] = select.select([sys.stdin], [], [], 0.01)
    #         if i:
    #             key = sys.stdin.read(1)
    #         else:
    #             key = None
    #     finally:
    #         termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    #     return key

    timer = Timer(0.02)
    for i in itertools.count():
        mjData.qpos[0:3] = robot.data.root_pos_w
        mjData.qpos[3:7] = robot.data.quaternion
        mjData.qpos[7:] = robot.data.q
        mjData.qvel[6:] = robot.data.dq
        mujoco.mj_forward(mjModel, mjData)

        # key = get_key()
        # if key == 'a':
        #     fsm.set_next_policy("sA")
        # elif key == 'b':
        #     fsm.set_next_policy("sB")

        action = fsm.run()
        robot.apply_action(action)

        if i % 500 == 0:
            robot.reset()

        viewer.sync()
        timer.sleep()

