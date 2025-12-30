import sys
import time
import mujoco
import mujoco.viewer
import numpy as np
import argparse
import rerun as rr
import itertools
import yaml
from pathlib import Path
from observation import Observation, Articulation
from timerfd import Timer
from policy import ONNXModule
from utils import extract_meshes


# Add the build directory to Python path to import the compiled module
build_dir = Path(__file__).parent.parent / "build"
sys.path.insert(0, str(build_dir))

# Import the g1_interface module
import g1_interface

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hardware", action="store_true", help="Use hardware interface")
    parser.add_argument("--rerun", action="store_true", help="Use rerun")
    parser.add_argument("--sync", action="store_true", help="Use sync mode")
    return parser.parse_args()


# Example usage
if __name__ == "__main__":
    # Create a G1Interface instance
    # Replace "eth0" with your actual network interface name
    args = parse_args()
    # onnx_path = "/home/btx0424/lab51/active-adaptation/scripts/exports/G1Flat29/policy-12-28_17-02.onnx"
    # onnx_path = "/home/btx0424/lab51/deploy/g1-deploy/checkpoints/policy-12-30_14-47.onnx"
    # config_path = Path(__file__).parent.parent / "cfg" / "loco_body.yaml"
    
    # onnx_path = Path(__file__).parent.parent / "checkpoints" / "motion.onnx"
    onnx_path = Path(__file__).parent.parent / "checkpoints" / "policy-12-30_15-28.onnx"
    config_path = Path(__file__).parent.parent / "cfg" / "test.yaml"
    
    onnx_module = ONNXModule(onnx_path)
    print(onnx_module)

    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    hardware = args.hardware
    mjcf_path = Path(__file__).parent.parent / "mjcf" / "g1.xml"
    if hardware:
        robot = g1_interface.G1HarwareInterface("enp58s0")
        robot.load_mjcf(str(mjcf_path)) # for computing FK
        mjModel = mujoco.MjModel.from_xml_path(str(mjcf_path))
    else:
        scene_path = Path(__file__).parent.parent / "mjcf" / "g1_with_floor.xml"
        robot = g1_interface.G1MujocoInterface(str(scene_path))
        robot.run(sync=args.sync)
        mjModel = mujoco.MjModel.from_xml_path(str(scene_path))
        print(f"timestep: {robot.get_timestep()}")

    mjData = mujoco.MjData(mjModel)
    mujoco.mj_forward(mjModel, mjData)

    mujoco_joint_names = [mjModel.joint(i).name for i in range(mjModel.njnt)]
    mujoco_joint_names.remove("floating_base_joint")
    robot = Articulation(
        robot,
        config["action"]["action_scaling"],
        config["default_joint_pos"],
        config["stiffness"],
        config["damping"],
    )

    observation_config = config["observation"]
    observation_groups = {}
    for group_name, group_config in observation_config.items():
        observation_groups[group_name] = []
        for observation_name, observation_config in group_config.items():
            observation_class = Observation.registry[observation_name]
            if observation_config is None:
                observation = observation_class(robot)
            else:
                observation = observation_class(robot, **observation_config)
            observation_groups[group_name].append(observation)
    
    def compute_observations():
        results = {}
        for group_name, group_observations in observation_groups.items():
            group_results = []
            for observation in group_observations:
                group_results.append(observation())
            results[group_name] = np.concatenate(group_results, axis=-1, dtype=np.float32)[None, ...]
        return results

    if args.rerun:
        meshes = extract_meshes(mjModel)
        print(len(meshes))
        rr.init("g1", recording_id="g1")
        rr.spawn()
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
    
    viewer = mujoco.viewer.launch_passive(mjModel, mjData)
    timer = Timer(0.02)
    for i in itertools.count():
        data = robot.data
        mjData.qpos[0:3] = data.root_pos_w
        mjData.qpos[3:7] = data.quaternion
        mjData.qpos[7:] = data.q
        mjData.qvel[6:] = data.dq
        mujoco.mj_forward(mjModel, mjData)

        inputs = onnx_module.dummy_input()
        inputs.update(compute_observations())
        action = onnx_module.forward(inputs)["action"]
        # action = onnx_module.forward(inputs)["linear_4"]
        robot.apply_action(action)

        if args.sync:
            decimation = int(0.02 / robot.robot.get_timestep())
            for _ in range(decimation):
                robot.robot.step()

        if i % 500 == 0:
            robot.reset()
        
        if args.rerun:
            rr.set_time("step", timestamp=i)
            # xpos = mjData.xpos[1:]
            # xquat = mjData.xquat[1:][:, [1, 2, 3, 0]]

            xpos = np.asarray(data.body_positions)
            xquat = np.asarray(data.body_quaternions)[:, [1, 2, 3, 0]]
            for ii, (body_name, mesh) in enumerate(meshes.items()):
                rr.log(
                    f"robot/{body_name}",
                    rr.Transform3D(translation=xpos[ii], quaternion=xquat[ii])
                )
        viewer.sync()
        timer.sleep()

