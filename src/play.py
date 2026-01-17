import sys
import time
import mujoco
import mujoco.viewer
import numpy as np
import argparse
import rerun as rr
import itertools
import yaml
import os
from pathlib import Path

import g1_deploy
import g1_deploy.observations
from g1_deploy.base import Observation, Articulation
from g1_deploy.policy import ONNXModule
from g1_deploy.utils import extract_meshes, Timer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hardware", action="store_true", help="Use hardware interface")
    parser.add_argument("--rerun_local", "-rl", action="store_true", help="Use rerun")
    parser.add_argument("--rerun_remote", "-rr", action="store_true", help="Use rerun")
    parser.add_argument("--sync", action="store_true", help="Use sync mode")
    return parser.parse_args()

CKPT_DIR = Path(__file__).parent.parent / "checkpoints"
CKPT_DIR.mkdir(exist_ok=True)
CFG_DIR = Path(__file__).parent.parent / "cfg"

# Example usage
if __name__ == "__main__":
    # Create a G1Interface instance
    # Replace "eth0" with your actual network interface name
    args = parse_args()
    
    onnx_path = CKPT_DIR / "policy-01-17_11-04.onnx"
    config_path = CFG_DIR / "test.yaml"

    # onnx_path = CKPT_DIR / "policy-12-30_14-47.onnx"
    # config_path = CFG_DIR / "cfg" / "loco_body.yaml"
    
    # onnx_path = CKPT_DIR / "policy-12-30_15-28.onnx"
    # config_path = CFG_DIR / "cfg" / "test.yaml"
    
    onnx_module = ONNXModule(onnx_path)
    print(onnx_module)

    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    hardware = args.hardware
    if hardware:
        mjcf_path = Path(__file__).parent.parent / "mjcf" / "g1.xml"
        robot = g1_deploy.G1HardwareInterface("wlp7s0", str(mjcf_path))
        mjModel = mujoco.MjModel.from_xml_path(str(mjcf_path))
    else:
        mjcf_path = Path(__file__).parent.parent / "mjcf" / "g1_with_floor.xml"
        robot = g1_deploy.G1MujocoInterface(str(mjcf_path))
        robot.run(sync=args.sync)
        mjModel = mujoco.MjModel.from_xml_path(str(mjcf_path))
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

    print(f"Current root_pos_w: {robot.root_pos_w}")
    print(f"Current root_quat_w: {robot.root_quat_w}")

    if config.get("command", None) is not None:
        from g1_deploy.commands.ref_motion import RefMotion
        command = RefMotion(
                        robot,
                        **config["command"]
                )
        command.enter(robot.root_pos_w, robot.root_quat_w)

    observation_config = config["observation"]
    observation_groups = {}
    for group_name, group_config in observation_config.items():
        observation_groups[group_name] = []
        for observation_name, observation_config in group_config.items():
            observation_class = Observation.registry[observation_name]
            kwargs = {"command": command}
            if observation_config is not None:
                kwargs.update(observation_config)
            observation = observation_class(robot, **kwargs)
            observation_groups[group_name].append(observation)
    
    def compute_observations():
        results = {}
        for group_name, group_observations in observation_groups.items():
            group_results = []
            for observation in group_observations:
                group_results.append(observation())
            results[group_name] = np.concatenate(group_results, axis=-1, dtype=np.float32)[None, ...]
        return results

    use_rerun = args.rerun_local or args.rerun_remote
    if use_rerun:
        meshes = extract_meshes(mjModel)
        print(len(meshes))
        rr.init("g1", recording_id="g1")
        if args.rerun_remote:
            ip = os.environ.get("RERUN_IP")
            port = os.environ.get("RERUN_PORT", 9876)
            rr.connect_grpc(f"rerun+http://{ip}:{port}/proxy")
        else:
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
    control_dt = 0.02
    timer = Timer(control_dt)

    last_real_time = time.perf_counter()
    last_sim_time = mjData.time

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
        robot.process_action(action)

        if args.sync:
            decimation = int(control_dt / robot.robot.get_timestep())
            for _ in range(decimation):
                robot.robot.step()
                robot.apply_action(0.8)
        else:
            robot.apply_action(0.8)

        mjData.time += control_dt
        if i % 100 == 0:
            current_real_time = time.perf_counter()
            current_sim_time = mjData.time
            
            real_time_delta = current_real_time - last_real_time
            sim_time_delta = current_sim_time - last_sim_time
            
            realtime_ratio = sim_time_delta / real_time_delta
            fps = 100 / real_time_delta
            
            print(f"FPS: {fps:.1f} | Sim/Real ratio: {realtime_ratio:.1f}")
            
            last_real_time = current_real_time
            last_sim_time = current_sim_time
        
        if use_rerun:
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
            rr.log(
                f"ground",
                rr.Boxes3D(
                    centers=np.array([0, 0, -0.01]),
                    half_sizes=np.array([10, 10, 0.02]),
                    colors=np.array([255, 255, 255]),
                    fill_mode="solid"
                )
            )
        viewer.sync()
        timer.sleep()

