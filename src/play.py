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
from g1_deploy.utils import extract_meshes, Timer, quat_from_euler_xyz, quat_mul, plot_array


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, default="loco", help="Config file name")
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
    
    config_path = CFG_DIR / f"{args.config}.yaml"
    
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    onnx_path = CKPT_DIR / config["onnx_path"]
    onnx_module = ONNXModule(onnx_path)
    print(onnx_module)
    
    hardware = args.hardware
    if hardware:
        mjcf_path = Path(__file__).parent.parent / "mjcf" / "g1.xml"
        mjModel = mujoco.MjModel.from_xml_path(str(mjcf_path))
    else:
        mjcf_path = Path(__file__).parent.parent / "mjcf" / "g1_with_floor.xml"
        mjModel = mujoco.MjModel.from_xml_path(str(mjcf_path))

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
        for body_name, mesh in meshes.items():
            rr.log(
                f"ref/{body_name}",
                rr.Mesh3D(
                    vertex_positions=mesh.vertices,
                    triangle_indices=mesh.faces,
                    vertex_normals=mesh.vertex_normals,
                    albedo_factor=[0.2, 0.6, 1.0],  # Blue color for ref motion
                ),
            )

    mjData = mujoco.MjData(mjModel)
    mujoco.mj_forward(mjModel, mjData)

    if hardware:
        robot = g1_deploy.G1HardwareInterface("eth0", str(mjcf_path))
    else:
        robot = g1_deploy.G1MujocoInterface(str(mjcf_path))
        # Randomize initial x, y position
        init_pos = np.array(robot.get_data().root_pos_w)
        init_pos[0] += np.random.uniform(-0.5, 0.5)  # x: +/- 0.5m
        init_pos[1] += np.random.uniform(-0.5, 0.5)  # y: +/- 0.5m
        # Randomize initial yaw
        init_quat = np.array(robot.get_data().quaternion)
        yaw = np.random.uniform(-1.5, 1.5)  # +/- ~30 degrees
        delta_quat = quat_from_euler_xyz(0, 0, yaw)
        init_quat = quat_mul(delta_quat, init_quat)
        robot.set_root_pose(list(init_pos), list(init_quat))
        robot.run(sync=args.sync)
        print(f"timestep: {robot.get_timestep()}")

    robot = Articulation(
        robot,
        config["action"]["action_scaling"],
        config["default_joint_pos"],
        config["stiffness"],
        config["damping"],
    )

    print(f"Current root_pos_w: {robot.root_pos_w}")
    print(f"Current root_quat_w: {robot.root_quat_w}")

    command = None
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
            kwargs = {"command": command} if command is not None else {}
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

    viewer = mujoco.viewer.launch_passive(mjModel, mjData)
    control_dt = 0.02
    timer = Timer(control_dt)

    last_real_time = time.perf_counter()

    # Data collection for visualization
    joint_pos_history = []
    applied_action_history = []

    try:
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
            alpha = 0.8

            if args.sync:
                decimation = int(control_dt / robot.robot.get_timestep())
                for _ in range(decimation):
                    robot.apply_action(alpha)
                    robot.robot.step()
            else:
                robot.apply_action(alpha)
            robot.t += 1

            # Collect data for visualization
            joint_pos_history.append(robot.joint_pos.copy())
            applied_action_history.append(robot.applied_action.copy())

            if i % 50 == 0:
                current_real_time = time.perf_counter()
                real_time_delta = current_real_time - last_real_time
                fps = 50 / real_time_delta
                
                print(f"FPS: {fps:.1f}")

                last_real_time = current_real_time
            
            # if i == command.motion_length - 1:
            #     break
            
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
                
                # Visualize ref motion if command exists
                if command is not None:
                    frame_idx = min(robot.t, command.motion_length - 1)
                    # Get ref motion body pos/quat in Isaac order, convert to MuJoCo order
                    ref_body_pos_isaac = command.body_pos_w[frame_idx]  # (num_bodies, 3)
                    ref_body_quat_isaac = command.body_quat_w[frame_idx]  # (num_bodies, 4)
                    ref_body_pos_mujoco = ref_body_pos_isaac[robot.body_indexing.isaac2mujoco]
                    ref_body_quat_mujoco = ref_body_quat_isaac[robot.body_indexing.isaac2mujoco][:, [1, 2, 3, 0]]
                    for ii, body_name in enumerate(meshes.keys()):
                        rr.log(
                            f"ref/{body_name}",
                            rr.Transform3D(translation=ref_body_pos_mujoco[ii], quaternion=ref_body_quat_mujoco[ii])
                        )
                
                q = np.asarray(robot.joint_pos)
                for j, name in enumerate(g1_deploy.utils.constants.JOINT_NAMES_ISAAC):
                    rr.log(
                        f"joint_pos/{name}",
                        rr.Scalars(q[j])
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
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Saving data...")
        viewer.close()

    # joint_pos_history = np.array(joint_pos_history)
    # applied_action_history = np.array(applied_action_history)
    # save_path = Path("mjc_loco.npz")
    # np.savez(
    #     save_path, 
    #     joint_pos_history=joint_pos_history, 
    #     applied_action_history=applied_action_history
    # )
    # print(f"Saved mujoco data to {save_path}")
    # print(f"Total steps: {i}")
    # print(f"Joint positions shape: {joint_pos_history.shape}")
    # print(f"Applied action shape: {applied_action_history.shape}")

    # from g1_deploy.utils.constants import JOINT_NAMES_MUJOCO
    # plot_array(
    #     data=joint_pos_history,
    #     num_plots=joint_pos_history.shape[1],
    #     plot_names=JOINT_NAMES_MUJOCO,
    #     dt=control_dt,
    # )
    # plot_array(
    #     data=applied_action_history,
    #     num_plots=applied_action_history.shape[1],
    #     plot_names=robot.action_joint_names,
    #     dt=control_dt,
    # )