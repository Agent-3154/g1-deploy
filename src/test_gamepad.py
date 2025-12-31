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
import datetime
import signal
import h5py
from pathlib import Path
from collections import OrderedDict
from observation import Observation, Articulation
from timerfd import Timer


class TorchJitModule:
    def __init__(self, torchscript_path: Path | str):
        self.torchscript_path = torchscript_path
        self.module = torch.jit.load(torchscript_path)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        outputs = self.module(input)
        return outputs


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

    torchscript_path = Path(__file__).parent.parent / "checkpoints" / "policy_29dof.pt"
    torchscript_module = TorchJitModule(torchscript_path)
    config_path = Path(__file__).parent.parent / "cfg" / "loco.yaml"

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
        robot.run_async()
        mjModel = mujoco.MjModel.from_xml_path(str(scene_path))
        print(f"timestep: {robot.get_timestep()}")

    mjData = mujoco.MjData(mjModel)
    mujoco.mj_forward(mjModel, mjData)
    robot = Articulation(
        robot,
        config["action"]["action_scaling"],
        config["default_joint_pos"],
        config["stiffness"],
        config["damping"],
    )

    datetime_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_path = Path(__file__).parent / f"trajectory-{datetime_str}.h5"
    dataset = []
    
    # Flush configuration
    FLUSH_INTERVAL_SAMPLES = 100  # Flush every 100 samples (2s at 50Hz)
    
    # Initialize h5py file (keep it open for efficient appending)
    h5_file = h5py.File(str(dataset_path), mode='a')
    
    def flush_to_disk():
        """Save accumulated dataset to disk using h5py, appending efficiently."""
        if len(dataset) == 0:
            return
        
        # Convert list of dicts to arrays
        new_root_quat_w = np.array([d["root_quat_w"] for d in dataset])
        new_joint_pos = np.array([d["joint_pos"] for d in dataset])
        new_joint_vel = np.array([d["joint_vel"] for d in dataset])
        new_is_user_control = np.array([d["is_user_control"] for d in dataset])
        
        # Get or create datasets
        def get_or_create_dataset(name, new_data):
            if name in h5_file:
                return h5_file[name]
            else:
                # Create resizable dataset with initial shape
                maxshape = (None, *new_data.shape[1:]) if new_data.ndim > 1 else (None,)
                shape = (0, *new_data.shape[1:]) if new_data.ndim > 1 else (0,)
                return h5_file.create_dataset(
                    name,
                    shape=shape,
                    maxshape=maxshape,
                    dtype=new_data.dtype,
                    chunks=(FLUSH_INTERVAL_SAMPLES, *shape[1:]),
                    compression='gzip',
                    compression_opts=4
                )
        
        root_quat_w_ds = get_or_create_dataset("root_quat_w", new_root_quat_w)
        joint_pos_ds = get_or_create_dataset("joint_pos", new_joint_pos)
        joint_vel_ds = get_or_create_dataset("joint_vel", new_joint_vel)
        is_user_control_ds = get_or_create_dataset("is_user_control", new_is_user_control)
        
        # Append new data (h5py handles resizing efficiently)
        start_idx = root_quat_w_ds.shape[0]
        end_idx = start_idx + len(new_root_quat_w)
        
        # Resize datasets and assign new data
        root_quat_w_ds.resize((end_idx, *root_quat_w_ds.shape[1:]))
        joint_pos_ds.resize((end_idx, *joint_pos_ds.shape[1:]))
        joint_vel_ds.resize((end_idx, *joint_vel_ds.shape[1:]))
        is_user_control_ds.resize((end_idx,))
        
        root_quat_w_ds[start_idx:end_idx] = new_root_quat_w
        joint_pos_ds[start_idx:end_idx] = new_joint_pos
        joint_vel_ds[start_idx:end_idx] = new_joint_vel
        is_user_control_ds[start_idx:end_idx] = new_is_user_control
        
        # Flush to ensure data is written to disk
        h5_file.flush()
        
        total_samples = root_quat_w_ds.shape[0]
        print(f"Flushed {len(dataset)} new samples (total: {total_samples} samples) to {dataset_path}")
        dataset.clear()
    
    # Register signal handler for graceful shutdown
    def signal_handler(sig, frame):
        print("\nInterrupt received, flushing data to disk...")
        flush_to_disk()
        h5_file.close()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    viewer = mujoco.viewer.launch_passive(mjModel, mjData)
    timer = Timer(0.02)

    try:
        start_time = time.perf_counter()
        for i in itertools.count():
            data = robot.data
            dataset.append(
                {
                    "root_quat_w": np.asarray(data.quaternion),
                    "joint_pos": np.asarray(data.q),
                    "joint_vel": np.asarray(data.dq),
                    "is_user_control": np.asarray(data.is_user_control),
                }
            )
            
            # Periodic flush: every N samples
            should_flush = i % FLUSH_INTERVAL_SAMPLES == 0
            
            if should_flush:
                freq = FLUSH_INTERVAL_SAMPLES / (time.perf_counter() - start_time)
                print(f"Loop frequency: {freq} Hz")
                flush_to_disk()
                start_time = time.perf_counter()
            
            mjData.qpos[0:3] = data.root_pos_w
            mjData.qpos[3:7] = data.quaternion
            mjData.qpos[7:] = data.q
            mjData.qvel[6:] = data.dq

            mujoco.mj_forward(mjModel, mjData)

            gamepad = robot.robot.get_gamepad()

            viewer.sync()
            timer.sleep()
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received, flushing data to disk...")
        flush_to_disk()
    finally:
        # Final flush of any remaining data and close file
        if len(dataset) > 0:
            print("Flushing remaining data to disk...")
            flush_to_disk()
        h5_file.close()
        print(f"Closed HDF5 file: {dataset_path}")

