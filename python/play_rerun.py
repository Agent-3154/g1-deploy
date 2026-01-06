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
from scipy.spatial.transform import Rotation


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
    # onnx_path = Path(__file__).parent.parent / "checkpoints" / "policy-12-30_15-28.onnx"
    # config_path = Path(__file__).parent.parent / "cfg" / "test.yaml"
    
    # onnx_module = ONNXModule(onnx_path)
    # print(onnx_module)

    # with open(config_path, "r") as f:
    #     config = yaml.load(f, Loader=yaml.FullLoader)
    
    hardware = args.hardware
    mjcf_path = Path(__file__).parent.parent / "mjcf" / "g1.xml"
    if hardware:
        robot = g1_interface.G1HarwareInterface("eth0")
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

    if args.rerun:
        meshes = extract_meshes(mjModel)
        print(len(meshes))
        rr.init("g1", recording_id="g1")
        # rr.spawn()
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
    
    timer = Timer(0.02)
    root_pos_w = []
    root_quat_w = []
    for i in itertools.count():
        
        if args.rerun:
            rr.set_time("step", timestamp=i)
            # xpos = mjData.xpos[1:]
            # xquat = mjData.xquat[1:][:, [1, 2, 3, 0]]
            data = robot.get_data()
            root_pos_w.append(np.asarray(data.root_pos_w))
            root_quat_w.append(np.asarray(data.quaternion))
            xpos = np.asarray(data.body_positions)
            xquat = np.asarray(data.body_quaternions)[:, [1, 2, 3, 0]]
            for ii, (body_name, mesh) in enumerate(meshes.items()):
                rr.log(
                    f"robot/{body_name}",
                    rr.Transform3D(translation=xpos[ii], quaternion=xquat[ii])
                )
            rr.log(
                f"world/origin",
                rr.Points3D(positions=np.array([0, 0, 0]), colors=[255, 0, 0], radii=0.1)
            )
            rr.log(
                f"world/frame",
                rr.Arrows3D(
                    origins=np.array([0, 0, 0]),
                    vectors=np.array([
                        [1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1],
                    ]),
                    colors=[
                        [255, 0, 0],
                        [0, 255, 0],
                        [0, 0, 255],
                    ],
                )
            )
            rr.log(
                f"robot/root_pos_w",
                rr.Points3D(positions=np.array(root_pos_w), colors=[0, 255, 0], radii=0.05)
            )
            unit_vectors = np.array([
                [1.0, 0.0, 0.0],  # X axis
                [0.0, 1.0, 0.0],  # Y axis
                [0.0, 0.0, 1.0]   # Z axis
            ])
            axis_length = 0.1
            
            origins_list = []
            vectors_list = []
            colors_list = []
            
            for pos, quat in zip(root_pos_w, root_quat_w):
                # quat is [w, x, y, z] format, convert to scipy format [x, y, z, w]
                quat_scipy = np.array([quat[1], quat[2], quat[3], quat[0]])
                # Create rotation object
                rot = Rotation.from_quat(quat_scipy)
                # Rotate unit vectors
                rotated_vectors = rot.apply(unit_vectors) * axis_length
                
                # Add three arrows for this position (X, Y, Z)
                for j, (vec, color) in enumerate(zip(rotated_vectors, 
                                                        [[255, 0, 0], [0, 255, 0], [0, 0, 255]])):
                    origins_list.append(pos)
                    vectors_list.append(vec)
                    colors_list.append(color)
            rr.log(
                f"robot/root_quat_w",
                rr.Arrows3D(
                    origins=np.array(origins_list),
                    vectors=np.array(vectors_list),
                    colors=np.array(colors_list),
                )
            )
        # viewer.sync()
        timer.sleep()

