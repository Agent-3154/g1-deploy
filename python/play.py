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
from pathlib import Path
from collections import OrderedDict
from observation import Observation, Articulation


class ONNXModule:
    def __init__(self, onnx_path: Path | str):
        self.onnx_path = onnx_path
        self.session = ort.InferenceSession(onnx_path)
        self.input_names = [input.name for input in self.session.get_inputs()]
        self.output_names = [output.name for output in self.session.get_outputs()]
        self.input_shapes = [input.shape for input in self.session.get_inputs()]
        self.output_shapes = [output.shape for output in self.session.get_outputs()]
        self.input_types = [input.type for input in self.session.get_inputs()]
        self.output_types = [output.type for output in self.session.get_outputs()]

    def __repr__(self) -> str:
        """Return a string representation of the ONNXModule."""
        lines = [
            f"ONNXModule(onnx_path={self.onnx_path!r}",
            f"  inputs: {len(self.input_names)}",
        ]
        for name, shape, dtype in zip(self.input_names, self.input_shapes, self.input_types):
            lines.append(f"    {name}: shape={shape}, dtype={dtype}")
        lines.append(f"  outputs: {len(self.output_names)}")
        for name, shape, dtype in zip(self.output_names, self.output_shapes, self.output_types):
            lines.append(f"    {name}: shape={shape}, dtype={dtype}")
        lines.append(")")
        return "\n".join(lines)
    
    def dummy_input(self) -> dict[str, np.ndarray]:
        return {input.name: np.zeros(input.shape, dtype=np.float32) for input in self.session.get_inputs()}
    
    def forward(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        # inputs = {name: value[None, ...] for name, value in inputs.items()}
        return self.session.run(self.output_names, inputs)


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
    onnx_path = Path(__file__).parent.parent / "checkpoints" / "motion.onnx"
    onnx_module = ONNXModule(onnx_path)
    print(onnx_module)

    config_path = Path(__file__).parent.parent / "cfg" / "config.yaml"
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

    asset_meta_path = Path(__file__).parent.parent / "checkpoints" / "asset_meta.json"
    with open(asset_meta_path, "r") as f:
        asset_meta = json.load(f)
    mujoco_joint_names = [mjModel.joint(i).name for i in range(mjModel.njnt)]
    mujoco_joint_names.remove("floating_base_joint")
    robot = Articulation(robot, asset_meta)

    observation_config = config["observation"]
    observation_groups = {}
    for group_name, group_config in observation_config.items():
        observation_groups[group_name] = []
        for observation_name, observation_config in group_config.items():
            observation_class = Observation.registry[observation_name]
            observation = observation_class(robot)
            observation_groups[group_name].append(observation)
    
    def compute_observations():
        results = {}
        for group_name, group_observations in observation_groups.items():
            group_results = []
            for observation in group_observations:
                group_results.append(observation())
            results[group_name] = np.concatenate(group_results, axis=-1, dtype=np.float32)[None, ...]
        return results
    
    inputs = onnx_module.dummy_input()
    inputs.update(compute_observations())
    outputs = onnx_module.forward(inputs)

    if hardware:
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
    for i in itertools.count():
        mjData.qpos[0:3] = robot.data.position
        mjData.qpos[3:7] = robot.data.quaternion
        mjData.qpos[7:] = robot.data.q
        mjData.qvel[6:] = robot.data.dq
        mujoco.mj_forward(mjModel, mjData)
        
        if i % 1000 == 0:
            robot.reset()

        # rr.set_time("step", timestamp=step)
        # # xpos = mjData.xpos[1:]
        # # xquat = mjData.xquat[1:][:, [1, 2, 3, 0]]
        # xpos = np.asarray(data.body_positions)[1:]
        # xquat = np.asarray(data.body_quaternions)[1:, [1, 2, 3, 0]]
        # for i, (body_name, mesh) in enumerate(meshes.items()):
        #     rr.log(
        #         f"robot/{body_name}",
        #         rr.Transform3D(translation=xpos[i], quaternion=xquat[i])
        #     )
        viewer.sync()
        time.sleep(0.01)

