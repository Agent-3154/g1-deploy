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
from pathlib import Path
from collections import OrderedDict

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
    viewer = mujoco.viewer.launch_passive(mjModel, mjData)

    asset_meta_path = Path(__file__).parent.parent / "checkpoints" / "asset_meta.json"
    with open(asset_meta_path, "r") as f:
        asset_meta = json.load(f)
    
    isaac_joint_names = asset_meta["joint_names_isaac"]
    default_joint_pos = np.asarray(asset_meta["default_joint_pos"])
    joint_stiffness = np.asarray(asset_meta["stiffness"])
    joint_damping = np.asarray(asset_meta["damping"])

    mujoco_joint_names = [mjModel.joint(i).name for i in range(mjModel.njnt)]
    mujoco_joint_names.remove("floating_base_joint")

    isaac2mujoco = [isaac_joint_names.index(name) for name in mujoco_joint_names]
    mujoco2isaac = [mujoco_joint_names.index(name) for name in isaac_joint_names]
    
    robot.set_joint_stiffness(joint_stiffness[isaac2mujoco])
    robot.set_joint_damping(joint_damping[isaac2mujoco])

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
    
    for i in itertools.count():
        data = robot.get_data()
        mjData.qpos[0:3] = data.position
        mjData.qpos[3:7] = data.quaternion
        mjData.qpos[7:] = data.q
        mjData.qvel[6:] = data.dq
        mujoco.mj_forward(mjModel, mjData)
        if i % 1000 == 0:
            robot.reset(joint_pos=default_joint_pos[isaac2mujoco])

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

