import mujoco
import trimesh
import numpy as np
from typing import List
from collections import OrderedDict

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

