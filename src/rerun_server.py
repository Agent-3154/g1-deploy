import mujoco
import rerun as rr
from utils import extract_meshes
from pathlib import Path

mjcf_path = Path(__file__).parent.parent / "mjcf" / "g1.xml"

rr.init("g1", recording_id="g1")
rr.spawn()
rr.set_time("step", timestamp=0.0)

model = mujoco.MjModel.from_xml_path(str(mjcf_path))
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

meshes = extract_meshes(model)
for i in range(model.nbody):
    if i not in meshes:
        continue
    mesh = meshes[i]
    body = model.body(i)

    pos = data.xpos[i]
    quat = data.xquat[i]
    rr.log(
        f"robot/{body.name}",
        rr.Mesh3D(
            vertex_positions=mesh.vertices,
            triangle_indices=mesh.faces,
            vertex_normals=mesh.vertex_normals,
        ),
    )

print("Done")

