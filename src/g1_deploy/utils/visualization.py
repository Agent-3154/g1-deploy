import mujoco
import trimesh
import numpy as np
import matplotlib.pyplot as plt
from typing import List
from collections import OrderedDict


def plot_array(
    data: np.ndarray,
    num_plots: int,
    plot_names: List[str],
    dt: float = 0.02,
    figsize: tuple = None,
    save_path: str = None,
):
    """
    Plot array data with multiple subplots.
    
    Args:
        data: Array of shape (T, D) where D >= num_plots
        num_plots: Number of subplots to create
        plot_names: Names for each subplot (length should match num_plots)
        dt: Time step for x-axis (default 0.02s)
        figsize: Figure size tuple (width, height). Auto-calculated if None.
        save_path: If provided, save figure to this path instead of showing.
    
    Example:
        plot_array(
            data=joint_pos_history,
            num_plots=12,
            plot_names=["hip_pitch_left", "hip_pitch_right", ...],
        )
    """
    # Determine grid layout
    cols = min(4, num_plots)
    rows = (num_plots + cols - 1) // cols
    
    if figsize is None:
        figsize = (4 * cols, 3 * rows)
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize, sharex=True)
    if num_plots == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Get time axis
    num_steps = len(data)
    time_axis = np.arange(num_steps) * dt
    
    for i in range(num_plots):
        ax = axes[i]
        if i < data.shape[1]:
            ax.plot(time_axis, data[:, i], linewidth=1.2)
        
        ax.set_title(plot_names[i] if i < len(plot_names) else f"dim_{i}", fontsize=9)
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(num_plots, len(axes)):
        axes[i].set_visible(False)
    
    # Set x-axis label on bottom row
    for ax in axes[-(cols):]:
        if ax.get_visible():
            ax.set_xlabel("Time (s)")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()
    
    return fig


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

