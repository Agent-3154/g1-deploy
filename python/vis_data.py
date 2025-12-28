import h5py
import mujoco
import mujoco.viewer
import numpy as np
import argparse
from pathlib import Path
from timerfd import Timer


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize recorded robot trajectory")
    parser.add_argument("trajectory_file", type=str, help="Path to trajectory HDF5 file")
    parser.add_argument("--mjcf", type=str, default=None, help="Path to MuJoCo XML file (default: g1_with_floor.xml)")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier (default: 1.0)")
    parser.add_argument("--start", type=int, default=0, help="Start frame index (default: 0)")
    parser.add_argument("--end", type=int, default=None, help="End frame index (default: all)")
    return parser.parse_args()


def load_trajectory(trajectory_file):
    """Load trajectory data from HDF5 file."""
    with h5py.File(trajectory_file, "r") as f:
        root_quat_w = f["root_quat_w"][:]
        joint_pos = f["joint_pos"][:]
        joint_vel = f["joint_vel"][:]
        is_user_control = f["is_user_control"][:]
    
    print(f"Loaded trajectory with {len(root_quat_w)} frames")
    print(f"  - Root quaternion shape: {root_quat_w.shape}")
    print(f"  - Joint positions shape: {joint_pos.shape}")
    print(f"  - Joint velocities shape: {joint_vel.shape}")
    print(f"  - User control frames: {np.sum(is_user_control)} / {len(is_user_control)}")
    
    return {
        "root_quat_w": root_quat_w,
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
        "is_user_control": is_user_control,
    }


def visualize_trajectory(trajectory_file, mjcf_path=None, speed=1.0, start=0, end=None):
    """Visualize the recorded trajectory in MuJoCo viewer."""
    
    # Load trajectory data
    data = load_trajectory(trajectory_file)
    
    # Determine MuJoCo model path
    if mjcf_path is None:
        script_dir = Path(__file__).parent
        mjcf_path = script_dir.parent / "mjcf" / "g1_with_floor.xml"
    
    mjcf_path = Path(mjcf_path)
    if not mjcf_path.exists():
        raise FileNotFoundError(f"MuJoCo model not found: {mjcf_path}")
    
    # Load MuJoCo model
    mjModel = mujoco.MjModel.from_xml_path(str(mjcf_path))
    mjData = mujoco.MjData(mjModel)
    
    # Determine frame range
    num_frames = len(data["root_quat_w"])
    end_frame = end if end is not None else num_frames
    end_frame = min(end_frame, num_frames)
    
    if start >= num_frames:
        raise ValueError(f"Start frame {start} is beyond trajectory length {num_frames}")
    
    print(f"\nVisualizing frames {start} to {end_frame-1} (total: {end_frame-start} frames)")
    print(f"Playback speed: {speed}x")
    
    # Initialize viewer
    viewer = mujoco.viewer.launch_passive(mjModel, mjData)
    timer = Timer(0.02 / speed)  # Adjust timer based on playback speed
    
    # Playback loop
    try:
        for frame_idx in range(start, end_frame):
            # Get data for this frame
            root_quat = data["root_quat_w"][frame_idx]
            q = data["joint_pos"][frame_idx]
            dq = data["joint_vel"][frame_idx]
            user_control = data["is_user_control"][frame_idx]
            
            # Set root position (not recorded, so use origin or previous position)
            # Note: root position was not recorded, so we use the previous frame's position
            # or start at origin. For better visualization, you might want to integrate
            # velocities or record root_pos_w in future recordings.
            if frame_idx == start:
                root_pos = np.zeros(3)
            else:
                # Keep previous position (or you could integrate velocities)
                root_pos = mjData.qpos[0:3].copy()
            
            # Set MuJoCo state
            mjData.qpos[0:3] = root_pos  # Root position (approximate)
            mjData.qpos[3:7] = root_quat  # Root quaternion
            mjData.qpos[7:] = q  # Joint positions
            
            mjData.qvel[6:] = dq  # Joint velocities (root velocities not recorded)
            
            # Forward kinematics
            mujoco.mj_forward(mjModel, mjData)
            
            # Update viewer
            viewer.sync()
            
            # Print status every 100 frames
            if frame_idx % 100 == 0:
                control_type = "USER" if user_control else "AUTO"
                print(f"Frame {frame_idx}/{end_frame-1} ({control_type})")
            
            timer.sleep()
        
        print(f"\nPlayback complete! Reached frame {end_frame-1}")
        
    except KeyboardInterrupt:
        print("\nPlayback interrupted by user")
    finally:
        viewer.close()


if __name__ == "__main__":
    args = parse_args()
    
    trajectory_file = Path(args.trajectory_file)
    if not trajectory_file.exists():
        raise FileNotFoundError(f"Trajectory file not found: {trajectory_file}")
    
    visualize_trajectory(
        trajectory_file=trajectory_file,
        mjcf_path=args.mjcf,
        speed=args.speed,
        start=args.start,
        end=args.end,
    )
