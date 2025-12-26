import sys
import time
import mujoco
import mujoco.viewer
import numpy as np
from pathlib import Path

# Add the build directory to Python path to import the compiled module
build_dir = Path(__file__).parent.parent / "build"
sys.path.insert(0, str(build_dir))

# Import the g1_interface module
import g1_interface

# Example usage
if __name__ == "__main__":
    # Create a G1Interface instance
    # Replace "eth0" with your actual network interface name
    hardware = False
    mjcf_path = Path(__file__).parent.parent / "mjcf" / "g1.xml"
    if hardware:
        robot = g1_interface.G1HarwareInterface("enp58s0")
        robot.load_mjcf(str(mjcf_path)) # for computing FK
    else:
        robot = g1_interface.G1MujocoInterface(str(mjcf_path))
        robot.run_async()
        print(f"timestep: {robot.get_timestep()}")
    
    mjModel = mujoco.MjModel.from_xml_path(str(mjcf_path))
    mjData = mujoco.MjData(mjModel)
    viewer = mujoco.viewer.launch_passive(mjModel, mjData)
    
    while True:
        data = robot.get_data()
        mjData.qpos[7:] = data.q
        mjData.qvel[6:] = data.dq
        mujoco.mj_forward(mjModel, mjData)
        viewer.sync()
        time.sleep(0.01)

