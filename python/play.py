import sys
import time
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
    robot = g1_interface.G1Interface("enp58s0")
    mjcf_path = Path(__file__).parent.parent / "mjcf" / "g1.xml"
    robot.load_mjcf(str(mjcf_path))
    
    while True:
        time.sleep(1)
        data = robot.get_data()
        print(f"Joint positions (q): {data.q}")
        print(f"Joint velocities (dq): {data.dq}")
        print(f"Joint torques (tau): {data.tau}")
        print(f"Quaternion: {data.quaternion}")
        print(f"RPY: {data.rpy}")
        print(f"Angular velocity (omega): {data.omega}")

