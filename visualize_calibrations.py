import matplotlib.pyplot as plt
import numpy as np
import json
import os

# Path to saved pose data
calib_dir = "calibration_results"
json_path = os.path.join(calib_dir, "multi_camera_global_poses.json")

# Load JSON
with open(json_path, "r") as f:
    pose_data = json.load(f)

# Extract positions
camera_names = list(pose_data.keys())
positions = [np.array(pose_data[cam]["T"]).reshape(3,) for cam in camera_names]
positions = np.array(positions)

# Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], color='blue', s=50)

for i, cam in enumerate(camera_names):
    ax.text(positions[i, 0], positions[i, 1], positions[i, 2], cam, fontsize=9)

ax.set_title("3D Camera Positions")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.grid(True)
plt.tight_layout()
plt.show()