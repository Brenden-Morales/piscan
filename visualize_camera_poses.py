import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # this registers the 3D projection

# Load the BA-refined camera poses
with open('results/camera_poses_ba.json', 'r') as f:
    poses = json.load(f)

camera_names = list(poses.keys())

# Extract positions
positions = np.array([poses[c]['T'] for c in camera_names])  # shape (N,3)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Scatter the camera centers
ax.scatter(positions[:,0], positions[:,1], positions[:,2])

# Compute axis length (10% of the bounding box diagonal)
diag = np.linalg.norm(positions.max(axis=0) - positions.min(axis=0))
axis_len = 0.1 * diag

# Draw each camera’s orientation axes
for cam in camera_names:
    R = np.array(poses[cam]['R'])      # 3×3 rotation matrix
    T = np.array(poses[cam]['T']).ravel()
    ax.text(*T, cam)                   # label
    # for each local axis (columns of R)
    for i in range(3):
        axis_vec = R[:,i] * axis_len
        ax.quiver(T[0], T[1], T[2],
                  axis_vec[0], axis_vec[1], axis_vec[2],
                  length=1, normalize=False)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('Bundle-Adjusted Camera Poses')
plt.show()
