import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_camera_poses_simple(json_path, arrow_fraction=0.02, view_elev=30, view_azim=45):
    """
    Simplified 3D visualization of camera poses.

    - Plots camera centers as black dots with labels.
    - Draws orientation axes using red (X), green (Y), blue (Z) arrows.
    - Centers and scales the view for clarity.

    Args:
        json_path (str): Path to the camera_poses JSON file.
        arrow_fraction (float): Fraction of scene diagonal for arrow length.
        view_elev (float): Elevation angle for 3D view.
        view_azim (float): Azimuth angle for 3D view.
    """
    # Load data
    with open(json_path, 'r') as f:
        cam_data = json.load(f)
    names = list(cam_data.keys())
    positions = np.array([cam_data[n]['T'] for n in names])
    rotations = {n: np.array(cam_data[n]['R']) for n in names}

    # Compute scene size
    mins = positions.min(axis=0)
    maxs = positions.max(axis=0)
    diag = np.linalg.norm(maxs - mins)
    arrow_len = diag * arrow_fraction

    # Create plot
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=view_elev, azim=view_azim)
    ax.set_title('Camera Poses')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Equal aspect
    ranges = maxs - mins
    max_range = ranges.max()
    mid = (maxs + mins) / 2
    ax.set_xlim(mid[0] - max_range/2, mid[0] + max_range/2)
    ax.set_ylim(mid[1] - max_range/2, mid[1] + max_range/2)
    ax.set_zlim(mid[2] - max_range/2, mid[2] + max_range/2)

    # Plot centers
    ax.scatter(positions[:,0], positions[:,1], positions[:,2], c='k', s=50)
    for i, n in enumerate(names):
        ax.text(*positions[i], n, color='k')

    # Draw axes
    for i, n in enumerate(names):
        p = positions[i]
        R = rotations[n]
        # X axis in red
        ax.quiver(*p, *(R[:,0]), length=arrow_len, color='r', normalize=True)
        # Y axis in green
        ax.quiver(*p, *(R[:,1]), length=arrow_len, color='g', normalize=True)
        # Z axis in blue
        ax.quiver(*p, *(R[:,2]), length=arrow_len, color='b', normalize=True)

    plt.tight_layout()
    plt.show()

# Example usage:
visualize_camera_poses_simple('calibration_results/camera_poses_ceres.json', arrow_fraction=0.02)
