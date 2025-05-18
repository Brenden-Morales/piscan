import numpy as np
import cv2
import os
import json
from scipy.spatial.transform import Rotation as Rsc
import glob

output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

intrinsics_dir = "calibration_results"
camera_names = [f"picam{i}.local" for i in range(6)]

# -----------------------------------------------------------------------------
# 1. Write cameras.txt
# -----------------------------------------------------------------------------
with open(f"{output_dir}/cameras.txt", "w") as out:
    out.write('# Camera list with one line of data per camera:\n')
    out.write("# CameraID, model, width, height, fx, fy, cx, cy, k1, k2, p1, p2\n")
    out.write(f'# Number of cameras: {len(camera_names)}\n')

    img = cv2.imread(os.path.join("captures", camera_names[0], "0.jpg"))
    h, w = img.shape[:2]

    for cam_id, cam in enumerate(camera_names, start=1):
        data = np.load(os.path.join(intrinsics_dir, f"{cam}_intrinsics_refined_full.npz"))
        K = data["K"]
        dist = data["dist"].ravel()  # [k1, k2, p1, p2, k3]
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        params = f"{fx:.6f} {fy:.6f} {cx:.6f} {cy:.6f} {dist[0]:.6e} {dist[1]:.6e} {dist[2]:.6e} {dist[3]:.6e}"
        out.write(f"{cam_id} OPENCV {w} {h} {params}\n")

# -----------------------------------------------------------------------------
# 2. Write images.txt
# -----------------------------------------------------------------------------
poses = json.load(open("results/refined_camera_poses.json"))

with open(f"{output_dir}/images.txt", "w") as out:
    out.write('# Image list for six‑camera rig\n')
    out.write("# ImageID, qw, qx, qy, qz, tx, ty, tz, cameraID, image_name\n")

    img_id = 1
    for cam_id, cam in enumerate(camera_names, start=1):
        files = sorted(glob.glob(f"captures/{cam}/*.jpg"))
        rot_mat = np.array(poses[cam]["R"])
        T = np.array(poses[cam]["T"]).reshape(3, 1)

        rot = Rsc.from_matrix(rot_mat)
        qx, qy, qz, qw = rot.as_quat()  # returns [x, y, z, w]

        for fname in files:
            out.write(f"{img_id} {qw:.6f} {qx:.6f} {qy:.6f} {qz:.6f} "
                      f"{T[0, 0]:.6f} {T[1, 0]:.6f} {T[2, 0]:.6f} "
                      f"{cam_id} {cam}/{os.path.basename(fname)}\n\n")
            img_id += 1

# -----------------------------------------------------------------------------
# 3. Write empty points3D.txt
# -----------------------------------------------------------------------------
with open(f"{output_dir}/points3D.txt", "w") as out:
    out.write('# 3D point list with one line per point:\n')
    out.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK_LENGTH, (IMAGE_ID, POINT2D_IDX)\n")
    out.write("\n")

# -----------------------------------------------------------------------------
# 4. Write rig.txt
# -----------------------------------------------------------------------------
with open("multi_camera_global_poses.json") as f:
    rig_poses = json.load(f)

camera_id_map = {name: idx + 1 for idx, name in enumerate(camera_names)}

with open(f"{output_dir}/rig.txt", "w") as f:
    for cam_name in camera_names:
        pose = rig_poses[cam_name]
        rot_mat = np.array(pose["R"])
        T_vec = np.array(pose["T"]).flatten()

        q_xyzw = Rsc.from_matrix(rot_mat).as_quat()
        q_colmap = [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]  # [w, x, y, z]

        camera_id = camera_id_map[cam_name]
        image_prefix = f"{cam_name}/"
        line = f"{image_prefix} {camera_id} " + " ".join(f"{v:.8f}" for v in q_colmap + T_vec.tolist()) + "\n"
        f.write(line)
