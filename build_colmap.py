import numpy as np
import cv2
import os
import json

output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

intrinsics_dir = "calibration_results"
camera_names = [f"picam{i}.local" for i in range(6)]
out = open(f"{output_dir}/cameras.txt","w")
out.write('# Camera list with one line of data per camera:\n')
out.write("# CameraID, model, width, height, fx, fy, cx, cy, k1\n")
out.write(f'# Number of cameras: {len(camera_names)}\n')

# assume all images same size; load one image to get W,H
img = cv2.imread(os.path.join("captures", camera_names[0], "0.jpg"))
h, w = img.shape[:2]

for cam_id, cam in enumerate(camera_names, start=1):
    data = np.load(os.path.join(intrinsics_dir, f"{cam}_intrinsics_refined_full.npz"))
    K    = data["K"]
    dist = data["dist"].ravel()  # [k1,k2,p1,p2,k3]
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]
    params = f"{fx:.6f} {fy:.6f} {cx:.6f} {cy:.6f} {dist[0]:.6e} {dist[1]:.6e} {dist[2]:.6e} {dist[3]:.6e}"
    out.write(f"{cam_id} OPENCV {w} {h} {params}\n")

out.close()


poses = json.load(open("results/refined_camera_poses.json"))
camera_names = sorted(poses.keys())

out = open(f"{output_dir}/images.txt","w")
out.write('# Image list for six‑camera rig\n')
out.write("# ImageID, qw, qx, qy, qz, tx, ty, tz, cameraID, image_name\n")

img_id = 1
for cam_id, cam in enumerate(camera_names, start=1):
    # assume N images per cam, name them 0000.jpg,0001.jpg…
    import glob
    files = sorted(glob.glob(f"captures/{cam}/*.jpg"))
    for fname in files:
        R = np.array(poses[cam]["R"])
        T = np.array(poses[cam]["T"]).reshape(3,1)
        # convert R→ quaternion (world→camera):
        # COLMAP expects q s.t. X_cam = R * X_world + T
        # we already have R and T in that form
        from scipy.spatial.transform import Rotation as Rsc
        rot = Rsc.from_matrix(R)
        qx, qy, qz, qw = rot.as_quat()  # returns x,y,z,w
        out.write(f"{img_id} {qw:.6f} {qx:.6f} {qy:.6f} {qz:.6f} "
                  f"{T[0,0]:.6f} {T[1,0]:.6f} {T[2,0]:.6f} "
                  f"{cam_id} {cam}/{os.path.basename(fname)}\n")
        out.write("\n")
        img_id += 1

out.close()

out = open(f"{output_dir}/points3D.txt","w")
out.write('# 3D point list with one line per point:\n')
out.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK_LENGTH, (IMAGE_ID, POINT2D_IDX)")
out.close()