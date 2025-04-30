import os
import numpy as np
import cv2
import json
from tqdm import tqdm

# === Config ===
root_dir = "captures"
calib_dir = "calibration_results"
camera_names = [f"picam{i}.local" for i in range(6)]
output_ply = "triangulated_points.ply"

def load_intrinsics(cam):
    data = np.load(os.path.join(calib_dir, f"{cam}_intrinsics.npz"))
    return data["K"], data["dist"], tuple(data["image_size"])

def load_global_poses():
    with open(os.path.join(calib_dir, "multi_camera_global_poses.json")) as f:
        poses = json.load(f)
    global_poses = {}
    for cam, pose in poses.items():
        R = np.array(pose["R"])
        T = np.array(pose["T"]).reshape(3, 1)
        global_poses[cam] = (R, T)
    return global_poses

def make_projection_matrix(K, R, T):
    RT = np.hstack((R, T))
    return K @ RT

def triangulate(P0, P1, pts0, pts1):
    points4D = cv2.triangulatePoints(P0, P1, pts0.T, pts1.T)
    points3D = points4D[:3] / points4D[3]
    return points3D.T

# === Load calibration ===
print("📥 Loading calibration data...")
intrinsics = {cam: load_intrinsics(cam) for cam in camera_names}
poses = load_global_poses()
proj_coords = {}

# === Load projector maps ===
for cam in camera_names:
    path = os.path.join(root_dir, cam, "projector_x_coords.npy")
    if not os.path.exists(path):
        print(f"❌ Missing projector_x_coords for {cam}")
        continue
    proj_coords[cam] = np.load(path)

# === Triangulate across all stereo pairs ===
all_points = []
print(f"🎯 Starting triangulation for {len(camera_names)} cameras...")

for i in range(len(camera_names)):
    for j in range(i + 1, len(camera_names)):
        camA = camera_names[i]
        camB = camera_names[j]

        if camA not in proj_coords or camB not in proj_coords:
            continue

        print(f"\n🔁 Calibrating stereo pair: {camA} ↔ {camB}")
        mapA = proj_coords[camA]
        mapB = proj_coords[camB]
        h, w = mapA.shape

        # Build projector-to-pixel maps
        proj_to_pix_A = {}
        for y in range(h):
            for x in range(w):
                val = mapA[y, x]
                if val > 0:
                    proj_to_pix_A[val] = (x, y)

        proj_to_pix_B = {}
        for y in range(h):
            for x in range(w):
                val = mapB[y, x]
                if val > 0:
                    proj_to_pix_B[val] = (x, y)

        # Find common projector coords
        common_vals = sorted(set(proj_to_pix_A.keys()) & set(proj_to_pix_B.keys()))
        matches = [(proj_to_pix_A[v], proj_to_pix_B[v]) for v in common_vals]

        print(f"🔗 Unique matches: {len(matches)}")
        if len(matches) < 50:
            print(f"⚠️ Too few matches — skipping {camA} ↔ {camB}")
            continue

        # Spatial distribution stats
        xs = [ptA[0] for ptA, _ in matches]
        ys = [ptA[1] for ptA, _ in matches]
        print(f"📐 X-distribution: {min(xs)} → {max(xs)} (Δ={max(xs)-min(xs)} px)")
        print(f"📐 Y-distribution: {min(ys)} → {max(ys)} (Δ={max(ys)-min(ys)} px)")

        # Triangulate
        ptsA = np.float32([m[0] for m in matches])
        ptsB = np.float32([m[1] for m in matches])

        K0, _, _ = intrinsics[camA]
        K1, _, _ = intrinsics[camB]
        R0, T0 = poses[camA]
        R1, T1 = poses[camB]
        baseline = np.linalg.norm(T0 - T1)
        print(f"🧭 Baseline: {baseline:.3f} meters")

        P0 = make_projection_matrix(K0, R0, T0)
        P1 = make_projection_matrix(K1, R1, T1)

        pts3D = triangulate(P0, P1, ptsA, ptsB)
        all_points.append(pts3D)

# === Save final .ply ===
if all_points:
    points = np.vstack(all_points)
    with open(output_ply, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\nend_header\n")
        for pt in points:
            f.write(f"{pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f}\n")
    print(f"\n✅ Saved {len(points)} triangulated points to {output_ply}")
else:
    print("❌ No points were triangulated.")
