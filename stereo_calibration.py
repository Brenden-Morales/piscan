# -*- coding: utf-8 -*-
"""
Stereo Calibration with Baseline Filtering & MST Pose Propagation
"""
import cv2
import cv2.aruco as aruco
import numpy as np
import os
import glob
import json
from itertools import combinations
import networkx as nx

# === Charuco board parameters (Side B: 16×16, DICT_4X4_1000) ===
paper_w_in = paper_h_in = 24.0
safe_margin_mm = 20.0
mm_per_inch = 25.4
page_w_mm = paper_w_in * mm_per_inch - 2 * safe_margin_mm

squaresX = squaresY = 16
square_size_m = (page_w_mm / squaresX) / 1000.0
marker_size_m = square_size_m * 0.75

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)
board = aruco.CharucoBoard(
    (squaresX, squaresY),
    squareLength=square_size_m,
    markerLength=marker_size_m,
    dictionary=aruco_dict
)
chessboard_3D = board.getChessboardCorners()

# === I/O and settings ===
camera_names = [f"picam{i}.local" for i in range(6)]
calib_dir    = "calibration_results"
captures_dir = "captures"
image_ext    = "*.jpg"

# Stereo-calib & filtering thresholds:
min_common_corners = 20    # min shared corners per view
min_valid_frames  = 5     # min frames for stereoCalibrate
max_rms_error     = 1.5   # px
min_baseline_m    = 0.01  # m, skip tiny baselines
max_baseline_m    = 0.8  # m, physically plausible maximum

os.makedirs(calib_dir, exist_ok=True)

# --- Load intrinsics ---
print("\n📅 Loading camera intrinsics...")
intrinsics = {}
for cam in camera_names:
    data = np.load(os.path.join(calib_dir, f"{cam}_intrinsics_refined_full.npz"))
    intrinsics[cam] = {"K": data["K"], "dist": data["dist"], "image_size": tuple(data["image_size"])}
print("✅ Intrinsics loaded for all cameras!\n")

# --- Stereo calibration for each pair ---
pair_results = []
print("🔍 Evaluating all camera pairs...")
for camA, camB in combinations(camera_names, 2):
    print(f"\n🔁 Trying: {camA} ↔ {camB}")
    K0, dist0, img_size0 = intrinsics[camA]["K"], intrinsics[camA]["dist"], intrinsics[camA]["image_size"]
    K1, dist1, img_size1 = intrinsics[camB]["K"], intrinsics[camB]["dist"], intrinsics[camB]["image_size"]
    # require same image size
    if img_size0 != img_size1:
        print("⚠️ Skipping — image size mismatch")
        continue

    imgsA = sorted(glob.glob(os.path.join(captures_dir, camA, image_ext)))
    imgsB = sorted(glob.glob(os.path.join(captures_dir, camB, image_ext)))
    if len(imgsA) != len(imgsB):
        print("⚠️ Skipping — frame count mismatch")
        continue

    obj_pts, img_ptsA, img_ptsB = [], [], []
    for pathA, pathB in zip(imgsA, imgsB):
        imgA = cv2.imread(pathA); grayA = cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY)
        imgB = cv2.imread(pathB); grayB = cv2.cvtColor(imgB, cv2.COLOR_BGR2GRAY)
        cA, idA, _ = aruco.detectMarkers(grayA, aruco_dict)
        cB, idB, _ = aruco.detectMarkers(grayB, aruco_dict)
        if idA is None or idB is None:
            continue
        _, ccA, cidA = aruco.interpolateCornersCharuco(cA, idA, grayA, board)
        _, ccB, cidB = aruco.interpolateCornersCharuco(cB, idB, grayB, board)
        if cidA is None or cidB is None:
            continue
        common = np.intersect1d(cidA.flatten(), cidB.flatten())
        if len(common) < min_common_corners:
            continue
        pts3d, ptsA, ptsB = [], [], []
        for cid in common:
            idxA = np.where(cidA==cid)[0][0]
            idxB = np.where(cidB==cid)[0][0]
            pts3d.append(chessboard_3D[int(cid)])
            ptsA.append(ccA[idxA][0]); ptsB.append(ccB[idxB][0])
        obj_pts.append(np.array(pts3d, dtype=np.float32))
        img_ptsA.append(np.array(ptsA, dtype=np.float32))
        img_ptsB.append(np.array(ptsB, dtype=np.float32))

    if len(obj_pts) < min_valid_frames:
        print(f"⚠️ Skipping — only {len(obj_pts)} valid frames")
        continue

    # stereoCalibrate
    retval, _, _, _, _, R, T, _, _ = cv2.stereoCalibrate(
        objectPoints=obj_pts,
        imagePoints1=img_ptsA,
        imagePoints2=img_ptsB,
        cameraMatrix1=K0,
        distCoeffs1=dist0,
        cameraMatrix2=K1,
        distCoeffs2=dist1,
        imageSize=img_size0,
        flags=cv2.CALIB_FIX_INTRINSIC,
        criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,100,1e-5)
    )
    baseline = np.linalg.norm(T)
    print(f"✅ RMS: {retval:.4f}, baseline: {baseline*100:.1f} cm, frames: {len(obj_pts)}")

    # filter by RMS and baseline
    if retval > max_rms_error:
        print("⚠️ Rejected: RMS too high")
        continue
    if baseline < min_baseline_m:
        print("⚠️ Rejected: baseline too small")
        continue
    if baseline > max_baseline_m:
        print("⚠️ Rejected: baseline too large")
        continue

    # save
    out_fn = os.path.join(calib_dir, f"stereo_calibration_{camA}_{camB}.npz")
    np.savez(out_fn, R=R, T=T, E=None, F=None, error=retval)
    pair_results.append((retval, camA, camB, R, T))
    print(f"📁 Saved: {out_fn}")

# === Build MST and propagate ===
print("\n🧠 Building pose graph with lowest-error links…")
G = nx.Graph()
for err, A, B, R, T in pair_results:
    G.add_edge(A, B, weight=err, R=R, T=T)
mst = nx.minimum_spanning_tree(G)

# propagate from camera0
global_RTs = {camera_names[0]:(np.eye(3), np.zeros((3,1)))}
def propagate(cam, vis):
    vis.add(cam)
    for nbr in mst.neighbors(cam):
        if nbr in vis: continue
        edge = mst[cam][nbr]
        R0, T0 = global_RTs[cam]
        Rn = edge['R'] @ R0
        Tn = edge['R'] @ T0 + edge['T']
        global_RTs[nbr] = (Rn, Tn)
        propagate(nbr, vis)
propagate(camera_names[0], set())

# save global poses
pose_json = {cam:{'R':R.tolist(),'T':T.flatten().tolist()} for cam,(R,T) in global_RTs.items()}
with open(os.path.join(calib_dir,'multi_camera_global_poses.json'),'w') as f:
    json.dump(pose_json,f,indent=2)

# print results
print("\n📉 Selected MST edges:")
for A,B,d in mst.edges(data=True):
    print(f"🔗 {A} ↔ {B} — RMS: {d['weight']:.4f} px")
print("\n📋 Global Camera Poses:")
for cam,(R,T) in global_RTs.items():
    print(f"\n🟢 {cam}\nR =\n{np.array2string(R, precision=4)}\nT =\n{np.array2string(T.T, precision=4)}")
