# -*- coding: utf-8 -*-
"""
Stereo Calibration with Pruning to min_valid_frames and View Saving
"""
import cv2
import cv2.aruco as aruco
import numpy as np
import os, glob, json, shutil
from itertools import product, combinations
import networkx as nx

# === Board setup ===
squaresX = squaresY = 16
paper_w_in = paper_h_in = 24.0
safe_margin = 20.0
mm_per_inch = 25.4
page_w_mm = paper_w_in * mm_per_inch - 2 * safe_margin
square_size_m = (page_w_mm / squaresX) / 1000.0
marker_size_m = square_size_m * 0.75
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)
board = aruco.CharucoBoard((squaresX, squaresY), squareLength=square_size_m, markerLength=marker_size_m, dictionary=aruco_dict)
chessboard_3D = board.getChessboardCorners()

# === I/O and parameters ===
camera_names = [f"picam{i}.local" for i in range(6)]
calib_dir = "calibration_results"
captures_dir = "captures"
image_ext = "*.jpg"
min_common_corners = 80
min_valid_frames = 25
max_rms_error = 1.5
min_baseline_m = 0.01
max_baseline_m = 0.8
os.makedirs(calib_dir, exist_ok=True)

print("\n📅 Loading camera intrinsics…")
intrinsics = {}
for cam in camera_names:
    fn = os.path.join(calib_dir, f"{cam}_intrinsics_refined.npz")
    data = np.load(fn)
    intrinsics[cam] = {
        "K": data["K"].astype(np.float64),
        "dist": data["dist"].astype(np.float64),
        "image_size": tuple(data["image_size"])
    }
print("✅ Intrinsics loaded for all cameras!\n")

def detect_charuco_pairs(imgsA, imgsB, subpix_criteria):
    obj_pts, img_ptsA, img_ptsB = [], [], []
    keep_filenames = []
    for idx, (fA, fB) in enumerate(zip(imgsA, imgsB)):
        imgA = cv2.imread(fA); grayA = cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY)
        imgB = cv2.imread(fB); grayB = cv2.cvtColor(imgB, cv2.COLOR_BGR2GRAY)
        mA, idA, _ = aruco.detectMarkers(grayA, aruco_dict)
        mB, idB, _ = aruco.detectMarkers(grayB, aruco_dict)
        if idA is None or idB is None:
            continue
        _, ccA, cidA = aruco.interpolateCornersCharuco(mA, idA, grayA, board)
        _, ccB, cidB = aruco.interpolateCornersCharuco(mB, idB, grayB, board)
        if cidA is None or cidB is None:
            continue
        common = np.intersect1d(cidA.flatten(), cidB.flatten())
        if len(common) < min_common_corners:
            continue
        ptsA = ccA.reshape(-1, 1, 2); ptsB = ccB.reshape(-1, 1, 2)
        cv2.cornerSubPix(grayA, ptsA, (7, 7), (-1, -1), subpix_criteria)
        cv2.cornerSubPix(grayB, ptsB, (7, 7), (-1, -1), subpix_criteria)
        pts3d, xA, xB = [], [], []
        for cid in common:
            iA = np.where(cidA == cid)[0][0]
            iB = np.where(cidB == cid)[0][0]
            pts3d.append(chessboard_3D[int(cid)])
            xA.append(ptsA[iA, 0])
            xB.append(ptsB[iB, 0])
        obj_pts.append(np.array(pts3d, dtype=np.float32))
        img_ptsA.append(np.array(xA, dtype=np.float32))
        img_ptsB.append(np.array(xB, dtype=np.float32))
        keep_filenames.append((fA, fB))
    print(f"✅ Detected {len(obj_pts)} valid synchronized frames.")
    return obj_pts, img_ptsA, img_ptsB, keep_filenames

def prune_views(obj_pts, img_ptsA, img_ptsB, filenames, K0, dist0, K1, dist1, R, T, min_frames, outdir):
    print("🔧 Computing per-view reprojection errors…")
    per_view_err = []
    for i, (obj3d, pA, pB) in enumerate(zip(obj_pts, img_ptsA, img_ptsB)):
        _, rA, tA = cv2.solvePnP(obj3d, pA, K0, dist0)
        R_A, _ = cv2.Rodrigues(rA)
        projA, _ = cv2.projectPoints(obj3d, rA, tA, K0, dist0)
        errA = np.linalg.norm(projA.reshape(-1, 2) - pA, axis=1).mean()
        R_B = R @ R_A
        tB = R @ tA + T
        rB, _ = cv2.Rodrigues(R_B)
        projB, _ = cv2.projectPoints(obj3d, rB, tB, K1, dist1)
        errB = np.linalg.norm(projB.reshape(-1, 2) - pB, axis=1).mean()
        per_view_err.append((i, (errA + errB) / 2))

    per_view_err.sort(key=lambda x: x[1], reverse=True)
    num_to_prune = max(0, len(obj_pts) - min_frames)
    drop_idx = {i for i, _ in per_view_err[:num_to_prune]}

    kept = [filenames[i] for i in range(len(filenames)) if i not in drop_idx]
    dropped = [filenames[i] for i in range(len(filenames)) if i in drop_idx]
    kept_dir = os.path.join(outdir, "kept")
    drop_dir = os.path.join(outdir, "dropped")
    os.makedirs(kept_dir, exist_ok=True)
    os.makedirs(drop_dir, exist_ok=True)

    for i, group in enumerate([kept, dropped]):
        dest = kept_dir if i == 0 else drop_dir
        for fA, fB in group:
            shutil.copy(fA, os.path.join(dest, os.path.basename(fA)))
            shutil.copy(fB, os.path.join(dest, os.path.basename(fB)))

    obj_pts_p = [o for i, o in enumerate(obj_pts) if i not in drop_idx]
    imgA_p = [x for i, x in enumerate(img_ptsA) if i not in drop_idx]
    imgB_p = [x for i, x in enumerate(img_ptsB) if i not in drop_idx]
    print(f"✅ {len(obj_pts_p)} views remaining after pruning.")
    return obj_pts_p, imgA_p, imgB_p

# === Calibration loop ===
print("🔍 Evaluating all camera pairs…")
pair_results = []
param_grid = [(1e-6, 100, cv2.CALIB_FIX_INTRINSIC)]

for camA, camB in combinations(camera_names, 2):
    print(f"\n🔁 Trying: {camA} ↔ {camB}")
    K0, dist0, img_size0 = intrinsics[camA].values()
    K1, dist1, img_size1 = intrinsics[camB].values()
    if img_size0 != img_size1:
        print("⚠️ Skipping — image size mismatch"); continue
    imgsA = sorted(glob.glob(os.path.join(captures_dir, camA, image_ext)))
    imgsB = sorted(glob.glob(os.path.join(captures_dir, camB, image_ext)))
    if len(imgsA) != len(imgsB):
        print("⚠️ Skipping — frame count mismatch"); continue

    best_rms = float('inf'); best_result = None
    for eps, max_iter, flags_final in param_grid:
        subpix_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, max_iter, eps)
        obj_pts, img_ptsA, img_ptsB, filenames = detect_charuco_pairs(imgsA, imgsB, subpix_criteria)
        if len(obj_pts) < min_valid_frames:
            print(f"🚫 Skipping — only {len(obj_pts)} valid frames found")
            continue
        retval1, _, _, _, _, R1, T1, _, _ = cv2.stereoCalibrate(
            obj_pts, img_ptsA, img_ptsB, K0, dist0, K1, dist1, img_size0,
            flags=cv2.CALIB_FIX_INTRINSIC,
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
        )
        print(f"🔁 Initial RMS: {retval1:.4f}")
        outdir = os.path.join("pruned_views", f"{camA}_{camB}")
        obj_pts_p, imgA_p, imgB_p = prune_views(obj_pts, img_ptsA, img_ptsB, filenames, K0, dist0, K1, dist1, R1, T1, min_valid_frames, outdir)
        if len(obj_pts_p) < min_valid_frames:
            print("🚫 Skipping — too few frames after pruning")
            continue
        retval2, _, _, _, _, R2, T2, _, _ = cv2.stereoCalibrate(
            obj_pts_p, imgA_p, imgB_p, K0, dist0, K1, dist1, img_size0,
            flags=flags_final,
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
        )
        baseline = np.linalg.norm(T2)
        print(f"📐 Final RMS: {retval2:.4f} px, Baseline: {baseline*100:.1f} cm\n")
        if retval2 >= max_rms_error:
            print("🚫 Rejected — RMS too high")
        elif not (min_baseline_m < baseline < max_baseline_m):
            print("🚫 Rejected — baseline out of bounds")
        elif retval2 < best_rms:
            best_rms = retval2
            best_result = (retval2, camA, camB, R2, T2, eps, max_iter, flags_final)

    if best_result:
        err, A, B, R, T, eps, iters, flags = best_result
        np.savez(os.path.join(calib_dir, f"stereo_best_{A}_{B}.npz"), R=R, T=T, error=err)
        print(f"✅ Best RMS: {err:.4f} px — Eps={eps}, Iter={iters}, Flags={flags}")
        pair_results.append((err, A, B, R, T))
    else:
        print("❌ No valid result for this pair.")

# === Pose Graph Assembly ===
print("\n🧠 Building pose graph with lowest-error links…")
G = nx.Graph()
for err, A, B, R, T in pair_results:
    G.add_edge(A, B, weight=err, R=R, T=T)
mst = nx.minimum_spanning_tree(G)

global_RTs = {camera_names[0]: (np.eye(3), np.zeros((3, 1)))}
def dfs(cam, visited):
    visited.add(cam)
    for nbr in mst.neighbors(cam):
        if nbr in visited: continue
        edge = mst[cam][nbr]
        R0, T0 = global_RTs[cam]
        Rn = edge['R'] @ R0
        Tn = edge['R'] @ T0 + edge['T']
        global_RTs[nbr] = (Rn, Tn)
        dfs(nbr, visited)
dfs(camera_names[0], set())

pose_json = {cam: {'R': R.tolist(), 'T': T.flatten().tolist()} for cam, (R, T) in global_RTs.items()}
with open(os.path.join(calib_dir, 'multi_camera_global_poses.json'), 'w') as f:
    json.dump(pose_json, f, indent=2)

print("\n📉 Selected MST edges:")
for A, B, d in mst.edges(data=True):
    print(f"🔗 {A} ↔ {B} — RMS: {d['weight']:.4f} px")

print("\n📋 Global Camera Poses:")
for cam, (R, T) in global_RTs.items():
    print(f"\n🟢 {cam}\nR =\n{np.array2string(R, precision=4)}\nT =\n{np.array2string(T.T, precision=4)}")
