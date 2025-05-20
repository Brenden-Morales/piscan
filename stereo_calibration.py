# -*- coding: utf-8 -*-
"""
Stereo Calibration with Sub-pixel Refinement, Extrinsic Guess & Iterative Prune
"""
import cv2
import cv2.aruco as aruco
import numpy as np
import os, glob, json
from itertools import combinations
import networkx as nx

# -----------------------------------------------------------------------------
# 1) Charuco board definition (16×16, DICT_4X4_1000)
# -----------------------------------------------------------------------------
paper_w_in   = paper_h_in = 24.0
safe_margin  = 20.0  # mm
mm_per_inch  = 25.4
page_w_mm    = paper_w_in * mm_per_inch - 2*safe_margin

squaresX = squaresY = 16
square_size_m = (page_w_mm / squaresX) / 1000.0
marker_size_m = square_size_m * 0.75

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)
board      = aruco.CharucoBoard(
    (squaresX, squaresY),
    squareLength=square_size_m,
    markerLength=marker_size_m,
    dictionary=aruco_dict
)
chessboard_3D = board.getChessboardCorners()

# -----------------------------------------------------------------------------
# 2) I/O & thresholds
# -----------------------------------------------------------------------------
camera_names      = [f"picam{i}.local" for i in range(6)]
calib_dir         = "calibration_results"
captures_dir      = "captures"
image_ext         = "*.jpg"

min_common_corners = 80    # require ≥40 shared corners
min_valid_frames   = 25    # require ≥20 frames to calibrate
max_rms_error      = 1.5   # px
min_baseline_m     = 0.01  # 1 cm
max_baseline_m     = 0.8   # 80 cm
prune_fraction     = 0.1   # drop worst 20% of frames

os.makedirs(calib_dir, exist_ok=True)

# -----------------------------------------------------------------------------
# 3) Sub-pixel refinement settings
# -----------------------------------------------------------------------------
subpix_win      = (7, 7)
subpix_zero     = (-1, -1)
subpix_criteria = (
    cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
    100,
    1e-5
)

# -----------------------------------------------------------------------------
# 4) Load intrinsics
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# 5) Stereo-calibration per pair
# -----------------------------------------------------------------------------
pair_results = []
print("🔍 Evaluating all camera pairs…")
for camA, camB in combinations(camera_names, 2):
    print(f"\n🔁 Trying: {camA} ↔ {camB}")
    K0, dist0, img_size0 = intrinsics[camA]["K"], intrinsics[camA]["dist"], intrinsics[camA]["image_size"]
    K1, dist1, img_size1 = intrinsics[camB]["K"], intrinsics[camB]["dist"], intrinsics[camB]["image_size"]

    # must match resolution
    if img_size0 != img_size1:
        print("⚠️ Skipping — image size mismatch")
        continue

    # gather synchronized image lists
    imgsA = sorted(glob.glob(os.path.join(captures_dir, camA, image_ext)))
    imgsB = sorted(glob.glob(os.path.join(captures_dir, camB, image_ext)))
    if len(imgsA) != len(imgsB):
        print("⚠️ Skipping — frame count mismatch")
        continue

    # detect & refine Charuco corners in each pair
    obj_pts   = []
    img_ptsA  = []
    img_ptsB  = []
    for fA, fB in zip(imgsA, imgsB):
        imgA = cv2.imread(fA); grayA = cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY)
        imgB = cv2.imread(fB); grayB = cv2.cvtColor(imgB, cv2.COLOR_BGR2GRAY)

        # detect markers + Charuco interpolation
        mA, idA, _ = aruco.detectMarkers(grayA, aruco_dict)
        mB, idB, _ = aruco.detectMarkers(grayB, aruco_dict)
        if idA is None or idB is None:
            continue
        _, ccA, cidA = aruco.interpolateCornersCharuco(mA, idA, grayA, board)
        _, ccB, cidB = aruco.interpolateCornersCharuco(mB, idB, grayB, board)
        if cidA is None or cidB is None:
            continue

        # enforce minimum shared corners
        common = np.intersect1d(cidA.flatten(), cidB.flatten())
        if len(common) < min_common_corners:
            continue

        # sub-pixel refine both views
        ptsA = ccA.reshape(-1,1,2)
        ptsB = ccB.reshape(-1,1,2)
        cv2.cornerSubPix(grayA, ptsA, subpix_win, subpix_zero, subpix_criteria)
        cv2.cornerSubPix(grayB, ptsB, subpix_win, subpix_zero, subpix_criteria)

        # build object-and-image-point lists
        pts3d, xA, xB = [], [], []
        for cid in common:
            iA = np.where(cidA==cid)[0][0]
            iB = np.where(cidB==cid)[0][0]
            pts3d.append(chessboard_3D[int(cid)])
            xA.append(ptsA[iA,0])
            xB.append(ptsB[iB,0])

        obj_pts.append(np.array(pts3d, dtype=np.float32))
        img_ptsA.append(np.array(xA, dtype=np.float32))
        img_ptsB.append(np.array(xB, dtype=np.float32))

    if len(obj_pts) < min_valid_frames:
        print(f"⚠️ Skipping — only {len(obj_pts)} valid frames")
        continue

    # --- 5.1) Initial stereoCalibrate (no guess) ---
    flags_init = cv2.CALIB_FIX_INTRINSIC
    crit      = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)

    retval1, _, _, _, _, R1, T1, _, _ = cv2.stereoCalibrate(
        objectPoints   = obj_pts,
        imagePoints1   = img_ptsA,
        imagePoints2   = img_ptsB,
        cameraMatrix1  = K0,
        distCoeffs1    = dist0,
        cameraMatrix2  = K1,
        distCoeffs2    = dist1,
        imageSize      = img_size0,
        flags          = flags_init,
        criteria       = crit
    )
    print(f"  ▶ Initial RMS: {retval1:.4f} px")

    # --- 5.2) Compute per-view stereo reprojection error & prune ---
    # --- compute per-frame reprojection errors ---
    per_view_err = []
    R1_mat = R1   # ← already the correct 3×3

    for i, (obj3d, pA, pB) in enumerate(zip(obj_pts, img_ptsA, img_ptsB)):
        # solvePnP on camA
        _, rA, tA = cv2.solvePnP(obj3d, pA, K0, dist0)
        R_A, _ = cv2.Rodrigues(rA)

        # reproject into camA
        projA, _ = cv2.projectPoints(obj3d, rA, tA, K0, dist0)
        errA = np.linalg.norm(projA.reshape(-1,2) - pA, axis=1).mean()

        # transform into camB using the 3×3 R1_mat and T1
        R_Bi = R1_mat @ R_A
        tBi  = R1_mat.dot(tA) + T1
        rBi, _ = cv2.Rodrigues(R_Bi)

        # reproject into camB
        projB, _ = cv2.projectPoints(obj3d, rBi, tBi, K1, dist1)
        errB = np.linalg.norm(projB.reshape(-1,2) - pB, axis=1).mean()

        per_view_err.append((i, (errA + errB) / 2))


    # drop worst-error fraction
    per_view_err.sort(key=lambda x: x[1], reverse=True)
    n_drop = int(len(per_view_err) * prune_fraction)
    drop_idx = {i for i,_ in per_view_err[:n_drop]}

    obj_pts_pruned  = [o for i,o in enumerate(obj_pts)  if i not in drop_idx]
    imgA_pruned     = [x for i,x in enumerate(img_ptsA) if i not in drop_idx]
    imgB_pruned     = [x for i,x in enumerate(img_ptsB) if i not in drop_idx]
    print(f"  ▶ Pruned {n_drop} worst views → {len(obj_pts_pruned)} remain")

    if len(obj_pts_pruned) < min_valid_frames:
        print("  ❌ Too few after prune, skipping final solve")
        continue

    # --- 5.3) Final stereoCalibrate with extrinsic guess ---
# --- 5.3) Final stereoCalibrate with initial R/T but no extrinsic-guess flag ---
    flags_final = (cv2.CALIB_FIX_INTRINSIC | cv2.CALIB_RATIONAL_MODEL)
    retval2, camM1b, dist1b, camM2b, dist2b, R2, T2, E2, F2 = cv2.stereoCalibrate(
        objectPoints   = obj_pts_pruned,
        imagePoints1   = imgA_pruned,
        imagePoints2   = imgB_pruned,
        cameraMatrix1  = K0,
        distCoeffs1    = dist0,
        cameraMatrix2  = K1,
        distCoeffs2    = dist1,
        imageSize      = img_size0,
        #R              = R1,     # initial R from pass 1
        #T              = T1,     # initial T from pass 1
        flags          = flags_final,
        criteria       = crit
    )
    baseline = np.linalg.norm(T2)
    print(f"  ✅ Final RMS: {retval2:.4f} px, baseline: {baseline*100:.1f} cm")

    # filter out bad pairs
    if retval2 > max_rms_error:
        print("⚠️ Rejected: RMS too high")
        continue
    if baseline < min_baseline_m:
        print("⚠️ Rejected: baseline too small")
        continue
    if baseline > max_baseline_m:
        print("⚠️ Rejected: baseline too large")
        continue

    # save the good pair
    out_fn = os.path.join(calib_dir, f"stereo_calibration_{camA}_{camB}.npz")
    np.savez(out_fn, R=R2, T=T2, error=retval2)
    pair_results.append((retval2, camA, camB, R2, T2))
    print(f"  📁 Saved: {out_fn}")

# -----------------------------------------------------------------------------
# 6) Build MST & propagate global poses
# -----------------------------------------------------------------------------
print("\n🧠 Building pose graph with lowest-error links…")
G = nx.Graph()
for err, A, B, R, T in pair_results:
    G.add_edge(A, B, weight=err, R=R, T=T)
mst = nx.minimum_spanning_tree(G)

# propagate from the first camera
global_RTs = {camera_names[0]:(np.eye(3), np.zeros((3,1)))}
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

# write out a JSON of global poses
pose_json = {
    cam:{'R':R.tolist(), 'T':T.flatten().tolist()}
    for cam,(R,T) in global_RTs.items()
}
with open(os.path.join(calib_dir,'multi_camera_global_poses.json'),'w') as f:
    json.dump(pose_json, f, indent=2)

print("\n📉 Selected MST edges:")
for A,B,d in mst.edges(data=True):
    print(f"🔗 {A} ↔ {B} — RMS: {d['weight']:.4f} px")

print("\n📋 Global Camera Poses:")
for cam,(R,T) in global_RTs.items():
    print(f"\n🟢 {cam}\nR =\n{np.array2string(R,precision=4)}\nT =\n{np.array2string(T.T,precision=4)}")
