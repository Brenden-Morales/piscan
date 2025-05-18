#!/usr/bin/env python3
"""
Multi-view Camera-only Bundle Adjustment
Using SciPy least_squares with Jacobian sparsity and Soft-L1 robust loss.
"""
import os
import glob
import json
import logging

import numpy as np
import cv2
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix

# ----------------------------------------------------------------------------
# Setup logging
# ----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("BA-SciPy")

# ----------------------------------------------------------------------------
# 1) Charuco board setup
# ----------------------------------------------------------------------------
paper_w_in = paper_h_in = 24.0
safe_margin_mm = 20.0
mm_per_inch = 25.4
page_w_mm = paper_w_in * mm_per_inch - 2 * safe_margin_mm
squaresX = squaresY = 16
square_size_m = (page_w_mm / squaresX) / 1000.0
marker_size_m = square_size_m * 0.75
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
board = cv2.aruco.CharucoBoard(
    (squaresX, squaresY), squareLength=square_size_m,
    markerLength=marker_size_m, dictionary=aruco_dict
)
chessboard_3D = board.getChessboardCorners()
logger.info("Charuco board: %dx%d squares, square=%.3fm", squaresX, squaresY, square_size_m)

# ----------------------------------------------------------------------------
# 2) I/O and settings
# ----------------------------------------------------------------------------
camera_names = [f"picam{i}.local" for i in range(6)]
intrinsics_dir = "calibration_results"
captures_dir   = "captures"
min_corners    = 50
sync_frame     = 0
os.makedirs(intrinsics_dir, exist_ok=True)

# ----------------------------------------------------------------------------
# 3) Load intrinsics
# ----------------------------------------------------------------------------
K = {}
dist = {}
for cam in camera_names:
    fn = os.path.join(intrinsics_dir, f"{cam}_intrinsics_refined_enhanced.npz")
    d = np.load(fn)
    K[cam] = d['K'].astype(np.float64)
    dist[cam] = d['dist'].astype(np.float64)
logger.info("Loaded intrinsics for %d cameras", len(camera_names))

# ----------------------------------------------------------------------------
# 4) Load initial poses from stereo
# ----------------------------------------------------------------------------
with open(os.path.join(intrinsics_dir, 'multi_camera_global_poses.json')) as f:
    global_RTs = json.load(f)
cam_r0 = []
cam_t0 = []
for cam in camera_names:
    R = np.array(global_RTs[cam]['R'], float)
    T = np.array(global_RTs[cam]['T'], float)
    rvec, _ = cv2.Rodrigues(R)
    cam_r0.append(rvec.flatten())
    cam_t0.append(T.flatten())
logger.info("Loaded initial camera poses")

# ----------------------------------------------------------------------------
# 5) Detect & cache observations
# ----------------------------------------------------------------------------
obs = []          # (cam_idx, frame_idx, pid, [u,v])
board_init = {}   # frame_idx -> (rvec, tvec)
img_lists = {cam: sorted(glob.glob(os.path.join(captures_dir, cam, '*.jpg'))) for cam in camera_names}
num_frames = len(img_lists[camera_names[0]])
for f in range(num_frames):
    img0 = cv2.imread(img_lists[camera_names[0]][f])
    gray0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    mc0, ids0, _ = cv2.aruco.detectMarkers(gray0, aruco_dict)
    _, cc0, cids0 = cv2.aruco.interpolateCornersCharuco(mc0, ids0, gray0, board)
    if cids0 is None or len(cids0) < min_corners:
        continue
    und0 = cv2.undistortPoints(cc0, K[camera_names[0]], dist[camera_names[0]], P=K[camera_names[0]]).reshape(-1,2)
    obj0 = np.array([chessboard_3D[int(cid)] for cid in cids0.flatten()])
    _, r0, t0 = cv2.solvePnP(obj0, und0, K[camera_names[0]], None)
    board_init[f] = (r0.flatten(), t0.flatten())
    for ci, cam in enumerate(camera_names):
        img = cv2.imread(img_lists[cam][f])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mc, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict)
        _, cc, cids = cv2.aruco.interpolateCornersCharuco(mc, ids, gray, board)
        if cids is None or len(cids) < min_corners:
            continue
        und = cv2.undistortPoints(cc, K[cam], dist[cam], P=K[cam]).reshape(-1,2)
        for i, pid in enumerate(cids.flatten()):
            obs.append((ci, f, int(pid), und[i]))
logger.info("Collected %d observations over %d frames", len(obs), len(board_init))

# ----------------------------------------------------------------------------
# 6) Build parameter vector & sparsity
# ----------------------------------------------------------------------------
ncams = len(camera_names)
frames = sorted([f for f in board_init if f != sync_frame])
x0 = np.zeros(ncams*6 + len(frames)*6)
for i in range(ncams):
    x0[6*i:6*i+3] = cam_r0[i]
    x0[6*i+3:6*i+6] = cam_t0[i]
for j, f in enumerate(frames):
    rj, tj = board_init[f]
    x0[ncams*6+6*j:ncams*6+6*j+3] = rj
    x0[ncams*6+6*j+3:ncams*6+6*j+6] = tj
m = len(obs)*2
n = x0.size
rows, cols = [], []
for i, (ci, f, pid, uv) in enumerate(obs):
    row_inds = [2*i, 2*i+1]
    for k in range(6):
        rows.extend(row_inds)
        cols.extend([ci*6+k, ci*6+k])
    if f != sync_frame:
        j = frames.index(f)
        base = ncams*6 + 6*j
        for k in range(6):
            rows.extend(row_inds)
            cols.extend([base+k, base+k])
data = np.ones(len(rows))
jac_sparsity = coo_matrix((data, (rows, cols)), shape=(m, n))
logger.info("Jacobian sparsity: %d entries", len(rows))

# ----------------------------------------------------------------------------
# 7) Residual function with robust handling for z<=0
# ----------------------------------------------------------------------------
def residuals(x):
    cam_params = x[:ncams*6].reshape(ncams,6)
    res = np.zeros(m)
    for i, (ci, f, pid, uv) in enumerate(obs):
        rvec, tvec = cam_params[ci,:3], cam_params[ci,3:6]
        Rcam, _ = cv2.Rodrigues(rvec)
        tc = tvec.reshape(3,1)
        if f == sync_frame:
            Rb, tb = np.eye(3), np.zeros((3,1))
        else:
            idx = frames.index(f)
            b = x[ncams*6+6*idx:ncams*6+6*idx+6]
            Rb, _ = cv2.Rodrigues(b[:3])
            tb = b[3:6].reshape(3,1)
        Xw = chessboard_3D[pid].reshape(3,1)
        Xc = Rcam.dot(Rb.dot(Xw)+tb) + tc
        z = Xc[2,0]
        if z <= 1e-6:
            # assign large residuals to keep array shape
            res[2*i] = 1e3
            res[2*i+1] = 1e3
        else:
            u = (Xc[0,0]/z)*K[camera_names[ci]][0,0] + K[camera_names[ci]][0,2]
            v = (Xc[1,0]/z)*K[camera_names[ci]][1,1] + K[camera_names[ci]][1,2]
            res[2*i] = u - uv[0]
            res[2*i+1] = v - uv[1]
    return res

# ----------------------------------------------------------------------------
# 8) Solve BA
# ----------------------------------------------------------------------------
logger.info("Running bundle adjustment...")
sol = least_squares(
    residuals, x0,
    jac_sparsity=jac_sparsity,
    method='trf', loss='soft_l1', f_scale=1.0,
    verbose=2
)
logger.info("BA done: cost=%.6f", sol.cost)

# ----------------------------------------------------------------------------
# 9) Save optimized poses
# ----------------------------------------------------------------------------
camera_poses = {}
for i, cam in enumerate(camera_names):
    rvec = sol.x[6*i:6*i+3]
    tvec = sol.x[6*i+3:6*i+6]
    Ropt, _ = cv2.Rodrigues(rvec)
    camera_poses[cam] = {'R': Ropt.tolist(), 'T': tvec.tolist()}
os.makedirs('results', exist_ok=True)
with open('results/camera_poses_ba.json', 'w') as f:
    json.dump(camera_poses, f, indent=2)
logger.info("Saved optimized poses to results/camera_poses_ba.json")
