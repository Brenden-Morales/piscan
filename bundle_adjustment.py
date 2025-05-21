#!/usr/bin/env python3
"""
Multi-view Camera-only Bundle Adjustment
Using pyceres manual CostFunction subclass and Soft-L1 robust loss.
"""
import os
import glob
import json
import logging

import numpy as np
import cv2
import pyceres

# ----------------------------------------------------------------------------
# Setup logging
# ----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("BA-Ceres")

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
    fn = os.path.join(intrinsics_dir, f"{cam}_intrinsics_refined.npz")
    d = np.load(fn)
    K[cam] = d['K'].astype(np.float64)
    dist[cam] = d['dist'].astype(np.float64)
logger.info("Loaded intrinsics for %d cameras", len(camera_names))

# ----------------------------------------------------------------------------
# 4) Load initial poses & precompute cam0 → world
# ----------------------------------------------------------------------------
with open(os.path.join(intrinsics_dir, 'multi_camera_global_poses.json')) as f:
    global_RTs = json.load(f)

cam_r0 = []
cam_t0 = []
for cam in camera_names:
    R = np.array(global_RTs[cam]['R'], float)
    T = np.array(global_RTs[cam]['T'], float).reshape(3,1)
    rvec, _ = cv2.Rodrigues(R)
    cam_r0.append(rvec.flatten())
    cam_t0.append(T.flatten())

R0, _  = cv2.Rodrigues(cam_r0[0].reshape(3,1))
t0     = cam_t0[0].reshape(3,1)
R0_inv = R0.T
t0_inv = -R0_inv @ t0
logger.info("Loaded initial camera poses and precomputed cam0→world")

# 5) Detect & convert board poses + cache observations
# ----------------------------------------------------------------------------
obs = []
board_init = {}
img_lists = {cam: sorted(glob.glob(os.path.join(captures_dir, cam, '*.jpg')))
             for cam in camera_names}
num_frames = len(img_lists[camera_names[0]])

for f in range(num_frames):
    best_score = -1
    best_rvec = best_tvec = None
    best_cam_idx = -1

    # Try each camera to find best solvePnPRansac for board pose
    for ci, cam in enumerate(camera_names):
        img = cv2.imread(img_lists[cam][f])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mc, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict)
        _, cc, cids = cv2.aruco.interpolateCornersCharuco(mc, ids, gray, board)
        if cids is None or len(cids) < min_corners:
            continue
        und = cv2.undistortPoints(cc, K[cam], dist[cam], P=K[cam]).reshape(-1,2)
        obj = np.array([chessboard_3D[int(cid)] for cid in cids.flatten()])

        success, rvec, tvec, inliers = cv2.solvePnPRansac(obj, und, K[cam], None,
                                                          flags=cv2.SOLVEPNP_ITERATIVE)
        if not success or inliers is None or len(inliers) < min_corners:
            continue

        if len(inliers) > best_score:
            best_score = len(inliers)
            best_rvec = rvec
            best_tvec = tvec
            best_cam_idx = ci
            best_inliers = inliers
            best_cids = cids
            best_und = und

    if best_score < min_corners:
        continue  # Skip frame if no camera has good enough view

    # Compute board pose in cam0's world coordinate system
    R_cam, _ = cv2.Rodrigues(cam_r0[best_cam_idx].reshape(3,1))
    t_cam = cam_t0[best_cam_idx].reshape(3,1)
    R_board, _ = cv2.Rodrigues(best_rvec)
    t_board = best_tvec.reshape(3,1)

    Rb_w = R_cam.T @ R_board
    tb_w = R_cam.T @ (t_board - t_cam)
    board_init[f] = (cv2.Rodrigues(Rb_w)[0].flatten(), tb_w.flatten())

    # Recompute filtered 2D obs for all cameras (including best_cam)
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

# ----------------------------------------------------------------------------
# 6) Filter by depth
# ----------------------------------------------------------------------------
filtered = []
for ci, f, pid, uv in obs:
    rvec_w, tvec_w = board_init[f]
    Rw, _ = cv2.Rodrigues(rvec_w)
    tw = tvec_w.reshape(3,1)
    Xw = Rw.dot(chessboard_3D[pid].reshape(3,1)) + tw

    Rci, _ = cv2.Rodrigues(cam_r0[ci].reshape(3,1))
    tci = cam_t0[ci].reshape(3,1)
    Xci = Rci.dot(Xw) + tci
    if Xci[2,0] > 1e-6:
        filtered.append((ci, f, pid, uv))
obs = filtered
logger.info("After depth filtering: %d observations", len(obs))

# ----------------------------------------------------------------------------
# 7) Param initialization
# ----------------------------------------------------------------------------
ncams = len(camera_names)
frames = sorted([f for f in board_init if f != sync_frame])
x0 = np.zeros(ncams*6 + len(frames)*6)

for i in range(ncams):
    x0[6*i:6*i+3] = cam_r0[i]
    x0[6*i+3:6*i+6] = cam_t0[i]
for j, f in enumerate(frames):
    rj, tj = board_init[f]
    base = ncams*6 + 6*j
    x0[base:base+3] = rj
    x0[base+3:base+6] = tj

cam_params = [np.concatenate([cam_r0[i], cam_t0[i]]).astype(np.float64) for i in range(ncams)]
board_params = {f: np.concatenate(board_init[f]).astype(np.float64) for f in frames}

# ----------------------------------------------------------------------------
# 8) Define residual classes manually
# ----------------------------------------------------------------------------
class ReprojectionResidualSync(pyceres.CostFunction):
    def __init__(self, ci, pid, uv, rvec_sync, tvec_sync):
        super().__init__()
        self.set_num_residuals(2)
        self.set_parameter_block_sizes([6])
        self.ci = ci
        self.pid = pid
        self.uv = uv
        self.rvec_sync = np.asarray(rvec_sync, dtype=np.float64)
        self.tvec_sync = np.asarray(tvec_sync, dtype=np.float64)

    def Evaluate(self, parameters, residuals, jacobians):
        cam = parameters[0]
        rci = cam[:3]
        tci = cam[3:6].reshape(3,1)
        Rci, _ = cv2.Rodrigues(rci)

        Rbw, _ = cv2.Rodrigues(self.rvec_sync)
        Xw = Rbw @ chessboard_3D[self.pid].reshape(3,1) + self.tvec_sync.reshape(3,1)
        Xci = Rci @ Xw + tci
        z = Xci[2,0]

        if not np.isfinite(z) or z <= 1e-6:
            print(f"[Sync] Skipping residual for ci={self.ci}, pid={self.pid}, z={z}")
            return False  # Signal failure, Ceres will skip this residual

        fx, fy = K[camera_names[self.ci]][0,0], K[camera_names[self.ci]][1,1]
        cx, cy = K[camera_names[self.ci]][0,2], K[camera_names[self.ci]][1,2]
        u = (Xci[0,0]/z)*fx + cx
        v = (Xci[1,0]/z)*fy + cy
        residuals[0] = float(u - self.uv[0])
        residuals[1] = float(v - self.uv[1])

        if jacobians is not None and jacobians[0] is not None:
            eps = 1e-4
            J = np.zeros((2,6), dtype=np.float64)
            base_res = np.array([residuals[0], residuals[1]], dtype=np.float64)
            cam_base = cam.astype(np.float64)
            for k in range(6):
                cam_eps = cam_base.copy()
                cam_eps[k] += eps
                rci_e = cam_eps[:3]
                tci_e = cam_eps[3:6].reshape(3,1)
                Rci_e, _ = cv2.Rodrigues(rci_e)
                Xci_e = Rci_e @ Xw + tci_e
                z_e = Xci_e[2,0]
                u_e = (Xci_e[0,0]/z_e)*fx + cx
                v_e = (Xci_e[1,0]/z_e)*fy + cy
                res_e = np.array([u_e - self.uv[0], v_e - self.uv[1]], dtype=np.float64)
                J[:, k] = (res_e - base_res) / eps
            jacobians[0][:] = J.flatten()

        return True

class ReprojectionResidual(pyceres.CostFunction):
    def __init__(self, ci, pid, uv):
        super().__init__()
        self.set_num_residuals(2)
        self.set_parameter_block_sizes([6, 6])
        self.ci = ci
        self.pid = pid
        self.uv = uv

    def Evaluate(self, parameters, residuals, jacobians):
        cam = parameters[0]
        board = parameters[1]
        rci = cam[:3]
        tci = cam[3:6].reshape(3,1)
        Rci, _ = cv2.Rodrigues(rci)

        rbw = board[:3]
        tbw = board[3:6].reshape(3,1)
        Rbw, _ = cv2.Rodrigues(rbw)
        Xw = Rbw @ chessboard_3D[self.pid].reshape(3,1) + tbw
        Xci = Rci @ Xw + tci
        z = Xci[2,0]

        if not np.isfinite(z) or z <= 1e-6:
            print(f"[Board] Skipping residual for ci={self.ci}, pid={self.pid}, z={z}")
            return False

        fx, fy = K[camera_names[self.ci]][0,0], K[camera_names[self.ci]][1,1]
        cx, cy = K[camera_names[self.ci]][0,2], K[camera_names[self.ci]][1,2]
        u = (Xci[0,0]/z)*fx + cx
        v = (Xci[1,0]/z)*fy + cy
        residuals[0] = float(u - self.uv[0])
        residuals[1] = float(v - self.uv[1])
        if jacobians is not None:
            eps = 1e-6
            base_res = np.array([residuals[0], residuals[1]], dtype=np.float64)
            if jacobians[0] is not None:
                Jc = np.zeros((2,6), dtype=np.float64)
                cam_base = cam.astype(np.float64)
                for k in range(6):
                    cam_eps = cam_base.copy()
                    cam_eps[k] += eps
                    rci_e = cam_eps[:3]
                    tci_e = cam_eps[3:6].reshape(3,1)
                    Rci_e, _ = cv2.Rodrigues(rci_e)
                    Xci_e = Rci_e @ Xw + tci_e
                    z_e = Xci_e[2,0]
                    u_e = (Xci_e[0,0]/z_e)*fx + cx
                    v_e = (Xci_e[1,0]/z_e)*fy + cy
                    res_e = np.array([u_e - self.uv[0], v_e - self.uv[1]], dtype=np.float64)
                    Jc[:, k] = (res_e - base_res) / eps
                jacobians[0][:] = Jc.flatten()
            if jacobians[1] is not None:
                Jb = np.zeros((2,6), dtype=np.float64)
                board_base = board.astype(np.float64)
                for k in range(6):
                    board_eps = board_base.copy()
                    board_eps[k] += eps
                    rbw_e = board_eps[:3]
                    tbw_e = board_eps[3:6].reshape(3,1)
                    Rbw_e, _ = cv2.Rodrigues(rbw_e)
                    Xw_e = Rbw_e @ chessboard_3D[self.pid].reshape(3,1) + tbw_e
                    Xci_e = Rci @ Xw_e + tci
                    z_e = Xci_e[2,0]
                    u_e = (Xci_e[0,0]/z_e)*fx + cx
                    v_e = (Xci_e[1,0]/z_e)*fy + cy
                    res_e = np.array([u_e - self.uv[0], v_e - self.uv[1]], dtype=np.float64)
                    Jb[:, k] = (res_e - base_res) / eps
                jacobians[1][:] = Jb.flatten()
        return True

# ----------------------------------------------------------------------------
# 9) Setup and solve problem
# ----------------------------------------------------------------------------
logger.info("Solving with Ceres")
problem = pyceres.Problem()
loss = pyceres.SoftLOneLoss(5.0)

for ci in range(ncams):
    problem.add_parameter_block(cam_params[ci], 6)
for f in frames:
    problem.add_parameter_block(board_params[f], 6)

for ci, f, pid, uv in obs:
    if f == sync_frame:
        r_sync, t_sync = board_init[sync_frame]
        test_cost = ReprojectionResidualSync(ci, pid, uv, r_sync, t_sync)
        if test_cost.Evaluate([cam_params[ci]], [0.0, 0.0], None):
            problem.add_residual_block(test_cost, loss, [cam_params[ci]])
        else:
            print(f"⚠️ Skipped invalid sync residual for ci={ci}, pid={pid}")
    else:
        test_cost = ReprojectionResidual(ci, pid, uv)
        if test_cost.Evaluate([cam_params[ci], board_params[f]], [0.0, 0.0], None):
            problem.add_residual_block(test_cost, loss, [cam_params[ci], board_params[f]])
        else:
            print(f"⚠️ Skipped invalid board residual for ci={ci}, pid={pid}, f={f}")


options = pyceres.SolverOptions()
options.max_num_iterations = 2000
options.minimizer_progress_to_stdout = True
summary = pyceres.SolverSummary()
pyceres.solve(options, problem, summary)
logger.info("BA done: final cost=%.6f", summary.final_cost)

# ----------------------------------------------------------------------------
# 10) Save optimized poses
# ----------------------------------------------------------------------------
camera_poses = {}
for i, cam in enumerate(camera_names):
    rvec = cam_params[i][:3]
    tvec = cam_params[i][3:6]
    Ropt, _ = cv2.Rodrigues(rvec)
    camera_poses[cam] = {'R': Ropt.tolist(), 'T': tvec.tolist()}

os.makedirs('results', exist_ok=True)
with open('calibration_results/camera_poses_ba.json', 'w') as f:
    json.dump(camera_poses, f, indent=2)
logger.info("Saved optimized poses to results/camera_poses_ba.json")

def compute_reprojection_errors():
    errors = []
    for ci, f, pid, uv in obs:
        cam = cam_params[ci]
        if f == sync_frame:
            r_sync, t_sync = board_init[sync_frame]
            residual = ReprojectionResidualSync(ci, pid, uv, r_sync, t_sync)
            if residual.Evaluate([cam], [0.0, 0.0], None):
                fx, fy = K[camera_names[ci]][0,0], K[camera_names[ci]][1,1]
                cx, cy = K[camera_names[ci]][0,2], K[camera_names[ci]][1,2]
                rci = cam[:3]
                tci = cam[3:6].reshape(3,1)
                Rci, _ = cv2.Rodrigues(rci)
                Rbw, _ = cv2.Rodrigues(r_sync)
                Xw = Rbw @ chessboard_3D[pid].reshape(3,1) + t_sync.reshape(3,1)
                Xci = Rci @ Xw + tci
                z = Xci[2,0]
                u = (Xci[0,0]/z)*fx + cx
                v = (Xci[1,0]/z)*fy + cy
                err = np.linalg.norm([u - uv[0], v - uv[1]])
                errors.append((err, ci, f, pid))
        else:
            board = board_params[f]
            residual = ReprojectionResidual(ci, pid, uv)
            if residual.Evaluate([cam, board], [0.0, 0.0], None):
                fx, fy = K[camera_names[ci]][0,0], K[camera_names[ci]][1,1]
                cx, cy = K[camera_names[ci]][0,2], K[camera_names[ci]][1,2]
                rci = cam[:3]
                tci = cam[3:6].reshape(3,1)
                Rci, _ = cv2.Rodrigues(rci)
                rbw = board[:3]
                tbw = board[3:6].reshape(3,1)
                Rbw, _ = cv2.Rodrigues(rbw)
                Xw = Rbw @ chessboard_3D[pid].reshape(3,1) + tbw
                Xci = Rci @ Xw + tci
                z = Xci[2,0]
                u = (Xci[0,0]/z)*fx + cx
                v = (Xci[1,0]/z)*fy + cy
                err = np.linalg.norm([u - uv[0], v - uv[1]])
                errors.append((err, ci, f, pid))

    return sorted(errors, reverse=True)

# Run and report top offenders
top_errors = compute_reprojection_errors()[:10]
for err, ci, f, pid in top_errors:
    print(f"⚠️  Error={err:.2f} px | Camera={camera_names[ci]} | Frame={f} | ID={pid}")