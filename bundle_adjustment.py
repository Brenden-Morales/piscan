#!/usr/bin/env python3
"""
Multi-view Camera-only Bundle Adjustment
- Uses synchronized multi-camera, multi-frame Charuco observations
- Robust Soft-L1 loss to reject outliers
- Iterative outlier removal (drop top 5%)
- Vectorized reprojection (no iterative cv2 calls)
- Optional stereoRectify visual check for cam0→cam1
"""
import cv2
import cv2.aruco as aruco
import numpy as np
import os, glob, json
from scipy.optimize import least_squares

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
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)
board = aruco.CharucoBoard(
    (squaresX, squaresY), squareLength=square_size_m,
    markerLength=marker_size_m, dictionary=aruco_dict
)
chessboard_3D = board.getChessboardCorners()

# ----------------------------------------------------------------------------
# 2) I/O and settings
# ----------------------------------------------------------------------------
camera_names = [f"picam{i}.local" for i in range(6)]
intrinsics_dir = "calibration_results"
captures_dir   = "captures"
image_ext      = "*.jpg"
min_corners    = 50      # minimum Charuco corners per view
drop_percent   = 20      # drop top 20% residuals
sync_frame     = 0       # anchor world at frame index 0
os.makedirs(intrinsics_dir, exist_ok=True)

# ----------------------------------------------------------------------------
# 3) Load enhanced intrinsics
# ----------------------------------------------------------------------------
print("Loading enhanced intrinsics...")
K_dict, dist_dict = {}, {}
for cam in camera_names:
    fn = os.path.join(intrinsics_dir, f"{cam}_intrinsics_refined_enhanced.npz")
    d = np.load(fn)
    K_dict[cam]    = d["K"].astype(np.float64)
    dist_dict[cam] = d["dist"].astype(np.float64)
print("Done loading intrinsics.")

# ----------------------------------------------------------------------------
# 4) Load stereo-based global poses (initial guesses)
# ----------------------------------------------------------------------------
print("Loading stereo calibration global poses...")
global_RT_path = os.path.join(intrinsics_dir, 'multi_camera_global_poses.json')
with open(global_RT_path, 'r') as f:
    global_RTs = json.load(f)
# Convert each global R/T to Rodrigues + translation vectors
cam_r0, cam_t0 = {}, {}
for cam in camera_names:
    R = np.array(global_RTs[cam]['R'], dtype=np.float64)
    T = np.array(global_RTs[cam]['T'], dtype=np.float64).reshape(3)
    rvec, _ = cv2.Rodrigues(R)
    cam_r0[cam] = rvec.flatten()
    cam_t0[cam] = T.flatten()
print("Initial camera poses set from stereo calibration.")

# ----------------------------------------------------------------------------
# 5) Detect & cache all frame observations
# ----------------------------------------------------------------------------
print("Detecting Charuco in all frames...")
img_lists = {cam: sorted(glob.glob(os.path.join(captures_dir, cam, image_ext)))
             for cam in camera_names}
num_frames = len(img_lists[camera_names[0]])
obs = []             # list of dicts: cam, frame, pid, xy
all_pts3d = {}       # pid -> 3D point (board corner)
board_init = {}      # frame_idx -> (rvec, tvec) of board in cam0

for f in range(num_frames):
    # Must see board in cam0
    img0 = cv2.imread(img_lists[camera_names[0]][f])
    gray0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    mc0, ids0, _ = aruco.detectMarkers(gray0, aruco_dict)
    _, cc0, cids0 = aruco.interpolateCornersCharuco(mc0, ids0, gray0, board)
    if cids0 is None or len(cids0) < min_corners:
        continue

    # SolvePnP in cam0 for board pose
    und0 = cv2.undistortPoints(cc0, K_dict[camera_names[0]], dist_dict[camera_names[0]], P=K_dict[camera_names[0]])
    obj0 = np.array([chessboard_3D[int(cid)] for cid in cids0.flatten()], np.float32)
    img0_pts = und0.reshape(-1,2)
    _, r0, t0 = cv2.solvePnP(obj0, img0_pts, K_dict[camera_names[0]], None)
    board_init[f] = (r0.flatten(), t0.flatten())

    # Record observations in each camera
    for cam_idx, cam in enumerate(camera_names):
        img = cv2.imread(img_lists[cam][f])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mc, ids, _ = aruco.detectMarkers(gray, aruco_dict)
        _, cc, cids = aruco.interpolateCornersCharuco(mc, ids, gray, board)
        if cids is None or len(cids) < min_corners:
            continue

        und = cv2.undistortPoints(cc, K_dict[cam], dist_dict[cam], P=K_dict[cam])
        pts = und.reshape(-1,2)
        for i, cid in enumerate(cids.flatten()):
            pid = int(cid)
            all_pts3d[pid] = chessboard_3D[pid]
            obs.append({'cam': cam_idx, 'frame': f, 'pid': pid, 'xy': pts[i]})
print(f"Collected {len(obs)} observations over {len(board_init)} synced frames.")

# ----------------------------------------------------------------------------
# 6) Build initial parameter vector x0
# ----------------------------------------------------------------------------
ncams = len(camera_names)
frame_list = sorted(board_init.keys())
boards = [f for f in frame_list if f != sync_frame]

# Parameter vector: [cam0 (r,t), cam1 (r,t), ..., board1 (r,t), board2, ...]
x0 = np.zeros(ncams*6 + len(boards)*6)
# Insert camera initial r/t from stereo calibration
for i, cam in enumerate(camera_names):
    x0[i*6:i*6+3]   = cam_r0[cam]
    x0[i*6+3:i*6+6] = cam_t0[cam]

# Compute board initial poses relative to sync_frame
r_sync, t_sync = board_init[sync_frame]
R_sync, _ = cv2.Rodrigues(r_sync)
t_sync = t_sync.reshape(3,1)
for j, f in enumerate(boards):
    rv, tv = board_init[f]
    Rf, _ = cv2.Rodrigues(rv)
    Tf = tv.reshape(3,1)
    R_rel = R_sync.T @ Rf
    T_rel = R_sync.T @ (Tf - t_sync)
    rv_rel, _ = cv2.Rodrigues(R_rel)
    base = ncams*6 + j*6
    x0[base:base+3]   = rv_rel.flatten()
    x0[base+3:base+6] = T_rel.flatten()

# ----------------------------------------------------------------------------
# 7) Residual function and BA invocation
# ----------------------------------------------------------------------------
def residuals(x, obs, Kd, pts3d):
    nc = len(camera_names)
    cam_params = x[:nc*6].reshape(nc, 6)
    rv_cam = cam_params[:, :3]
    tv_cam = cam_params[:, 3:6]

    rv_b, tv_b = {}, {}
    idx = nc*6
    for j, f in enumerate(boards):
        rv_b[f] = x[idx + 6*j : idx + 6*j + 3]
        tv_b[f] = x[idx + 6*j + 3 : idx + 6*j + 6]

    res = []
    for o in obs:
        ci, fi, pid = o['cam'], o['frame'], o['pid']
        Rc, _ = cv2.Rodrigues(rv_cam[ci])
        tc = tv_cam[ci].reshape(3,1)

        if fi == sync_frame:
            Rb, Tb = np.eye(3), np.zeros((3,1))
        else:
            Rb, _ = cv2.Rodrigues(rv_b[fi])
            Tb = tv_b[fi].reshape(3,1)

        Xw = pts3d[pid].reshape(3,1)
        Xc = Rc @ (Rb @ Xw + Tb) + tc
        z = Xc[2,0]

        if z <= 1e-6:
            # give this “bad” point a big residual instead of dropping it
            # so the output vector stays the same length
            res.extend([1e3, 1e3])
        else:
            x_proj = Xc[:2,0] / z
            K = Kd[camera_names[ci]]
            u = K[0,0]*x_proj[0] + K[0,2]
            v = K[1,1]*x_proj[1] + K[1,2]
            res.extend([u - o['xy'][0], v - o['xy'][1]])

    return np.array(res)

print("Running multi-view BA using stereo-initialized poses...")
sol1 = least_squares(
    residuals, x0,
    args=(obs, K_dict, all_pts3d),
    method='trf', loss='soft_l1', f_scale=1.0,
    verbose=2
)
# take the optimized x from the first solve
x_opt = sol1
# ... iterative pruning and re-run as before ...
# final extraction and saving can remain unchanged


# ----------------------------------------------------------------------------
# 8) Iterative outlier removal
# ----------------------------------------------------------------------------
print("Pruning top-5% worst observations and re-running BA...")
r0 = sol1.fun.reshape(-1,2)
errs = np.hypot(r0[:,0], r0[:,1])
thr = np.percentile(errs, 100-drop_percent)
mask = errs <= thr
obs2 = [obs[i] for i in range(len(errs)) if mask[i]]
sol2 = least_squares(
    residuals, x_opt,
    args=(obs2, K_dict, all_pts3d),
    method='trf',
    loss='soft_l1',
    f_scale=1.0,
    verbose=2
)

# final optimized
x_fin = sol2.x
print("BA complete.")

# ----------------------------------------------------------------------------
# 9) Extract and save camera poses
# ----------------------------------------------------------------------------
print("Saving camera poses...")
camera_poses = {}
for i, cam in enumerate(camera_names):
    rvc = x_fin[i*6:i*6+3]
    tvc = x_fin[i*6+3:i*6+6]
    Rc,_ = cv2.Rodrigues(rvc)
    camera_poses[cam] = {'R': Rc.tolist(), 'T': tvc.tolist()}
with open('results/camera_poses_ba.json','w') as f:
    json.dump(camera_poses, f, indent=2)
print("Saved to results/camera_poses_ba.json")

# ----------------------------------------------------------------------------
# 10) (Optional) stereoRectify visual check for cam0 & cam1
# ----------------------------------------------------------------------------
cam0, cam1 = camera_names[0], camera_names[1]
R0 = cv2.Rodrigues(x_fin[0:3])[0]; t0 = x_fin[3:6].reshape(3,1)
R1 = cv2.Rodrigues(x_fin[6:9])[0]; t1 = x_fin[9:12].reshape(3,1)
img_size = (int(img0.shape[1]), int(img0.shape[0]))
R01 = R1 @ R0.T; T01 = t1 - R01 @ t0
R0r, R1r, P0r, P1r, Q, _, _ = cv2.stereoRectify(
    K_dict[cam0], dist_dict[cam0],
    K_dict[cam1], dist_dict[cam1],
    img_size, R01, T01, flags=cv2.CALIB_ZERO_DISPARITY
)
map0x, map0y = cv2.initUndistortRectifyMap(
    K_dict[cam0], dist_dict[cam0], R0r, P0r, img_size, cv2.CV_32FC1
)
map1x, map1y = cv2.initUndistortRectifyMap(
    K_dict[cam1], dist_dict[cam1], R1r, P1r, img_size, cv2.CV_32FC1
)
imgA = cv2.imread(img_lists[cam0][sync_frame])
imgB = cv2.imread(img_lists[cam1][sync_frame])
rectA = cv2.remap(imgA, map0x, map0y, cv2.INTER_LINEAR)
rectB = cv2.remap(imgB, map1x, map1y, cv2.INTER_LINEAR)
# draw scanlines
h = imgA.shape[0]
for y in np.linspace(0, h, 10, dtype=int):
    cv2.line(rectA, (0,y),(img_size[0],y),(0,255,0),1)
    cv2.line(rectB, (0,y),(img_size[0],y),(0,255,0),1)
cv2.imshow('Rectified Cam0', rectA)
cv2.imshow('Rectified Cam1', rectB)
cv2.waitKey(0)
cv2.destroyAllWindows()
