#!/usr/bin/env python3
"""Multi-view Camera-only Bundle Adjustment using pyceres."""
import glob
import json
import logging
import os
from typing import Dict, List, Tuple, Set

import cv2
import numpy as np
import pyceres

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("BA-Ceres")

# ---------------------------------------------------------------------------
# Constants and configuration
# ---------------------------------------------------------------------------
PAPER_IN = 24.0
SAFE_MARGIN_MM = 20.0
MM_PER_INCH = 25.4
SQUARES = 16
MARKER_SCALE = 0.75
CAMERA_NAMES = [f"picam{i}.local" for i in range(6)]
INTRINSICS_DIR = "calibration_results"
CAPTURES_DIR = "captures"
MIN_CORNERS = 50

# Target counts used during pruning
TARGET_FRAMES = 30
TARGET_RESIDUALS = 5000

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def create_charuco_board() -> Tuple[cv2.aruco_CharucoBoard, np.ndarray, cv2.aruco_Dictionary]:
    """Create the Charuco board and return board, 3D corners and dictionary."""
    page_w_mm = PAPER_IN * MM_PER_INCH - 2 * SAFE_MARGIN_MM
    square_size_m = (page_w_mm / SQUARES) / 1000.0
    marker_size_m = square_size_m * MARKER_SCALE
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
    board = cv2.aruco.CharucoBoard(
        (SQUARES, SQUARES),
        squareLength=square_size_m,
        markerLength=marker_size_m,
        dictionary=dictionary,
    )
    chessboard_3d = board.getChessboardCorners()
    logger.info("Charuco board: %dx%d squares, square=%.3fm", SQUARES, SQUARES, square_size_m)
    return board, chessboard_3d, dictionary


def load_intrinsics(names: List[str], directory: str) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Load camera intrinsics from disk."""
    K, dist = {}, {}
    for cam in names:
        data = np.load(os.path.join(directory, f"{cam}_intrinsics_refined.npz"))
        K[cam] = data["K"].astype(np.float64)
        dist[cam] = data["dist"].astype(np.float64)
    logger.info("Loaded intrinsics for %d cameras", len(names))
    return K, dist


def load_initial_poses(directory: str, names: List[str]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Load initial global poses and convert to Rodrigues form."""
    with open(os.path.join(directory, "multi_camera_global_poses.json")) as f:
        global_RTs = json.load(f)

    cam_rvecs, cam_tvecs = [], []
    for cam in names:
        R = np.asarray(global_RTs[cam]["R"], dtype=float)
        T = np.asarray(global_RTs[cam]["T"], dtype=float).reshape(3, 1)
        rvec, _ = cv2.Rodrigues(R)
        cam_rvecs.append(rvec.flatten())
        cam_tvecs.append(T.flatten())
    logger.info("Loaded initial camera poses")
    return cam_rvecs, cam_tvecs


def detect_board_poses(
    board: cv2.aruco_CharucoBoard,
    dictionary: cv2.aruco_Dictionary,
    chessboard_3d: np.ndarray,
    cam_rvecs: List[np.ndarray],
    cam_tvecs: List[np.ndarray],
    K: Dict[str, np.ndarray],
    dist: Dict[str, np.ndarray],
) -> Tuple[Dict[int, Tuple[np.ndarray, np.ndarray]], List[Tuple[int, int, int, np.ndarray]]]:
    """Detect board poses and cache observations for all frames."""
    obs = []
    board_init: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    img_lists = {
        cam: sorted(glob.glob(os.path.join(CAPTURES_DIR, cam, "*.jpg"))) for cam in CAMERA_NAMES
    }
    num_frames = len(img_lists[CAMERA_NAMES[0]])

    for f in range(num_frames):
        best_score = -1
        best_rvec = None
        best_tvec = None
        best_cam_idx = -1

        for ci, cam in enumerate(CAMERA_NAMES):
            img = cv2.imread(img_lists[cam][f])
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            markers, ids, _ = cv2.aruco.detectMarkers(gray, dictionary)
            _, cc, cids = cv2.aruco.interpolateCornersCharuco(markers, ids, gray, board)
            if cids is None or len(cids) < MIN_CORNERS:
                continue
            und = cv2.undistortPoints(cc, K[cam], dist[cam], P=K[cam]).reshape(-1, 2)
            obj = np.array([chessboard_3d[int(cid)] for cid in cids.flatten()])
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                obj,
                und,
                K[cam],
                None,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if not success or inliers is None or len(inliers) < MIN_CORNERS:
                continue
            if len(inliers) > best_score:
                best_score = len(inliers)
                best_rvec = rvec
                best_tvec = tvec
                best_cam_idx = ci
        if best_score < MIN_CORNERS:
            continue

        R_cam, _ = cv2.Rodrigues(cam_rvecs[best_cam_idx].reshape(3, 1))
        t_cam = cam_tvecs[best_cam_idx].reshape(3, 1)
        R_board, _ = cv2.Rodrigues(best_rvec)
        t_board = best_tvec.reshape(3, 1)

        Rb_w = R_cam.T @ R_board
        tb_w = R_cam.T @ (t_board - t_cam)
        board_init[f] = (cv2.Rodrigues(Rb_w)[0].flatten(), tb_w.flatten())

        for ci, cam in enumerate(CAMERA_NAMES):
            img = cv2.imread(img_lists[cam][f])
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            markers, ids, _ = cv2.aruco.detectMarkers(gray, dictionary)
            _, cc, cids = cv2.aruco.interpolateCornersCharuco(markers, ids, gray, board)
            if cids is None or len(cids) < MIN_CORNERS:
                continue
            und = cv2.undistortPoints(cc, K[cam], dist[cam], P=K[cam]).reshape(-1, 2)
            for i, pid in enumerate(cids.flatten()):
                obs.append((ci, f, int(pid), und[i]))
    return board_init, obs


def filter_by_depth(
    obs: List[Tuple[int, int, int, np.ndarray]],
    board_init: Dict[int, Tuple[np.ndarray, np.ndarray]],
    cam_rvecs: List[np.ndarray],
    cam_tvecs: List[np.ndarray],
    chessboard_3d: np.ndarray,
) -> List[Tuple[int, int, int, np.ndarray]]:
    """Remove observations that project behind the camera."""
    filtered = []
    for ci, f, pid, uv in obs:
        rvec_w, tvec_w = board_init[f]
        Rw, _ = cv2.Rodrigues(rvec_w)
        tw = tvec_w.reshape(3, 1)
        Xw = Rw.dot(chessboard_3d[pid].reshape(3, 1)) + tw

        Rci, _ = cv2.Rodrigues(cam_rvecs[ci].reshape(3, 1))
        tci = cam_tvecs[ci].reshape(3, 1)
        Xci = Rci.dot(Xw) + tci
        if Xci[2, 0] > 1e-6:
            filtered.append((ci, f, pid, uv))
    logger.info("After depth filtering: %d observations", len(filtered))
    return filtered


def select_sync_frame(board_init: Dict[int, Tuple[np.ndarray, np.ndarray]]) -> int:
    """Select the frame whose board pose is closest to the mean pose."""
    if not board_init:
        raise ValueError("No board poses detected to select sync frame from")

    rvecs = np.array([pose[0] for pose in board_init.values()], dtype=np.float64)
    tvecs = np.array([pose[1] for pose in board_init.values()], dtype=np.float64)

    mean_rvec = np.mean(rvecs, axis=0)
    mean_tvec = np.mean(tvecs, axis=0)

    best_frame = None
    best_score = np.inf
    for f, (rvec, tvec) in board_init.items():
        dr = np.linalg.norm(np.asarray(rvec) - mean_rvec)
        dt = np.linalg.norm(np.asarray(tvec) - mean_tvec)
        score = dr + dt
        if score < best_score:
            best_score = score
            best_frame = f

    logger.info("Selected frame %d as SYNC_FRAME (score %.3f)", best_frame, best_score)
    return int(best_frame)


def initialize_parameters(
    board_init: Dict[int, Tuple[np.ndarray, np.ndarray]],
    cam_rvecs: List[np.ndarray],
    cam_tvecs: List[np.ndarray],
    sync_frame: int,
) -> Tuple[List[int], List[np.ndarray], Dict[int, np.ndarray]]:
    """Create parameter arrays for cameras and boards."""
    ncams = len(cam_rvecs)
    frames = sorted([f for f in board_init if f != sync_frame])
    cam_params = [np.concatenate([cam_rvecs[i], cam_tvecs[i]]).astype(np.float64) for i in range(ncams)]
    board_params = {f: np.concatenate(board_init[f]).astype(np.float64) for f in frames}
    return frames, cam_params, board_params


class ReprojectionResidualSync(pyceres.CostFunction):
    """Residual when the board pose is fixed (synchronized frame)."""

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
        tci = cam[3:6].reshape(3, 1)
        Rci, _ = cv2.Rodrigues(rci)

        Rbw, _ = cv2.Rodrigues(self.rvec_sync)
        Xw = Rbw @ chessboard_3D[self.pid].reshape(3, 1) + self.tvec_sync.reshape(3, 1)
        Xci = Rci @ Xw + tci
        z = Xci[2, 0]

        if not np.isfinite(z) or z <= 1e-6:
            return False

        fx, fy = K[CAMERA_NAMES[self.ci]][0, 0], K[CAMERA_NAMES[self.ci]][1, 1]
        cx, cy = K[CAMERA_NAMES[self.ci]][0, 2], K[CAMERA_NAMES[self.ci]][1, 2]
        u = (Xci[0, 0] / z) * fx + cx
        v = (Xci[1, 0] / z) * fy + cy
        residuals[0] = float(u - self.uv[0])
        residuals[1] = float(v - self.uv[1])

        if jacobians is not None and jacobians[0] is not None:
            eps = 1e-4
            J = np.zeros((2, 6), dtype=np.float64)
            base_res = np.array([residuals[0], residuals[1]], dtype=np.float64)
            cam_base = cam.astype(np.float64)
            for k in range(6):
                cam_eps = cam_base.copy()
                cam_eps[k] += eps
                rci_e = cam_eps[:3]
                tci_e = cam_eps[3:6].reshape(3, 1)
                Rci_e, _ = cv2.Rodrigues(rci_e)
                Xci_e = Rci_e @ Xw + tci_e
                z_e = Xci_e[2, 0]
                u_e = (Xci_e[0, 0] / z_e) * fx + cx
                v_e = (Xci_e[1, 0] / z_e) * fy + cy
                res_e = np.array([u_e - self.uv[0], v_e - self.uv[1]], dtype=np.float64)
                J[:, k] = (res_e - base_res) / eps
            jacobians[0][:] = J.flatten()

        return True


class ReprojectionResidual(pyceres.CostFunction):
    """Residual with board parameters as variables."""

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
        tci = cam[3:6].reshape(3, 1)
        Rci, _ = cv2.Rodrigues(rci)

        rbw = board[:3]
        tbw = board[3:6].reshape(3, 1)
        Rbw, _ = cv2.Rodrigues(rbw)
        Xw = Rbw @ chessboard_3D[self.pid].reshape(3, 1) + tbw
        Xci = Rci @ Xw + tci
        z = Xci[2, 0]

        if not np.isfinite(z) or z <= 1e-6:
            return False

        fx, fy = K[CAMERA_NAMES[self.ci]][0, 0], K[CAMERA_NAMES[self.ci]][1, 1]
        cx, cy = K[CAMERA_NAMES[self.ci]][0, 2], K[CAMERA_NAMES[self.ci]][1, 2]
        u = (Xci[0, 0] / z) * fx + cx
        v = (Xci[1, 0] / z) * fy + cy
        residuals[0] = float(u - self.uv[0])
        residuals[1] = float(v - self.uv[1])
        if jacobians is not None:
            eps = 1e-6
            base_res = np.array([residuals[0], residuals[1]], dtype=np.float64)
            if jacobians[0] is not None:
                Jc = np.zeros((2, 6), dtype=np.float64)
                cam_base = cam.astype(np.float64)
                for k in range(6):
                    cam_eps = cam_base.copy()
                    cam_eps[k] += eps
                    rci_e = cam_eps[:3]
                    tci_e = cam_eps[3:6].reshape(3, 1)
                    Rci_e, _ = cv2.Rodrigues(rci_e)
                    Xci_e = Rci_e @ Xw + tci_e
                    z_e = Xci_e[2, 0]
                    u_e = (Xci_e[0, 0] / z_e) * fx + cx
                    v_e = (Xci_e[1, 0] / z_e) * fy + cy
                    res_e = np.array([u_e - self.uv[0], v_e - self.uv[1]], dtype=np.float64)
                    Jc[:, k] = (res_e - base_res) / eps
                jacobians[0][:] = Jc.flatten()
            if jacobians[1] is not None:
                Jb = np.zeros((2, 6), dtype=np.float64)
                board_base = board.astype(np.float64)
                for k in range(6):
                    board_eps = board_base.copy()
                    board_eps[k] += eps
                    rbw_e = board_eps[:3]
                    tbw_e = board_eps[3:6].reshape(3, 1)
                    Rbw_e, _ = cv2.Rodrigues(rbw_e)
                    Xw_e = Rbw_e @ chessboard_3D[self.pid].reshape(3, 1) + tbw_e
                    Xci_e = Rci @ Xw_e + tci
                    z_e = Xci_e[2, 0]
                    u_e = (Xci_e[0, 0] / z_e) * fx + cx
                    v_e = (Xci_e[1, 0] / z_e) * fy + cy
                    res_e = np.array([u_e - self.uv[0], v_e - self.uv[1]], dtype=np.float64)
                    Jb[:, k] = (res_e - base_res) / eps
                jacobians[1][:] = Jb.flatten()
        return True


def compute_reprojection_errors(
    obs: List[Tuple[int, int, int, np.ndarray]],
    cam_params: List[np.ndarray],
    board_params: Dict[int, np.ndarray],
    board_init: Dict[int, Tuple[np.ndarray, np.ndarray]],
    sync_frame: int,
) -> List[Tuple[Tuple[float, int, int, int], Tuple[int, int, int, np.ndarray]]]:
    """Compute reprojection error for each observation."""
    errors = []
    for ci, f, pid, uv in obs:
        cam = cam_params[ci]
        if f == sync_frame:
            r_sync, t_sync = board_init[sync_frame]
            residual = ReprojectionResidualSync(ci, pid, uv, r_sync, t_sync)
            if residual.Evaluate([cam], [0.0, 0.0], None):
                fx, fy = K[CAMERA_NAMES[ci]][0, 0], K[CAMERA_NAMES[ci]][1, 1]
                cx, cy = K[CAMERA_NAMES[ci]][0, 2], K[CAMERA_NAMES[ci]][1, 2]
                rci = cam[:3]
                tci = cam[3:6].reshape(3, 1)
                Rci, _ = cv2.Rodrigues(rci)
                Rbw, _ = cv2.Rodrigues(r_sync)
                Xw = Rbw @ chessboard_3D[pid].reshape(3, 1) + t_sync.reshape(3, 1)
                Xci = Rci @ Xw + tci
                z = Xci[2, 0]
                u = (Xci[0, 0] / z) * fx + cx
                v = (Xci[1, 0] / z) * fy + cy
                err = np.linalg.norm([u - uv[0], v - uv[1]])
                errors.append(((err, ci, f, pid), (ci, f, pid, uv)))
        else:
            board = board_params[f]
            residual = ReprojectionResidual(ci, pid, uv)
            if residual.Evaluate([cam, board], [0.0, 0.0], None):
                fx, fy = K[CAMERA_NAMES[ci]][0, 0], K[CAMERA_NAMES[ci]][1, 1]
                cx, cy = K[CAMERA_NAMES[ci]][0, 2], K[CAMERA_NAMES[ci]][1, 2]
                rci = cam[:3]
                tci = cam[3:6].reshape(3, 1)
                Rci, _ = cv2.Rodrigues(rci)
                rbw = board[:3]
                tbw = board[3:6].reshape(3, 1)
                Rbw, _ = cv2.Rodrigues(rbw)
                Xw = Rbw @ chessboard_3D[pid].reshape(3, 1) + tbw
                Xci = Rci @ Xw + tci
                z = Xci[2, 0]
                u = (Xci[0, 0] / z) * fx + cx
                v = (Xci[1, 0] / z) * fy + cy
                err = np.linalg.norm([u - uv[0], v - uv[1]])
                errors.append(((err, ci, f, pid), (ci, f, pid, uv)))
    return sorted(errors, key=lambda x: -x[0][0])


def solve_bundle_adjustment(
    cam_params: List[np.ndarray],
    board_params: Dict[int, np.ndarray],
    obs_kept: List[Tuple[int, int, int, np.ndarray]],
    board_init: Dict[int, Tuple[np.ndarray, np.ndarray]],
    frames: List[int],
    sync_frame: int,
) -> pyceres.SolverSummary:
    """Set up and solve the Ceres problem."""
    logger.info("Solving with Ceres")
    problem = pyceres.Problem()
    loss = pyceres.SoftLOneLoss(5.0)

    for ci in range(len(cam_params)):
        problem.add_parameter_block(cam_params[ci], 6)
    for f in frames:
        problem.add_parameter_block(board_params[f], 6)

    for ci, f, pid, uv in obs_kept:
        if f == sync_frame:
            r_sync, t_sync = board_init[sync_frame]
            cost = ReprojectionResidualSync(ci, pid, uv, r_sync, t_sync)
            if cost.Evaluate([cam_params[ci]], [0.0, 0.0], None):
                problem.add_residual_block(cost, loss, [cam_params[ci]])
        else:
            cost = ReprojectionResidual(ci, pid, uv)
            if cost.Evaluate([cam_params[ci], board_params[f]], [0.0, 0.0], None):
                problem.add_residual_block(cost, loss, [cam_params[ci], board_params[f]])

    options = pyceres.SolverOptions()
    options.max_num_iterations = 2000
    options.minimizer_progress_to_stdout = True
    summary = pyceres.SolverSummary()
    pyceres.solve(options, problem, summary)
    logger.info("BA done: final cost=%.6f", summary.final_cost)
    return summary


def save_camera_poses(cam_params: List[np.ndarray]) -> None:
    """Write optimized camera poses to disk."""
    camera_poses = {}
    for i, cam in enumerate(CAMERA_NAMES):
        rvec = cam_params[i][:3]
        tvec = cam_params[i][3:6]
        Ropt, _ = cv2.Rodrigues(rvec)
        camera_poses[cam] = {"R": Ropt.tolist(), "T": tvec.tolist()}

    os.makedirs("results", exist_ok=True)
    with open(os.path.join(INTRINSICS_DIR, "camera_poses_ba.json"), "w") as f:
        json.dump(camera_poses, f, indent=2)
    logger.info("Saved optimized poses to results/camera_poses_ba.json")


def prune_frames_to_target(
    residual_errors: List[Tuple[Tuple[float, int, int, int], Tuple[int, int, int, np.ndarray]]],
    board_params: Dict[int, np.ndarray],
    board_init: Dict[int, Tuple[np.ndarray, np.ndarray]],
    frames: List[int],
    sync_frame: int,
    target_frames: int,
) -> Tuple[
    List[Tuple[Tuple[float, int, int, int], Tuple[int, int, int, np.ndarray]]],
    Dict[int, np.ndarray],
    Dict[int, Tuple[np.ndarray, np.ndarray]],
    List[int],
    Set[int]
]:
    """Prune the worst frames by mean reprojection error until target_frames remain."""
    from collections import defaultdict
    import numpy as np

    frame_errors = defaultdict(list)
    for (err, ci, f, pid), _ in residual_errors:
        frame_errors[f].append(err)

    if sync_frame not in frame_errors:
        frame_errors[sync_frame] = []

    n_total = len(frame_errors)
    target_frames = max(1, target_frames)
    frames_to_remove = max(0, n_total - target_frames)

    if frames_to_remove <= 0:
        logger.info("Target frame count %d >= available frames %d - no pruning", target_frames, n_total)
        return residual_errors, board_params, board_init, frames, set()

    frame_mean_error = {f: np.mean(errs) for f, errs in frame_errors.items()}
    sorted_frames = sorted(
        [(f, e) for f, e in frame_mean_error.items() if f != sync_frame],
        key=lambda x: x[1],
        reverse=True,
    )

    bad_frames = {f for f, _ in sorted_frames[:frames_to_remove]}

    residual_errors_pruned = [
        entry for entry in residual_errors if entry[1][1] not in bad_frames
    ]
    board_params_pruned = {f: p for f, p in board_params.items() if f not in bad_frames}
    board_init_pruned = {f: p for f, p in board_init.items() if f not in bad_frames}
    frames_pruned = [f for f in frames if f not in bad_frames]

    good_frames = sorted(f for f in frame_mean_error if f not in bad_frames)

    logger.info(
        "Pruning to %d frames: removed %d frames with highest reprojection error",
        target_frames,
        len(bad_frames),
    )
    logger.info("✔️  Good frames kept (%d): %s", len(good_frames), good_frames)
    logger.info("❌  Bad frames pruned (%d): %s", len(bad_frames), sorted(bad_frames))
    logger.info("Kept %d residuals after frame pruning", len(residual_errors_pruned))

    return residual_errors_pruned, board_params_pruned, board_init_pruned, frames_pruned, bad_frames



def main() -> None:
    global K, chessboard_3D

    os.makedirs(INTRINSICS_DIR, exist_ok=True)

    board, chessboard_3D, dictionary = create_charuco_board()
    K, dist = load_intrinsics(CAMERA_NAMES, INTRINSICS_DIR)
    cam_rvecs, cam_tvecs = load_initial_poses(INTRINSICS_DIR, CAMERA_NAMES)

    board_init, obs = detect_board_poses(board, dictionary, chessboard_3D, cam_rvecs, cam_tvecs, K, dist)
    obs = filter_by_depth(obs, board_init, cam_rvecs, cam_tvecs, chessboard_3D)

    sync_frame = select_sync_frame(board_init)

    frames, cam_params, board_params = initialize_parameters(board_init, cam_rvecs, cam_tvecs, sync_frame)

    residual_errors = compute_reprojection_errors(obs, cam_params, board_params, board_init, sync_frame)

    residual_errors, board_params, board_init, frames, bad_frames = prune_frames_to_target(
        residual_errors, board_params, board_init, frames, sync_frame, TARGET_FRAMES
    )

    n_keep = min(len(residual_errors), TARGET_RESIDUALS)
    obs_kept = [entry[1] for entry in residual_errors[:n_keep]]

    logger.info(
        "Kept %d residuals out of %d after pruning",
        n_keep,
        len(residual_errors),
    )

    summary = solve_bundle_adjustment(cam_params, board_params, obs_kept, board_init, frames, sync_frame)

    save_camera_poses(cam_params)

    top_errors = residual_errors[:10]
    logger.info("Top 10 highest residuals among kept observations:")
    for err, ci, f, pid in [e[0] for e in top_errors]:
        print(
            f"⚠️  Kept Error={err:.2f} px | Camera={CAMERA_NAMES[ci]} | Frame={f} | ID={pid}"
        )


if __name__ == "__main__":
    main()
