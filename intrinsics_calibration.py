import cv2
import cv2.aruco as aruco
import numpy as np
import glob, os, shutil

# -----------------------------------------------------------------------------
# 0) Charuco board setup (16×16, DICT_4X4_1000)
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# 1) Paths & parameters
# -----------------------------------------------------------------------------
root_dir    = "captures"
output_dir  = "calibration_results"
os.makedirs(output_dir, exist_ok=True)

subpix_win      = (11, 11)
subpix_zero     = (-1, -1)
subpix_criteria = (
    cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
    400,
    1e-7
)

final_flags = 0
final_criteria = (
    cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
    400,
    1e-7
)

max_views_kept = 25

# -----------------------------------------------------------------------------
# 2) Loop over each camera
# -----------------------------------------------------------------------------
for cam in sorted(os.listdir(root_dir)):
    cam_path = os.path.join(root_dir, cam)
    if not os.path.isdir(cam_path):
        continue

    print(f"\n🔍 Calibrating intrinsics for {cam}")
    charuco_corners = []
    charuco_ids     = []
    file_paths      = []
    image_size      = None

    for fn in sorted(glob.glob(os.path.join(cam_path, "*.jpg"))):
        img = cv2.imread(fn)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        image_size = gray.shape[::-1]

        m_corners, m_ids, _ = aruco.detectMarkers(gray, aruco_dict)
        if m_ids is None or len(m_ids) < 4:
            continue
        _, cc, cids = aruco.interpolateCornersCharuco(
            m_corners, m_ids, gray, board
        )
        if cids is None or len(cids) < 20:
            continue

        pts = cc.reshape(-1,1,2)
        cv2.cornerSubPix(gray, pts, subpix_win, subpix_zero, subpix_criteria)
        charuco_corners.append(pts)
        charuco_ids.append(cids)
        file_paths.append(fn)

    n_views = len(charuco_corners)
    print(f"  Collected {n_views} Charuco views")
    if n_views < max_views_kept:
        print(f"  ❌ Too few frames (need ≥{max_views_kept}), skipping")
        continue

    # -----------------------------------------------------------------------------
    # 3) Initial calibration (for reprojection errors)
    # -----------------------------------------------------------------------------
    print("  ▶ Running initial calibrateCameraCharuco…")
    rms, K, dist, rvecs, tvecs = aruco.calibrateCameraCharuco(
        charucoCorners=charuco_corners,
        charucoIds=charuco_ids,
        board=board,
        imageSize=image_size,
        cameraMatrix=None,
        distCoeffs=None,
        flags=final_flags,
        criteria=final_criteria
    )
    print(f"  Initial RMS: {rms:.4f} px")

    # -----------------------------------------------------------------------------
    # 4) Compute per-view reprojection errors
    # -----------------------------------------------------------------------------
    per_view_err = []
    for i in range(len(charuco_corners)):
        img_points_proj, _ = cv2.projectPoints(
            board.getChessboardCorners()[charuco_ids[i].flatten()],
            rvecs[i], tvecs[i], K, dist
        )
        err = cv2.norm(charuco_corners[i], img_points_proj, cv2.NORM_L2) / len(img_points_proj)
        per_view_err.append((i, err))

    # -----------------------------------------------------------------------------
    # 5) Keep only the best N views
    # -----------------------------------------------------------------------------
    per_view_err.sort(key=lambda x: x[1])
    keep_idxs = [i for i, _ in per_view_err[:max_views_kept]]

    keep_corners = [charuco_corners[i] for i in keep_idxs]
    keep_ids     = [charuco_ids[i] for i in keep_idxs]

    print(f"  Keeping {len(keep_corners)} best views based on reprojection error")

    # Optional: Save kept images
    keep_dir = os.path.join(output_dir, f"{cam}_kept")
    os.makedirs(keep_dir, exist_ok=True)
    for i in keep_idxs:
        shutil.copy(file_paths[i], os.path.join(keep_dir, os.path.basename(file_paths[i])))

    # -----------------------------------------------------------------------------
    # 6) Final calibration
    # -----------------------------------------------------------------------------
    print("  ▶ Running final calibrateCameraCharuco…")
    final_rms, K_final, dist_final, _, _ = aruco.calibrateCameraCharuco(
        charucoCorners=keep_corners,
        charucoIds=keep_ids,
        board=board,
        imageSize=image_size,
        cameraMatrix=K,
        distCoeffs=dist,
        flags=(final_flags | cv2.CALIB_USE_INTRINSIC_GUESS),
        criteria=final_criteria
    )
    print(f"  ✅ Final RMS: {final_rms:.4f} px")

    # -----------------------------------------------------------------------------
    # 7) Save results
    # -----------------------------------------------------------------------------
    out_path = os.path.join(output_dir, f"{cam}_intrinsics_refined.npz")
    np.savez(out_path,
             K=K_final,
             dist=dist_final,
             image_size=image_size,
             rms_initial=rms,
             rms_final=final_rms)
    print(f"  Saved enhanced intrinsics → {out_path}")
