import cv2
import cv2.aruco as aruco
import numpy as np
import glob, os

# ----- Board definition (Side B: 16×16, DICT_4X4_1000) -----
paper_w_in = paper_h_in = 24.0
safe_margin_mm = 20.0
mm_per_inch = 25.4

page_w_mm = paper_w_in * mm_per_inch - 2 * safe_margin_mm
page_h_mm = paper_h_in * mm_per_inch - 2 * safe_margin_mm

squaresX = squaresY = 16
square_size_mm = min(page_w_mm / squaresX, page_h_mm / squaresY)
marker_size_mm = square_size_mm * 0.75

square_length_m = square_size_mm / 1000.0
marker_length_m = marker_size_mm / 1000.0

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)
board = aruco.CharucoBoard(
    (squaresX, squaresY),
    squareLength=square_length_m,
    markerLength=marker_length_m,
    dictionary=aruco_dict
)

# ----- Paths -----
root_dir = "captures"
output_dir = "calibration_results"
os.makedirs(output_dir, exist_ok=True)

# ----- Sub-pixel refinement parameters -----
subpix_win = (5, 5)
subpix_zero = (-1, -1)
subpix_criteria = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    30, 1e-3
)

# ----- Extended calibration flags & criteria -----
ext_flags = cv2.CALIB_RATIONAL_MODEL
ext_criteria = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    100, 1e-6
)

# Thresholds for pruning
VIEW_ERR_THRESH = 1.3    # allow up to 1.3 px reprojection error
STD_INT_THRESH  = 0.3    # allow more intrinsics variation
STD_EXT_THRESH  = 0.3    # allow more extrinsics variation

for cam in sorted(os.listdir(root_dir)):
    cam_path = os.path.join(root_dir, cam)
    if not os.path.isdir(cam_path):
        continue

    print(f"\n🔍 Calibrating intrinsics for {cam}")
    charuco_corners = []
    charuco_ids = []
    image_size = None

    # 1) Detect & refine corners
    for img_file in sorted(glob.glob(os.path.join(cam_path, "*.jpg"))):
        img = cv2.imread(img_file)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        image_size = gray.shape[::-1]

        corners, ids, _ = aruco.detectMarkers(gray, aruco_dict)
        if ids is None or len(ids) < 4:
            continue

        _, cc, cids = aruco.interpolateCornersCharuco(
            corners, ids, gray, board
        )
        if cc is None or cids is None or len(cids) < 4:
            continue

        # sub-pixel refine
        pts = cc.reshape(-1,1,2)
        cv2.cornerSubPix(
            gray, pts, subpix_win, subpix_zero, subpix_criteria
        )
        charuco_corners.append(pts)
        charuco_ids.append(cids)

    print(f"  Collected {len(charuco_corners)} frames")
    if len(charuco_corners) < 10:
        print("  ❌ Not enough frames, skipping")
        continue

    # 2) Extended calibration (rational model)
    print("  ▶ Running calibrateCameraCharucoExtended…")
    (ret_ext, K_ext, dist_ext,
     rvecs_ext, tvecs_ext,
     std_int, std_ext,
     per_view_err) = aruco.calibrateCameraCharucoExtended(
        charucoCorners=charuco_corners,
        charucoIds=charuco_ids,
        board=board,
        imageSize=image_size,
        cameraMatrix=None,
        distCoeffs=None,
        flags=ext_flags,
        criteria=ext_criteria
    )
    print(f"  Extended RMS: {ret_ext:.4f} px")

    # 3) Prune bad frames based on per-view error + stddevs
    keep_corners = []
    keep_ids = []
    for cc, cids, err, si, se in zip(
            charuco_corners, charuco_ids,
            per_view_err, std_int, std_ext
    ):
        if err <= VIEW_ERR_THRESH and np.max(si) <= STD_INT_THRESH and np.max(se) <= STD_EXT_THRESH:
            keep_corners.append(cc)
            keep_ids.append(cids)
    print(f"  Pruned to {len(keep_corners)} frames (err<={VIEW_ERR_THRESH}px, std_int<={STD_INT_THRESH}, std_ext<={STD_EXT_THRESH})")

    if len(keep_corners) < 5:
        print("  ❌ Too few frames after pruning, skipping")
        continue

    # 4) Final calibration on pruned set (default 5-parameter model)
    print("  ▶ Running calibrateCameraCharuco…")
    ret_final, K_final, dist_final, _, _ = aruco.calibrateCameraCharuco(
        charucoCorners=keep_corners,
        charucoIds=keep_ids,
        board=board,
        imageSize=image_size,
        cameraMatrix=None,
        distCoeffs=None
    )
    print(f"  ✅ Final RMS: {ret_final:.4f} px")

    fx, fy = K_final[0,0], K_final[1,1]
    w, h = image_size
    print(f"  K =\n{K_final}")
    print(f"  fx≈w/2? {fx:.1f} vs {w/2:.1f}, fy≈h/2? {fy:.1f} vs {h/2:.1f}")
    print(f"  dist = {dist_final.ravel()}")

    # Save refined intrinsics
    out_file = os.path.join(
        output_dir, f"{cam}_intrinsics_refined_full.npz"
    )
    np.savez(out_file, K=K_final, dist=dist_final, image_size=image_size)
    print(f"  Saved to {out_file}")
