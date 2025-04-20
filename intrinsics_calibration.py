import cv2
import cv2.aruco as aruco
import numpy as np
import glob
import os

# ----- Board definition -----
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_7X7_1000)
board = aruco.CharucoBoard(
    (11, 15),        # squaresX, squaresY
    0.017,           # squareLength in meters
    0.012,           # markerLength in meters
    aruco_dict
)

# ----- Paths -----
root_dir = "captures"
output_dir = "calibration_results"
os.makedirs(output_dir, exist_ok=True)

# ----- Process each camera -----
for cam_folder in sorted(os.listdir(root_dir)):
    cam_path = os.path.join(root_dir, cam_folder)
    if not os.path.isdir(cam_path):
        continue

    print(f"\n🔍 Calibrating camera: {cam_folder}")
    all_charuco_corners = []
    all_charuco_ids = []
    image_size = None

    image_files = sorted(glob.glob(os.path.join(cam_path, "*.jpg")))
    for img_file in image_files:
        img = cv2.imread(img_file)
        if img is None:
            print(f"⚠️ Could not load image: {img_file}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        image_size = gray.shape[::-1]

        corners, ids, _ = aruco.detectMarkers(gray, aruco_dict)
        print(f"[1] Detected {len(ids) if ids is not None else 0} ArUco markers")

        if ids is not None and len(ids) > 3:
            ret, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
                markerCorners=corners,
                markerIds=ids,
                image=gray,
                board=board
            )

            if (
                    charuco_corners is not None and
                    charuco_ids is not None and
                    len(charuco_ids) > 3
            ):
                print(f"[2] Interpolated {len(charuco_corners)} ChArUco corners")
                all_charuco_corners.append(charuco_corners)
                all_charuco_ids.append(charuco_ids)
            else:
                print("[2] ChArUco interpolation failed")
        else:
            print("[2] Not enough ArUco markers to interpolate")

    print(f"🧩 Total valid ChArUco frames: {len(all_charuco_corners)}")
    print(f"🖼️ Image size: {image_size}")

    # ----- Validate per-frame geometry -----
    filtered_corners = []
    filtered_ids = []

    print(f"🧪 Testing geometry on {len(all_charuco_corners)} candidates...")

    for c, i in zip(all_charuco_corners, all_charuco_ids):
        if c is not None and i is not None and len(i) >= 10:
            try:
                # Try single-frame calibration to check geometry
                cv2.aruco.calibrateCameraCharuco(
                    charucoCorners=[c],
                    charucoIds=[i],
                    board=board,
                    imageSize=image_size,
                    cameraMatrix=None,
                    distCoeffs=None
                )
                filtered_corners.append(c)
                filtered_ids.append(i)
            except cv2.error:
                print("⚠️ Skipping one bad frame due to unusable ChArUco geometry")

    print(f"✅ Filtered to {len(filtered_corners)} safe frames")

    if len(filtered_corners) < 5:
        print(f"❌ Skipping {cam_folder} — not enough usable frames after geometry check\n")
        continue

    # ----- Calibrate -----
    try:
        ret, K, dist, rvecs, tvecs = aruco.calibrateCameraCharuco(
            charucoCorners=filtered_corners,
            charucoIds=filtered_ids,
            board=board,
            imageSize=image_size,
            cameraMatrix=None,
            distCoeffs=None
        )
    except cv2.error as e:
        print(f"❌ Calibration failed for {cam_folder}: {e}")
        continue

    print(f"✅ Done calibrating {cam_folder}")
    print(f"📐 RMS reprojection error: {ret:.4f}")
    print("📸 Camera matrix:\n", K)
    print("🎯 Distortion coefficients:\n", dist.ravel())

    # ----- Save calibration -----
    out_file = os.path.join(output_dir, f"{cam_folder}_intrinsics.npz")
    np.savez(out_file, K=K, dist=dist, image_size=image_size)
    print(f"📁 Saved to {out_file}\n")
