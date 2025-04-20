import cv2
import cv2.aruco as aruco
import numpy as np
import os
import glob
import json

# === Parameters ===
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_7X7_1000)
board = aruco.CharucoBoard((11, 15), 0.017, 0.012, aruco_dict)
chessboard_3D = board.getChessboardCorners()

camera_names = [f"picam{i}.local" for i in range(6)]
calib_dir = "calibration_results"
captures_dir = "captures"
image_ext = "*.jpg"
min_common_corners = 20

os.makedirs(calib_dir, exist_ok=True)

# === Load intrinsics ===
intrinsics = {}
print("📥 Loading camera intrinsics...")
for cam in camera_names:
    intr_path = os.path.join(calib_dir, f"{cam}_intrinsics.npz")
    if not os.path.exists(intr_path):
        raise FileNotFoundError(f"❌ Missing: {intr_path}")
    data = np.load(intr_path)
    intrinsics[cam] = {
        "K": data["K"],
        "dist": data["dist"],
        "image_size": tuple(data["image_size"])
    }
print("✅ Intrinsics loaded for all cameras!")

# === Initialize global pose map ===
global_RTs = {camera_names[0]: (np.eye(3), np.zeros((3, 1)))}

# === Loop through stereo pairs ===
for i in range(len(camera_names) - 1):
    camA = camera_names[i]
    camB = camera_names[i + 1]

    print(f"\n🔁 Calibrating stereo pair: {camA} ↔ {camB}")

    K0, dist0 = intrinsics[camA]["K"], intrinsics[camA]["dist"]
    K1, dist1 = intrinsics[camB]["K"], intrinsics[camB]["dist"]
    image_size = intrinsics[camA]["image_size"]

    imgsA = sorted(glob.glob(os.path.join(captures_dir, camA, image_ext)))
    imgsB = sorted(glob.glob(os.path.join(captures_dir, camB, image_ext)))
    assert len(imgsA) == len(imgsB), f"❌ Image count mismatch: {camA}, {camB}"

    obj_points, img_points0, img_points1 = [], [], []

    for img0_path, img1_path in zip(imgsA, imgsB):
        img0 = cv2.imread(img0_path)
        img1 = cv2.imread(img1_path)
        gray0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

        corners0, ids0, _ = aruco.detectMarkers(gray0, aruco_dict)
        corners1, ids1, _ = aruco.detectMarkers(gray1, aruco_dict)
        if ids0 is None or ids1 is None:
            continue

        _, c0, id0 = aruco.interpolateCornersCharuco(corners0, ids0, gray0, board)
        _, c1, id1 = aruco.interpolateCornersCharuco(corners1, ids1, gray1, board)
        if c0 is None or c1 is None:
            continue

        common_ids = np.intersect1d(id0.flatten(), id1.flatten())
        if len(common_ids) < min_common_corners:
            print(f"⚠️ Skipping frame — only {len(common_ids)} common corners")
            continue

        pts3D, pts0, pts1 = [], [], []
        for cid in common_ids:
            idx0 = np.where(id0 == cid)[0][0]
            idx1 = np.where(id1 == cid)[0][0]
            pts3D.append(chessboard_3D[cid])
            pts0.append(c0[idx0][0])
            pts1.append(c1[idx1][0])

        # Optional geometry validation
        try:
            cv2.stereoCalibrate(
                [np.array(pts3D, dtype=np.float32)],
                [np.array(pts0, dtype=np.float32)],
                [np.array(pts1, dtype=np.float32)],
                K0, dist0, K1, dist1, image_size,
                flags=cv2.CALIB_FIX_INTRINSIC,
                criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-5)
            )
        except cv2.error:
            print("🛑 Skipping frame — bad geometry")
            continue

        obj_points.append(np.array(pts3D, dtype=np.float32))
        img_points0.append(np.array(pts0, dtype=np.float32))
        img_points1.append(np.array(pts1, dtype=np.float32))

    print(f"🧩 Valid stereo pairs: {len(obj_points)}")
    if len(obj_points) < 5:
        print(f"❌ Not enough stereo data to calibrate {camA} ↔ {camB}")
        continue

    print("🧪 Running stereo calibration...")
    retval, K0, dist0, K1, dist1, R, T, E, F = cv2.stereoCalibrate(
        objectPoints=obj_points,
        imagePoints1=img_points0,
        imagePoints2=img_points1,
        cameraMatrix1=K0,
        distCoeffs1=dist0,
        cameraMatrix2=K1,
        distCoeffs2=dist1,
        imageSize=image_size,
        flags=cv2.CALIB_FIX_INTRINSIC,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
    )

    print(f"📐 RMS error for {camA} ↔ {camB}: {retval:.4f}")
    out_path = os.path.join(calib_dir, f"stereo_calibration_{camA}_{camB}.npz")
    np.savez(out_path, R=R, T=T, E=E, F=F, error=retval)

    # Propagate global pose to camB
    R_prev, T_prev = global_RTs[camA]
    R_global = R @ R_prev
    T_global = R @ T_prev + T
    global_RTs[camB] = (R_global, T_global)
    print(f"🧭 Computed pose for {camB} relative to {camera_names[0]}")

# === Save all global poses ===
pose_json = {}
for cam, (R, T) in global_RTs.items():
    pose_json[cam] = {
        "R": np.round(R, 6).tolist(),
        "T": np.round(T, 6).tolist()
    }

global_json_path = os.path.join(calib_dir, "multi_camera_global_poses.json")
with open(global_json_path, "w") as f:
    json.dump(pose_json, f, indent=2)

print("\n✅ All stereo pairs calibrated")
print("🗂 Global poses saved to calibration_results/multi_camera_global_poses.json")