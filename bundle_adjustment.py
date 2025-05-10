# bundle_adjustment_sync.py

import cv2
import cv2.aruco as aruco
import numpy as np
import os, glob, json
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

print("\n🚀 Starting camera-only BA with synchronized solvePnP init…")

# === Charuco board (Side B: 16×16, DICT_4X4_1000) ===
paper_w_in = paper_h_in = 24.0
safe_margin_mm = 20.0
mm_per_inch = 25.4
page_w_mm = paper_w_in * mm_per_inch - 2 * safe_margin_mm

squaresX = squaresY = 16
square_size_mm = page_w_mm / squaresX    # in mm
marker_size_mm = square_size_mm * 0.75   # in mm

# define board in mm units so object points are O(10-20)
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)
board = aruco.CharucoBoard(
    (squaresX, squaresY),
    squareLength=square_size_mm,
    markerLength=marker_size_mm,
    dictionary=aruco_dict
)
chessboard_3D = board.getChessboardCorners()

# === I/O & camera list ===
camera_names = [f"picam{i}.local" for i in range(6)]
intrinsics_dir = "calibration_results"
captures_dir   = "captures"
image_ext      = "*.jpg"

# --- load refined intrinsics ---
print("\n📅 Loading refined intrinsics…")
K_dict, dist_dict = {}, {}
for cam in camera_names:
    fn = os.path.join(intrinsics_dir, f"{cam}_intrinsics_refined_full.npz")
    data = np.load(fn)
    K_dict[cam]    = data["K"]
    dist_dict[cam] = data["dist"]
    fx, fy = data["K"][0,0], data["K"][1,1]
    cx, cy = data["K"][0,2], data["K"][1,2]
    print(f"  {cam}: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")
print("✅ Refined intrinsics loaded!")

# === pick a synchronized frame index ===
SYNC_IDX = 3  # change to the shot number you want to anchor on

# build sorted lists and sync files
img_lists = {cam: sorted(glob.glob(os.path.join(captures_dir, cam, image_ext))) for cam in camera_names}
assert all(len(img_lists[cam]) == len(img_lists[camera_names[0]]) for cam in camera_names), "Unsynced image counts!"
sync_files = {cam: img_lists[cam][SYNC_IDX] for cam in camera_names}

# === per‑camera solvePnP on that synchronized shot ===
print(f"\n🎯 Anchoring all cams on frame index {SYNC_IDX}…")
global_RTs = {}
for cam, path in sync_files.items():
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = aruco.detectMarkers(gray, aruco_dict)
    _, cc, cids = aruco.interpolateCornersCharuco(corners, ids, gray, board)
    if cids is None or len(cids) < 20:
        raise RuntimeError(f"{cam}: too few Charuco corners ({0 if cids is None else len(cids)})")

    # undistort detected corners
    undist_corners = cv2.undistortPoints(cc, K_dict[cam], dist_dict[cam], P=K_dict[cam])
    obj_pts = np.array([chessboard_3D[int(cid)] for cid in cids.flatten()], dtype=np.float32)
    img_pts = undist_corners.reshape(-1,2).astype(np.float32)

    # initial solvePnP
    _, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K_dict[cam], None, flags=cv2.SOLVEPNP_ITERATIVE)
    # refine with LM
    rvec, tvec = cv2.solvePnPRefineLM(obj_pts, img_pts, K_dict[cam], None, rvec, tvec,
                                      criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6))
    R, _ = cv2.Rodrigues(rvec)
    T = tvec.reshape(3,1)

    # compute init reproj RMS (with distortion)
    proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K_dict[cam], dist_dict[cam])
    errs = np.linalg.norm(proj.reshape(-1,2) - cc.reshape(-1,2), axis=1)
    rms = np.sqrt(np.mean(errs**2))
    print(f"  {cam}: |T|={np.linalg.norm(T):.3f} mm, corners={len(obj_pts)}, init RMS={rms:.2f} px")
    if rms > 10:
        print(f"    ❌ rejected (RMS {rms:.1f}px)")
        continue
    global_RTs[cam] = (R, T)
print("✅ All cameras anchored.\n")

# === extract observations only from the synchronized frame ===
print("🔍 Extracting observations from sync frame only…")
obs = []
all_pts3d = {}
for cam, path in sync_files.items():
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = aruco.detectMarkers(gray, aruco_dict)
    if ids is None: continue
    _, cc, cids = aruco.interpolateCornersCharuco(corners, ids, gray, board)
    if cids is None: continue
    # undistort corners for consistency
    undist_corners = cv2.undistortPoints(cc, K_dict[cam], dist_dict[cam], P=K_dict[cam])
    img_pts = undist_corners.reshape(-1,2)
    for i, cid in enumerate(cids.flatten()):
        pid = str(int(cid))
        all_pts3d.setdefault(pid, chessboard_3D[int(cid)])
        obs.append({"camera":cam, "point_id":pid, "xy":cc[i,0].tolist()})
print(f"  Total observations: {len(obs)} from sync frame")

# === initial reprojection RMS check ===
errs0 = []
for o in obs:
    R,T = global_RTs[o["camera"]]
rvec,_ = cv2.Rodrigues(R)
# project using distortion to compare to raw image corners
pr,_ = cv2.projectPoints(all_pts3d[o["point_id"]].reshape(1,1,3), rvec, T, K_dict[o["camera"]], dist_dict[o["camera"]])
errs0.append(np.linalg.norm(pr[0,0] - o["xy"]))
init_rms = np.sqrt(np.mean(np.square(errs0)))
print(f"📈 Initial RMS reprojection error (sync only): {init_rms:.2f} px")
errs0 = []
for o in obs:
    R,T = global_RTs[o["camera"]]
rvec,_ = cv2.Rodrigues(R)
pr,_ = cv2.projectPoints(all_pts3d[o["point_id"]].reshape(1,1,3), rvec, T, K_dict[o["camera"]], dist_dict[o["camera"]])
errs0.append(np.linalg.norm(pr[0,0] - o["xy"]))
init_rms = np.sqrt(np.mean(np.square(errs0)))
print(f"📈 Initial RMS reprojection error: {init_rms:.2f} px\n")

# === bundle‑adjust only extrinsics ===
def bundle_adjust_cam_only(obs, Kd, initRT, pts3d, dd):
    cams = list(initRT.keys())
    idx = {c:i for i,c in enumerate(cams)}
    x0 = np.zeros(len(cams)*6)
    for c,(R,T) in initRT.items():
        i = idx[c]
        rv,_ = cv2.Rodrigues(R)
        x0[i*6:i*6+3] = rv.flatten()
        x0[i*6+3:i*6+6] = T.flatten()

    def residuals(x):
        r = []
        for o in obs:
            i = idx[o["camera"]]
            rv = x[i*6:i*6+3]
            tv = x[i*6+3:i*6+6].reshape(3,1)
            pr,_ = cv2.projectPoints(pts3d[o["point_id"]].reshape(1,1,3), rv, tv, Kd[o["camera"]], dd[o["camera"]])
            r.extend((pr[0,0]-o["xy"]).tolist())
        return np.array(r)

    print("🔧 Running BA (LM)…")
    sol = least_squares(residuals, x0, method='lm', verbose=2)
    refined = {}
    for c,i in idx.items():
        rv = sol.x[i*6:i*6+3]
        tv = sol.x[i*6+3:i*6+6].reshape(3,1)
        Rf,_ = cv2.Rodrigues(rv)
        refined[c] = (Rf, tv)
    return refined

refined_RTs = bundle_adjust_cam_only(obs, K_dict, global_RTs, all_pts3d, dist_dict)

# === final reprojection error ===
errs1 = []
for o in obs:
    R,T = refined_RTs[o["camera"]]
    rvec,_ = cv2.Rodrigues(R)
    pr,_ = cv2.projectPoints(all_pts3d[o["point_id"]].reshape(1,1,3), rvec, T, K_dict[o["camera"]], dist_dict[o["camera"]])
    errs1.append(np.linalg.norm(pr[0,0] - o["xy"]))
mean1 = np.mean(errs1)
rms1  = np.sqrt(np.mean(np.square(errs1)))
print(f"\n📊 Final mean: {mean1:.2f} px, RMS: {rms1:.2f} px")

# === save refined poses ===
os.makedirs("results", exist_ok=True)
with open("results/refined_camera_poses.json","w") as f:
    json.dump({c:{"R":R.tolist(),"T":T.flatten().tolist()} for c,(R,T) in refined_RTs.items()}, f, indent=2)
print("✅ Refined poses saved to results/refined_camera_poses.json")

# === histogram ===
plt.figure()
plt.hist(errs1, bins=50)
plt.xlabel("Reproj error (px)")
plt.ylabel("Count")
plt.title("Post-BA reprojection errors")
plt.show()

print("\n✅ Done!")
