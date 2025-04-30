import os
import cv2
import numpy as np
import glob

# === Config ===
root_dir = "captures"
camera_names = [f"picam{i}.local" for i in range(6)]
threshold = 128
debug = True

def load_and_sort_images(path, pattern_prefix):
    files = sorted(glob.glob(os.path.join(path, f"{pattern_prefix}_*.jpg")))
    if debug:
        print(f"📂 {os.path.basename(path)} [{pattern_prefix}]: Found {len(files)} captured frames")
    return files

def decode_gray_code_image_stack(image_paths):
    num_bits = len(image_paths)
    if num_bits == 0:
        raise ValueError("No images provided.")

    binaries = []
    for i, path in enumerate(image_paths):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"Failed to load {path}")
        binary = (img > threshold).astype(np.uint16)
        binaries.append(binary)
        if debug:
            print(f"🧠 Processed pattern {i}: {os.path.basename(path)}")

    stack = np.stack(binaries, axis=-1)

    # Convert from Gray code to binary
    gray_values = np.zeros(stack.shape[:2], dtype=np.uint16)
    for i in range(num_bits):
        gray_values |= (stack[:, :, i] << (num_bits - i - 1))

    def gray_to_binary(gray):
        binary = np.zeros_like(gray)
        for i in range(num_bits):
            bit = (gray >> i) & 1
            if i == 0:
                binary |= bit << i
            else:
                prev = (binary >> (i - 1)) & 1
                binary |= (bit ^ prev) << i
        return binary

    binary_coords = gray_to_binary(gray_values)
    return binary_coords

# === Loop through each camera folder ===
for cam in camera_names:
    cam_dir = os.path.join(root_dir, cam)
    if not os.path.isdir(cam_dir):
        print(f"🚫 Skipping missing directory: {cam_dir}")
        continue

    print(f"\n🔍 Decoding Gray codes for {cam}")

    for orientation in ["horizontal", "vertical"]:
        try:
            image_paths = load_and_sort_images(cam_dir, f"gray_{orientation}")
            decoded = decode_gray_code_image_stack(image_paths)
        except Exception as e:
            print(f"❌ Failed to decode {orientation} for {cam}: {e}")
            continue

        coord_type = "projector_x_coords" if orientation == "horizontal" else "projector_y_coords"
        out_path = os.path.join(cam_dir, f"{coord_type}.npy")
        np.save(out_path, decoded)
        print(f"✅ Saved {coord_type} to {out_path}")

        # Save preview
        preview = (255 * (decoded / decoded.max())).astype(np.uint8)
        cv2.imwrite(os.path.join(cam_dir, f"decoded_{coord_type}.png"), preview)
        print(f"🖼 Preview saved to {cam_dir}/decoded_{coord_type}.png")