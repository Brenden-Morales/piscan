from PIL import Image
import numpy as np
import os
import time
import glob

# === Settings ===
PATTERN_DIR = "graycode_patterns"

def read_file(path):
    with open(path, 'r') as f:
        return f.read().strip()

def get_framebuffer_info(fbdev='fb0'):
    base = f"/sys/class/graphics/{fbdev}"
    return {
        "fbdev": f"/dev/{fbdev}",
        "virtual_size": read_file(os.path.join(base, "virtual_size")),
        "bpp": int(read_file(os.path.join(base, "bits_per_pixel")))
    }

def show_image_on_fb(img, fb_path, bpp):
    if bpp == 32:
        bgra = Image.new("RGBA", img.size)
        bgra.paste(img)
        raw = bgra.tobytes("raw", "BGRA")
        with open(fb_path, "wb") as f:
            f.write(raw)
    elif bpp == 16:
        arr = np.array(img)
        r = (arr[:, :, 0] >> 3).astype(np.uint16)
        g = (arr[:, :, 1] >> 2).astype(np.uint16)
        b = (arr[:, :, 2] >> 3).astype(np.uint16)
        rgb565 = (r << 11) | (g << 5) | b
        with open(fb_path, "wb") as f:
            f.write(rgb565.astype("<u2").tobytes())
    else:
        raise NotImplementedError(f"BPP={bpp} not supported yet")

# === Main ===
fb_info = get_framebuffer_info()
WIDTH, HEIGHT = map(int, fb_info["virtual_size"].split(","))
BPP = fb_info["bpp"]
FB_PATH = fb_info["fbdev"]

# Load images from disk
pattern_paths = sorted(glob.glob(os.path.join(PATTERN_DIR, "*.png")))
if not pattern_paths:
    raise FileNotFoundError(f"No .png files found in {PATTERN_DIR}")

print(f"📂 Found {len(pattern_paths)} patterns in {PATTERN_DIR}")
print(f"📽️ Displaying to {FB_PATH} ({WIDTH}x{HEIGHT}, {BPP} bpp)")

for i, path in enumerate(pattern_paths):
    input(f"🔎 Press [Enter] to show pattern {i+1}/{len(pattern_paths)}: {os.path.basename(path)}")
    img = Image.open(path).convert("RGB").resize((WIDTH, HEIGHT))
    show_image_on_fb(img, FB_PATH, BPP)