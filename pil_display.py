from PIL import Image
import os

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

fb_info = get_framebuffer_info()
WIDTH, HEIGHT = map(int, fb_info["virtual_size"].split(","))
BPP = fb_info["bpp"]

# Create horizontal stripe pattern
stripe_height = 20
img = Image.new("RGB", (WIDTH, HEIGHT))
for y in range(HEIGHT):
    color = 255 if (y // stripe_height) % 2 == 0 else 0
    for x in range(WIDTH):
        img.putpixel((x, y), (color, color, color))

# === Fastest Way to Write (for 32bpp) ===
if BPP == 32:
    # Add padding for 4-byte alignment
    bgra_img = Image.new("RGBA", img.size)
    bgra_img.paste(img)
    raw = bgra_img.tobytes("raw", "BGRA")  # BGR0
    with open("/dev/fb0", "wb") as f:
        f.write(raw)

elif BPP == 1
    # Slightly slower: convert RGB to RGB565
    import numpy as np
    arr = np.array(img)
    r = (arr[:, :, 0] >> 3).astype(np.uint16)
    g = (arr[:, :, 1] >> 2).astype(np.uint16)
    b = (arr[:, :, 2] >> 3).astype(np.uint16)
    rgb565 = (r << 11) | (g << 5) | b
    with open("/dev/fb0", "wb") as f:
        f.write(rgb565.astype("<u2").tobytes())  # little-endian uint16
