import cv2
import os

# === Output config ===
pattern_width = 1920   # projector resolution
pattern_height = 1080
output_dir = "graycode_patterns"
os.makedirs(output_dir, exist_ok=True)

# === Create pattern generator ===
graycode = cv2.structured_light.GrayCodePattern.create(pattern_width, pattern_height)

# === Generate patterns ===
print("🧠 Generating Gray code patterns...")
ok, patterns = graycode.generate()
if not ok:
    raise RuntimeError("Failed to generate Gray code patterns.")

print(f"✅ Generated {len(patterns)} patterns")

# === Save and preview ===
for i, pattern in enumerate(patterns):
    filename = os.path.join(output_dir, f"gray_{i:02d}.png")
    cv2.imwrite(filename, pattern)
    print(f"🖼 Saved: {filename}")

print("\n📁 All patterns saved to:", output_dir)