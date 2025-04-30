import cv2
import cv2.aruco as aruco
import numpy as np
from PIL import Image

# ----- Settings -----
paper_width_inches = 24
paper_height_inches = 24
dpi = 300  # print resolution
safe_margin_mm = 20  # larger buffer to avoid printer cutoff

# Constants
mm_per_inch = 25.4
page_width_mm = paper_width_inches * mm_per_inch - 2 * safe_margin_mm
page_height_mm = paper_height_inches * mm_per_inch - 2 * safe_margin_mm
image_width_px = int(paper_width_inches * dpi)
image_height_px = int(paper_height_inches * dpi)
margin_px = int(safe_margin_mm * dpi / 25.4)

# Helper to save a high-quality PDF
def save_high_quality_pdf(image_path, output_pdf_path):
    img = Image.open(image_path).convert('RGB')
    img.save(output_pdf_path, "PDF", resolution=300.0)

# Helper to generate a ChArUco board image and save it
def generate_charuco_board(squaresX, squaresY, aruco_dict_name, marker_to_square_ratio):
    aruco_dict = aruco.getPredefinedDictionary(aruco_dict_name)

    # Calculate square and marker sizes
    square_size_mm = min(page_width_mm / squaresX, page_height_mm / squaresY)
    marker_size_mm = square_size_mm * marker_to_square_ratio

    # Create board
    board = aruco.CharucoBoard(
        (squaresX, squaresY),
        squareLength=square_size_mm / 1000,  # mm -> meters
        markerLength=marker_size_mm / 1000,
        dictionary=aruco_dict
    )

    # Calculate board size in pixels
    board_width_px = int(squaresX * square_size_mm * dpi / 25.4)
    board_height_px = int(squaresY * square_size_mm * dpi / 25.4)

    # Generate the board image
    board_image = board.generateImage((board_width_px, board_height_px), marginSize=0, borderBits=1)

    # Paste onto full white canvas
    canvas = 255 * np.ones((image_height_px, image_width_px), dtype=np.uint8)
    offset_x = (image_width_px - board_width_px) // 2
    offset_y = (image_height_px - board_height_px) // 2
    canvas[offset_y:offset_y+board_height_px, offset_x:offset_x+board_width_px] = board_image

    # Auto-generate filenames
    dict_name = aruco_dict_name.name if hasattr(aruco_dict_name, 'name') else str(aruco_dict_name)
    base_filename = f"charuco_{squaresX}x{squaresY}_{dict_name}_margin{safe_margin_mm}mm"
    image_filename = base_filename + ".png"
    pdf_filename = base_filename + ".pdf"

    # Save PNG
    cv2.imwrite(image_filename, canvas)

    # Save high-quality PDF
    save_high_quality_pdf(image_filename, pdf_filename)

# ----- Generate Side A (Close-up, larger squares) -----
generate_charuco_board(
    squaresX=8,
    squaresY=8,
    aruco_dict_name=aruco.DICT_6X6_1000,
    marker_to_square_ratio=0.7
)

# ----- Generate Side B (Farther, smaller squares) -----
generate_charuco_board(
    squaresX=16,
    squaresY=16,
    aruco_dict_name=aruco.DICT_4X4_1000,
    marker_to_square_ratio=0.75
)
