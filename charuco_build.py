import cv2
import cv2.aruco as aruco
import numpy as np

# ----- Settings -----
paper_width_inches = 8.5
paper_height_inches = 11
dpi = 300  # print resolution
safe_margin_mm = 5  # small buffer to avoid printer cutoff

# Board dimensions
squaresX = 11   # number of squares across
squaresY = 15   # number of squares down
square_mm = 17  # size of a square
marker_mm = 12  # size of marker inside square

# Dictionary
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_7X7_1000)

# ----- Calculate page size -----
mm_per_inch = 25.4
page_width_mm = paper_width_inches * mm_per_inch - 2 * safe_margin_mm
page_height_mm = paper_height_inches * mm_per_inch - 2 * safe_margin_mm

# Determine the maximum square size to fit the page
square_size_mm = min(page_width_mm / squaresX, page_height_mm / squaresY)
marker_size_mm = square_size_mm * (marker_mm / square_mm)

# Generate board with max square size
board = aruco.CharucoBoard(
    (squaresX, squaresY),
    squareLength=square_size_mm / 1000,
    markerLength=marker_size_mm / 1000,
    dictionary=aruco_dict
)

# Convert paper size to pixels
image_width_px = int(paper_width_inches * dpi)
image_height_px = int(paper_height_inches * dpi)

# Generate the board image to exactly fit the printable area
board_width_px = int(squaresX * square_size_mm * dpi / 25.4)
board_height_px = int(squaresY * square_size_mm * dpi / 25.4)
margin_px = int(safe_margin_mm * dpi / 25.4)

# Draw the board
board_image = board.generateImage((board_width_px, board_height_px), marginSize=0, borderBits=1)

# Paste onto full white canvas
canvas = 255 * np.ones((image_height_px, image_width_px), dtype=np.uint8)
offset_x = (image_width_px - board_width_px) // 2
offset_y = (image_height_px - board_height_px) // 2
canvas[offset_y:offset_y+board_height_px, offset_x:offset_x+board_width_px] = board_image

# Save it
cv2.imwrite("charuco_board_fullpage_8.5x11.png", canvas)