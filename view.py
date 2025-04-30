import matplotlib.pyplot as plt
import numpy as np

img = np.load("captures/picam0.local/projector_x_coords.npy")
plt.imshow(img, cmap="jet")
plt.title("Decoded projector X")
plt.colorbar()
plt.show()