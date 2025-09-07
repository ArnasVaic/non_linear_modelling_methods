# %%

import numpy as np
import matplotlib.pyplot as plt

width = 1000
height = 1000

with open("input.txt", "r") as f:
    pixels = [float(line.strip()) for line in f]

img = np.array(pixels).reshape((height, width))

plt.imshow(
    np.flipud(img),
    origin='upper',
    cmap='inferno', 
    interpolation='nearest', 
    extent=[-5,5,-5,5]
)
plt.colorbar()
plt.show()
