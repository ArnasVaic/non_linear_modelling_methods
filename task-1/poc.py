# %%

import numpy as np
import matplotlib.pyplot as plt

# proof of concept

alpha = 0.2
beta = 1.01

nx, ny = (500, 500)

extent = [
  -5, 5, # x min max
  -5, 5 # y min max
]

iter = 0

# xy coordinates
x = np.linspace(extent[0], extent[1], nx)
y = np.linspace(extent[2], extent[3], ny)
# mesh
X, Y = np.meshgrid(x, y)

result = np.zeros((nx, ny))

target = 100

while True:
  x_new = 1 - alpha * X * X + Y
  y_new = beta * X
  iter = iter + 1

  # check which entries reached target
  # assign current iteration

  # mask out entries that reached target
  # shape [nx, ny], each entry -- boolean indicating if cell reached target
  cond = np.abs(x_new) + np.abs(y_new) > target

  # elements that have not been set yet
  unset = result == 0

  if np.all(result > 0):
    break # picture complete

  # we only want to set those elements which were not set before AND reached the target!!!!
  mask = unset & cond

  result[mask] = iter

  # update entries
  X = x_new
  Y = y_new

np.save('henon.npy', result)

# %%
result = np.load('henon.npy')

plt.imshow(result, origin='lower', extent=extent, aspect='auto', cmap='inferno')
plt.colorbar(label='Z values')
plt.savefig("henon.png", dpi=300)

plt.show()
