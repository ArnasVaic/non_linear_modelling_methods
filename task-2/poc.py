# %%

import logging
import numpy as np
from scipy.signal import convolve2d
from scipy.linalg import solve_banded
from numpy.linalg import norm
import matplotlib.pyplot as plt

# D Netiesinė išvestinių atžvilgiu Šriodingerio lygtis

logging.basicConfig(
  filename='debug.log',
  filemode='w',
  format='%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s',
  datefmt='%Y-%m-%d %H:%M:%S',
  level=logging.DEBUG)

MAX_ITER = 100
N = 999
T = 1

beta = 0.1
delta = 10e-5
tau = 0.0001
h = 1 / N

total_time_steps = int(T / tau)

def f(x, t):
  return 0

x = np.linspace(0, 1, N + 1)

# [T, N], first
u = np.zeros((int(T / tau), N + 1), dtype=complex)

# some initial condition
phi = np.pi/4                 # constant phase
u[0] = np.cos(2*np.pi*x) * np.exp(1j * phi)

def laplacian_neumann(u):
  left  = np.empty_like(u)
  right = np.empty_like(u)

  left[1:]  = u[:-1]
  right[:-1] = u[1:]

  # ghost points (Neumann: derivative zero -> mirror the neighbour)
  left[0]   = u[1]     # u_{-1} := u_1  => left neighbour for index 0 is u1
  right[-1] = u[-2]    # u_{N+1} := u_{N-1} => right neighbour for last index is u_{N-1}

  return left - 2*u + right

# coefficient matrix (diags)
def create_coef_mat():
  upper = np.ones(N)
  main = 2 * (1j / tau * h**2 - 1) * np.ones(N + 1)
  lower = np.ones(N)
  
  main[ 0] = (2j / tau * h**2 - 1)
  main[-1] = (2j / tau * h**2 - 1)

  ab = np.zeros((3, N + 1), dtype=complex)
  ab[0,1:] = upper
  ab[1,:] = main
  ab[2,:-1] = lower
  
  return ab

m = create_coef_mat()

# storage for incremental approximations to non linear terms
u_old = np.zeros(N + 1, dtype=complex)
u_new = np.zeros(N + 1, dtype=complex)

# each iteration we calculate the value of the solution
# at time step n+1 so the loop goes from 0 to total_time_steps-1
for n in range(total_time_steps - 1):
  
  u_old = u[n]
  
  for iter in range(MAX_ITER):
    
    # previous step from time derivative approximation
    A = 2j * (h**2) / tau * u[n] 
    
    # diffusion part, previous time step solutions
    B = -laplacian_neumann(u[n])

    # non linear first order part
    C = 1j * h * beta * (
      np.abs(0.5 * (np.roll(u_old,-1) + np.roll(u[n],-1)))**2 * 0.5 * (np.roll(u_old,-1) + np.roll(u[n],-1)) -
      np.abs(0.5 * (np.roll(u_old, 1) + np.roll(u[n], 1)))**2 * 0.5 * (np.roll(u_old, 1) + np.roll(u[n], 1))
    )
    # function part
    D = 1j * (h**2) * (f(x, tau * n) + f(x, tau * (n + 1)))
    
    rhs = A + B + C + D
    
    nan_vals_id = np.where(np.isnan(rhs))
    inf_vals_id = np.where(np.isinf(rhs))
    
    if len(nan_vals_id[0]) > 0:
      logging.error(f"time step [{n:05d}/{int(T / tau)}] NaN values in rhs at indices {nan_vals_id}")
      raise ValueError(f"NaN values in rhs, [step={n:05d},iter={iter:05d}]")
    
    if len(inf_vals_id[0]) > 0:
      logging.error(f"time step [{n:05d}/{int(T / tau)}] Inf values in rhs at indices {inf_vals_id}")
      raise ValueError(f"NaN values in rhs, [step={n:05d},iter={iter:05d}]")
    
    u_new[:] = solve_banded((1, 1), m, rhs)
    
    # enforce boundary conditions
    u_new[0] = u_new[1]
    u_new[-1] = u_new[-2]
    
    # check convergence
    norm_val = norm(u_new - u_old, np.inf)
    
    logging.info(f"time step [{n:05d}/{int(T / tau)}] Iteration {iter}, norm = {norm_val}")
    
    if norm_val < delta:
      logging.info(f"time step [{n:05d}/{int(T / tau)}] Converged at iteration {iter}")
      if iter == MAX_ITER - 1:
        logging.warning(f"time step [{n:05d}/{int(T / tau)}] Reached max iterations {MAX_ITER}")
      break
    
    # assign the old value
    u_old = np.copy(u_new)
    
  u[n + 1, :] = u_new

np.save("solution.npy", u)


# %%

# Plot solution quantity over time
# u = np.load("solution.npy")

print(u.shape)

step = 1000
plt.plot(x, u[step].real, label="Re(u)")
plt.plot(x, u[step].imag, label="Im(u)")
plt.legend()
plt.show()


# %% Render video

import numpy as np
import matplotlib.pyplot as plt
import cv2
from io import BytesIO
from PIL import Image
from tqdm import tqdm

# Parameters
n_frames = 1000
fps = 60
out_file = "plot_video.mp4"

# Create a VideoWriter
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec
frame_size = (640, 480)
video = cv2.VideoWriter(out_file, fourcc, fps, frame_size)

# Generate frames
for i in tqdm(range(n_frames)):
    
    # Plot
    fig, ax = plt.subplots(figsize=(6.4, 4.8))  # size matches 640x480
    
    ax.plot(x, u[i].real, label="Re(u)")
    ax.plot(x, u[i].imag, label="Im(u)")
    
    ax.set_ylim(-1, 1)
    
    # Save plot to a buffer
    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    
    # Convert buffer to image
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    img = img.resize(frame_size)  # Ensure correct size
    frame = np.array(img)[:, :, ::-1]  # Convert RGB → BGR for OpenCV
    
    # Write frame
    video.write(frame)

# Release video
video.release()
print(f"Video saved as {out_file}")