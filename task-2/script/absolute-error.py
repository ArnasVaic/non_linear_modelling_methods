# %%

import numpy as np
from loader import SolverConfig, load_solution
import matplotlib.pyplot as plt

cfg = SolverConfig( 
  N = 999,
  T = 10,
  tau = 0.001
)
t_stride, x_stride = 25, 5

# numerical solution
u = load_solution(cfg, "../build/data.bin")[::t_stride, ::x_stride]

# exact solution
A, w0, sigma, x0 = 0.1, 100.0, 0.09, 0.5

def u_exact(x, t):
  B = np.exp(-(x - x0) * (x - x0)/(4.0 * sigma * sigma))
  W = np.exp(1.0j * w0 * x)
  M = np.exp(-1.0j * t)
  return (A * B * W * M).T

t = np.arange(0, cfg.T, t_stride * cfg.tau)
x = np.linspace(0, 1, cfg.total_points // x_stride)
ts, xs = np.meshgrid(t, x)

# error
u_true = u_exact(xs, ts)
err = np.abs(u - u_true)

def render_frame(step, ax):
  ax.set_ylabel("Absolute pointwise error")
  ax.set_xlabel("x")
  ax.plot(x, err[step].real)
  ax.set_ylim(-0.05, 0.05)

# Preview initial error (should be 0)
fig, ax = plt.subplots(figsize=(6,4))
render_frame(0, ax)
fig.show()

# %% Render solution video

from renderer import render_video

render_video(
  '../videos/solution-absolute-error.mp4', 
  render_frame, 
  cfg.total_time_steps // t_stride
)
