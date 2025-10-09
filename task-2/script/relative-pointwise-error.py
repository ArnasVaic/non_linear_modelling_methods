# %%

import numpy as np
from loader import SolverConfig, load_solution
from renderer import render_video

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
  B = np.exp(-(x - x0)**2/(4 * sigma**2))
  W = np.exp(1j * w0 * x)
  M = np.exp(-1j * t)
  return (A * B * W * M).T

t = np.arange(0, cfg.T, t_stride * cfg.tau)
x = np.linspace(0, 1, cfg.total_points // x_stride)
ts, xs = np.meshgrid(t, x)

# error
err = np.abs( (u - u_exact(xs, ts)) / u)

def render_frame(step, ax):
  ax.set_ylabel("Relative pointwise error")
  ax.set_xlabel("x")
  ax.plot(x, err[step].real)
  ax.set_ylim(0, 5)

# %% Render solution video

render_video(
  '../videos/solution-error.mp4', 
  render_frame, 
  cfg.total_time_steps // t_stride,
  fps = 30,
  frame_size = (720, 640)
)
