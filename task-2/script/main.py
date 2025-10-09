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

u = load_solution(cfg, "../build/data.bin")[::t_stride, ::x_stride]
x = np.linspace(0, 1, cfg.total_points // x_stride)

def render_frame(step, ax):
  ax.plot(x, u[step].real, label="$\\Re(u(x, t))$")
  ax.plot(x, u[step].imag, label="$\\Im(u(x, t))$")
  ax.set_ylim(-1, 1)
  ax.legend(loc="upper right")

# %% Render solution video

render_video(
  '../videos/solution-example.mp4', 
  render_frame, 
  cfg.total_time_steps // t_stride
)
