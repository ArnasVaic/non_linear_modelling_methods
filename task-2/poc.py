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
N = 9
T = 1

beta = 0.1
delta = 10e-5
tau = 0.01
h = 1 / N

total_time_steps = int(T / tau)

def f(x, t):
  return 0

x = np.linspace(0, 1, N + 1)

# [T, N], first
u = np.zeros((int(T / tau), N + 1), dtype=complex)

# some initial condition
u[0, :] = np.cos(2 * np.pi * x) + 1j * np.sin(2 * np.pi * x)

# coefficient matrix (diags)
def create_coef_mat():
  # upper = -0.5 * tau * 1j * (h ** -2) * np.ones(N)
  # main = 1 + tau * 1j * (h ** -2) * np.ones(N + 1)
  # lower = -0.5 * tau * 1j * (h ** -2) * np.ones(N)
  
  # ab = np.zeros((3, N + 1), dtype=complex)   # shape = (l+u+1, N)
  # ab[0,1:] = upper        # upper diag (1 above main)
  # ab[1,:]  = main         # main diag
  # ab[2,:-1]= lower        # lower diag (1 below main)
  
  upper = np.ones(N)
  main = -2 * (1 - 1j * h**2) * np.ones(N + 1)
  lower = np.ones(N)
  
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
    A = 2j * (h**2) * u[n]
    
    # diffusion part, previous time step solutions
    B = - tau * (np.roll(u[n], 1) - 2 * u[n] + np.roll(u[n], -1))
     
    # non linear first order part
    C = 2j * (h**2) * beta * tau * (
      np.abs(0.5 * (np.roll(u_old,-1) + np.roll(u[n],-1)))**2 * 0.5 * (np.roll(u_old,-1) + np.roll(u[n],-1)) -
      np.abs(0.5 * (np.roll(u_old, 1) + np.roll(u[n], 1)))**2 * 0.5 * (np.roll(u_old, 1) + np.roll(u[n], 1))
    )
    # function part
    D = 1j * tau * (h**2) * (f(x, tau * n) + f(x, tau * (n + 1)))
    
    rhs = u[n] + A + B + C + D
    
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

# %%
#u = np.load("solution.npy")
step = 3
plt.plot(x, u[step].real, label="Re(u)")
plt.plot(x, u[step].imag, label="Im(u)")
plt.legend()
plt.show()
    
    

  
  
