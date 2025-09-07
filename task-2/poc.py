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

def f(x, t):
  return 0

x = np.linspace(0, 1, N + 1)

# [T, N], first
u = np.zeros((int(T / tau), N + 1), dtype=complex)

# some initial condition
u[0, :] = 1

# coefficient matrix (diags)
def create_coef_mat():
  upper = -0.5 * tau * 1j * (h ** -2) * np.ones(N)
  main = 1 + tau * 1j * (h ** -2) * np.ones(N + 1)
  lower = -0.5 * tau * 1j * (h ** -2) * np.ones(N)
  
  ab = np.zeros((3, N + 1), dtype=complex)   # shape = (l+u+1, N)
  ab[0,1:] = upper        # upper diag (1 above main)
  ab[1,:]  = main         # main diag
  ab[2,:-1]= lower        # lower diag (1 below main)
  return ab

m = create_coef_mat()

# storage for incremental approximations to non linear terms
u_old = np.zeros(N + 1, dtype=complex)
u_new = np.zeros(N + 1, dtype=complex)

for n in range(int(T / tau) - 1):
  
  u_old = u[n]
  
  for iter in range(MAX_ITER):
    
    # diffusion part. Rolling 
    # sus part 1, what happens when roll is around the edge, perhaps it's okay
    # because we use boundary conditions to fix values at edges but still
    # could be a source of an error
    A = 0.5 * tau * 1j * (h ** -2) * (np.roll(u_old, 1) - 2 * u_old + np.roll(u_old, -1))
    
    A[0] = 0.5 * tau * 1j * (h ** -2) * (u_old[1] - 2 * u_old[0] + u_old[1])
    
    # non linear first order part
    B = 0.5 / h * tau * beta * (
      np.abs(0.5 * (np.roll(u_old,-1) + np.roll(u[n],-1)))**2 * 0.5 * (np.roll(u_old,-1) + np.roll(u[n],-1)) -
      np.abs(0.5 * (np.roll(u_old, 1) + np.roll(u[n], 1)))**2 * 0.5 * (np.roll(u_old, 1) + np.roll(u[n], 1))
    )
    # function part
    C = 0.5 * tau * (f(x, tau * n) + f(x, tau * (n + 1)))
    
    rhs = u[n] + A + B + C
    
    nan_vals_id = np.where(np.isnan(rhs))
    inf_vals_id = np.where(np.isinf(rhs))
    
    if len(nan_vals_id[0]) > 0:
      logging.error(f"time step [{n:05d}/{int(T / tau)}] NaN values in rhs at indices {nan_vals_id}")
      raise ValueError(f"NaN values in rhs, [step={n:05d},iter={iter:05d}]")
    
    if len(inf_vals_id[0]) > 0:
      logging.error(f"time step [{n:05d}/{int(T / tau)}] Inf values in rhs at indices {inf_vals_id}")
      raise ValueError(f"NaN values in rhs, [step={n:05d},iter={iter:05d}]")
    
    u_new = solve_banded((1, 1), m, rhs)
    
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
u = np.load("solution.npy")
step = 60
plt.plot(x, u[step].real, label="Re(u)")
plt.plot(x, u[step].imag, label="Im(u)")
plt.legend()
plt.show()
    
    

  
  
