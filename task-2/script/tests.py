# %% imports
import numpy as np
import matplotlib.pyplot as plt

# %% functions
beta = 1.0
A, w0, sigma, x0 = 0.1, 2.0, 0.09, 0.5

def B(x):
  return np.exp(-(x - x0)**2/(4 * sigma**2))

def W(x):
  return np.exp(1j * w0 * x)

def M(t):
  return np.exp(-1j * t)

def u(x, t):
  return A * B(x) * W(x) * M(t)

def Dx_B(x):
  return - 1.0 / (2.0 * sigma **2) * B(x) * (x - x0)

def Dxx_B(x):
  return - 1.0 / (2.0 * sigma **2) * (Dx_B(x) * (x - x0) + B(x))

def Dx_W(x):
  return 1.0j * w0 * W(x)

def Dxx_W(x):
  return - w0 * w0 * W(x)

def Dt_u(x, t):
  return -1.0j * u(x, t)

def Dxx_u(x, t):
  return A * M(t) * (Dxx_B(x) * W(x) + 2.0 * Dx_B(x) * Dx_W(x) + B(x) * Dxx_W(x))

def Dx_u2u(x, t):
  return A * A * A * M(t) * ( 3 * B(x) * B(x) * Dx_B(x) * W(x) + B(x) * B(x) * B(x) * Dx_W(x));

def f(x, t):
  return Dt_u(x, t) - 1.0j * Dxx_u(x, t) - beta * Dx_u2u(x, t)

def LHS(x, t, tau):
  return ( u(x, t + tau) - u(x, t) ) / tau

def RHS(x, t, h, tau):
  p1_sum = (u(x + h, t + tau) + u(x + h, t)) / 2.0
  p2_sum = (u(x - h, t + tau) + u(x - h, t)) / 2.0

  rhs_A = \
  0.5j / h**2 * ( \
    u(x + h, t + tau) - 2.0 * u(x, t + tau) + u(x - h, t + tau) + \
    u(x + h, t      ) - 2.0 * u(x, t      ) + u(x - h, t      )   \
  )
  
  rhs_B = beta / (2.0 * h) * (np.abs(p1_sum)**2 * p1_sum - np.abs(p2_sum)**2 * p2_sum)

  rhs_C = ( f(x, t + tau) + f(x, t) ) / 2.0

  return rhs_A + rhs_B + rhs_C

def T1(x, t, h, tau):
  return np.abs(LHS(x, t, tau) - RHS(x, t, h, tau))

def T2(x, t, h, tau):
  lhs = u(x + h, t + tau) + (2j / tau * h**2 - 2) * u(x, t + tau) + u(x - h, t + tau)

  p1_sum = (u(x + h, t + tau) + u(x + h, t)) / 2.0
  p2_sum = (u(x - h, t + tau) + u(x - h, t)) / 2.0

  rhs_A = 2j / tau * h**2 * u(x, t) 

  rhs_B = u(x + h, t) - 2.0 * u(x, t) + u(x - h, t)

  rhs_C = 1j * beta * h * (np.abs(p1_sum)**2 * p1_sum - np.abs(p2_sum)**2 * p2_sum)

  rhs_D = 1j * h**2 * ( f(x, t + tau) + f(x, t) )

  rhs = rhs_A - rhs_B + rhs_C + rhs_D
  return np.abs(lhs - rhs)


# %% plot exact solution

t = 20.3
x = np.linspace(0, 1, 200, dtype=np.float128)
plt.plot(x, np.real(u(x, t)))
plt.plot(x, np.abs(u(x, t)))

# %% plot T1

x, t = 0.32, 1.25

N = 10
a0 = 0.01

configs = [ (a0 / (10 ** i), a0 / (10 ** i)) for i in range(N) ]

loss = [ T1( np.float128(x),  np.float128(t),  np.float128(h),  np.float128(tau)) for h, tau in configs ]

plt.plot([c[0] for c in configs], loss)
plt.xlabel("step size ($h, \\tau$)")
plt.ylabel("T1")
plt.xscale('log')  # logarithmic x-axis
plt.yscale('log')  # logarithmic y-axis
plt.show()

# %% T2

x, t = 0.32, 1.25

N = 4
a0 = 0.01

configs = [ (a0 / (10 ** i), a0 / (10 ** i)) for i in range(N) ]

loss = [ T2( np.float128(x),  np.float128(t),  np.float128(h),  np.float128(tau)) for h, tau in configs ]

plt.plot([c[0] for c in configs], loss)
plt.xlabel("step size ($h, \\tau$)")
plt.ylabel("T1")
plt.xscale('log')  # logarithmic x-axis
plt.yscale('log')  # logarithmic y-axis
plt.show()


