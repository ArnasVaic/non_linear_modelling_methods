# %% Utils

from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import cv2
from io import BytesIO
from PIL import Image
from tqdm import tqdm

@dataclass
class SolverParams:
    N: int # number of points -1
    T: float # total sim time
    tau: float # time step

def load_data(params: SolverParams, path):

    total_points = params.N + 1
    T = params.T
    total_time_steps = int(T / params.tau)

    # Read binary file as complex128
    # (same layout as std::complex<double>)
    u = np.fromfile(path, dtype=np.complex128)

    if u.size % total_points == total_time_steps:
        u = np.reshape(u, (total_time_steps, total_points))
    else:
        u =  np.reshape(u, (u.size // total_points, total_points))

    print(u.shape)
    return u


def render_video(
    path,
    solution,
    params,
    stride = 1, 
    max_frames = 250, 
    lims = [ -1.2, 1.2]
    ):

    total_points = params.N + 1
    x = np.linspace(0, 1, total_points)

    n_frames = min(u.shape[0], max_frames) // stride
    fps = 30
    out_file = path

    # Create a VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec
    frame_size = (640, 480)
    video = cv2.VideoWriter(out_file, fourcc, fps, frame_size)

    u_strided = solution[::stride, :]

    # Generate frames
    for i in tqdm(range(n_frames)):
        
        # Plot
        fig, ax = plt.subplots(figsize=(6.4, 4.8))  # size matches 640x480
        
        # ax.set_title('$u(x, 0) = A\\exp\\left(-\\frac{(x - x_0)^2}{2\\sigma^2}\\right)e^{ix}, f(x, t) = 0, \\beta = 1$')
        ax.plot(x, u_strided[i].real, label="$\\Re(u(x, t))$")
        ax.plot(x, u_strided[i].imag, label="$\\Im(u(x, t))$")
        
        ax.set_ylim(*lims)
        ax.legend()

        # Save plot to a buffer
        buf = BytesIO()
        plt.savefig(buf, format="png")
        plt.close(fig)
        
        # Convert buffer to image
        buf.seek(0)

        img = Image.open(buf).convert("RGB")

        # Ensure correct size
        img = img.resize(frame_size)  

        # Convert RGB → BGR for OpenCV
        frame = np.array(img)[:, :, ::-1]  
        
        

        # Write frame
        video.write(frame)

    # Release video
    video.release()
    print(f"Video saved as {out_file}")

# testai:
# - test uzduotis nepadare klaidu? sasiuviny 10 kartu sprendziau
# - algoritma pertvarkėm į kompaktišką ? no idea, im using lapack 
# - įsitikinti ... erroras mazas ant closed solution, sprendinys stabilus ant sudetingesnio

# %% Load data

params = SolverParams( 
    N=19,
    T=0.03,
    tau=0.0001
)

u = load_data(params, "build/test-01-smallest.bin")

# %%

plt.plot(np.real(u[1]))
plt.plot(np.imag(u[1]))

# %% Render video
render_video('test-01-small.mp4', u, params, stride=1, max_frames=300)


# %% Difference from real solution

def sol(x, t):
    return np.cos(4 * np.pi * x) * ( np.cos(16 * np.pi **2 * t) - 1j * np.sin(16 * np.pi ** 2 * t))

t = np.arange(0, params.T, params.tau)
x = np.linspace(0, 1, params.N + 1)

ts, xs = np.meshgrid(t, x)

u_error = np.abs(u - sol(xs, ts).T)

render_video('test-01-smallest-error.mp4', u_error, params, 1, 300, lims=[-1, 1.0])

# %% Differences from test 2

import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
import cv2
from tqdm import tqdm
from scipy.interpolate import interp1d

def sol(x, t):
    return np.cos(4 * np.pi * x) * ( np.cos(16 * np.pi **2 * t) - 1j * np.sin(16 * np.pi ** 2 * t))

t = np.arange(0, params.T, params.tau)
x = np.linspace(0, 1, params.N + 1)

ts, xs = np.meshgrid(t, x)

u_true = sol(xs, ts).T # we expect arrays of shape [T, N + 1]

u_high = load_data(SolverParams(999, 0.03, 0.0001), 'build/test-01.bin')

u_mid = load_data(SolverParams(99, 0.03, 0.0001), 'build/test-01-small.bin')

u_small = load_data(SolverParams(19, 0.03, 0.0001), 'build/test-01-smallest.bin')

def generate_error_video(u_true, xs_true, u_approxs, xs_list, labels, path="error_video.mp4", stride=1, max_frames=300):
    """
    u_true : [T, N+1] array
    xs_true : x-grid for u_true
    u_approxs : list of [T, Ni+1] arrays (different resolutions)
    xs_list : list of x-grids corresponding to u_approxs
    labels : list of strings
    """
    # Interpolate approximations onto true grid
    u_approxs_interp = []
    for u, xs in zip(u_approxs, xs_list):
        T = u.shape[0]
        u_interp = np.zeros((T, len(xs_true)))
        for t in range(T):
            f = interp1d(xs, u[t], kind='cubic', fill_value="extrapolate")
            u_interp[t] = f(xs_true)
        u_approxs_interp.append(u_interp)

    n_frames = min(u_true.shape[0], max_frames) // stride
    fps = 30
    frame_size = (640, 480)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(path, fourcc, fps, frame_size)

    u_true_strided = u_true[::stride, :]
    u_approxs_strided = [u[::stride, :] for u in u_approxs_interp]

    # Determine consistent y-limits
    all_errors = [np.abs(u_true - u) for u in u_approxs_interp]
    y_max = max(np.max(e) for e in all_errors)

    for i in tqdm(range(n_frames), desc="Generating error video"):
        fig, ax = plt.subplots(figsize=(6.4, 4.8))

        for u_approx, label in zip(u_approxs_strided, labels):
            error = np.abs(u_true_strided[i] - u_approx[i])
            ax.plot(xs_true, error, label=label)

        ax.set_ylim(0, y_max)
        ax.set_xlabel("x")
        ax.set_ylabel("|u_true - u_approx|")
        ax.set_title(f"Pointwise error at time step {i*stride}")
        ax.legend()
        ax.grid(True)

        buf = BytesIO()
        plt.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)

        img = Image.open(buf).convert("RGB")
        img = img.resize(frame_size)
        frame = np.array(img)[:, :, ::-1]
        video.write(frame)

    video.release()
    print(f"Error video saved as {path}")

# --- Example usage ---
xs_true = np.linspace(0, 1, u_true.shape[1])
xs_high = np.linspace(0, 1, u_high.shape[1])
xs_mid = np.linspace(0, 1, u_mid.shape[1])
xs_small = np.linspace(0, 1, u_small.shape[1])

generate_error_video(
    u_true,
    xs_true,
    u_approxs=[u_high, u_mid, u_small],
    xs_list=[xs_high, xs_mid, xs_small],
    labels=["High res (N=999)", "Medium res (N=99)", "Small res (N=19)"],
    path="error_video_interp.mp4",
    stride=1,
    max_frames=300
)

