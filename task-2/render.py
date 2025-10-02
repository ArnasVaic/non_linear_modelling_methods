# %% Load data

import numpy as np
import matplotlib.pyplot as plt
import cv2
from io import BytesIO
from PIL import Image
from tqdm import tqdm

# Parameters must match your C++ code
N = 999
total_points = N + 1
T = 1
tau = 0.0001
total_time_steps = int(T / tau)

# Read binary file as complex128 (same layout as std::complex<double>)
u = np.fromfile("build/data.bin", dtype=np.complex128)

if u.size % total_points == total_time_steps:
    u = np.reshape(u, (total_time_steps, total_points))
else:
    u =  np.reshape(u, (u.size // total_points, total_points))

print(u.shape)

x = np.linspace(0, 1, total_points)

# %% Test, render single frame

step = 0
plt.plot(x, u[step].real, label="Re(u)")
plt.plot(x, u[step].imag, label="Im(u)")

# %% render video
# Parameters
max_frames = 250
stride = 1
n_frames = min(u.shape[0], max_frames) // stride
fps = 30
out_file = "gauss_wavepacket_03.mp4"

# Create a VideoWriter
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec
frame_size = (640, 480)
video = cv2.VideoWriter(out_file, fourcc, fps, frame_size)

u_strided = u[::stride, :]

# Generate frames
for i in tqdm(range(n_frames)):
    
    # Plot
    fig, ax = plt.subplots(figsize=(6.4, 4.8))  # size matches 640x480
    
    ax.set_title('$u(x, 0) = A\\exp\\left(-\\frac{(x - x_0)^2}{2\\sigma^2}\\right)e^{ix}, f(x, t) = 0, \\beta = 1$')
    ax.plot(x, u_strided[i].real, label="Re(u)")
    ax.plot(x, u_strided[i].imag, label="Im(u)")
    
    ax.set_ylim(-1.2, 1.2)
    
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

# %% Image

plt.imshow(np.abs(u[:250, :]))