import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

def render_video(path, frame_func, n_frames, fps=30):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # Create figure once
    fig, ax = plt.subplots()
    fig.tight_layout(pad=5)
    
    # Pre-draw once
    frame_func(0, ax)
    fig.canvas.draw()

    # Dynamically get figure pixel size
    width, height = fig.canvas.get_width_height()
    frame_size = (width, height)
    video = cv2.VideoWriter(path, fourcc, fps, frame_size)

    for i in tqdm(range(n_frames), desc="Rendering video"):
        ax.clear()
        frame_func(i, ax)
        fig.canvas.draw()

        img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        video.write(frame)

    video.release()
    plt.close(fig)
    print(f"✅ Video saved as {path} ({frame_size[0]}x{frame_size[1]} px)")
