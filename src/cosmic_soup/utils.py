import numpy as np
import PIL.Image
import jax
import subprocess
import time

def np2pil(a: np.ndarray) -> PIL.Image.Image:
    """Converts a NumPy array to a PIL Image.

    If the array is float, it's clipped to [0, 1] and scaled to [0, 255].
    """
    if a.dtype in [np.float32, np.float64]:
        a = np.uint8(np.clip(a, 0, 1) * 255)
    return PIL.Image.fromarray(a)

def vmap2(f):
    """Applies jax.vmap twice to the function f."""
    return jax.vmap(jax.vmap(f))

class VideoWriter:
    """
    A simplified class to write video frames using ffmpeg.
    Removes IPython/widget specific display code from the notebook version.
    """
    def __init__(self, filename: str = '_tmp.mp4', fps: float = 30.0, **kwargs):
        self.ffmpeg = None
        self.filename = filename
        self.fps = fps
        # Removed: self.view = widgets.Output()
        # Removed: self.last_preview_time = 0.0
        self.frame_count = 0
        # Removed: self.show_on_finish = show_on_finish
        # Removed: display(self.view)

    def add(self, img: np.ndarray):
        """Adds a frame to the video."""
        img_array = np.asarray(img) # Ensure it's a numpy array
        h, w = img_array.shape[:2]
        if self.ffmpeg is None:
            self.ffmpeg = self._open(w, h)
        
        if img_array.dtype in [np.float32, np.float64]:
            img_array = np.uint8(img_array.clip(0, 1) * 255)
        if len(img_array.shape) == 2: # Grayscale
            img_array = np.repeat(img_array[..., None], 3, -1) # Convert to RGB
        
        self.ffmpeg.stdin.write(img_array.tobytes())
        self.frame_count += 1
        # Removed: IPython/widget preview logic

    def __call__(self, img: np.ndarray):
        return self.add(img)

    def _open(self, w: int, h: int):
        cmd = [
            'ffmpeg', '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{w}x{h}',
            '-pix_fmt', 'rgb24', # Assuming RGB input after potential conversion
            '-r', str(self.fps),
            '-i', '-',
            '-pix_fmt', 'yuv420p', # Common output format
            '-c:v', 'libx264',
            '-crf', '20', # Adjust quality (lower is better quality, larger file)
            self.filename
        ]
        return subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    def close(self):
        """Closes the ffmpeg process."""
        if self.ffmpeg:
            if self.ffmpeg.stdin:
                self.ffmpeg.stdin.close()
            self.ffmpeg.wait()
            if self.ffmpeg.returncode != 0:
                # Capture stderr for debugging
                error = self.ffmpeg.stderr.read().decode()
                print(f"FFmpeg error for {self.filename}:\n{error}")
            self.ffmpeg = None
        # Removed: self.view specific close logic

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        # Removed: self.show() which relied on IPython 