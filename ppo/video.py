from pathlib import Path

import cv2
import imageio
import numpy as np


class VideoRecorder:
    def __init__(self, root_dir, render_size=256, fps=20):
        self.save_dir = Path(root_dir) / "eval_video" if root_dir is not None else None
        if self.save_dir is not None:
            self.save_dir.mkdir(parents=True, exist_ok=True)

        self.render_size = render_size
        self.fps = fps
        self.frames = []
        self.enabled = False

    def init(self, env, enabled=True):
        self.frames = []
        self.enabled = self.save_dir is not None and enabled
        if self.enabled:
            try:
                self.record(env)  # Try to record first frame immediately
            except Exception as e:
                print(f"[WARNING] Failed to render first frame: {e}")
                self.enabled = False

    def record(self, env):
        if not self.enabled:
            return

        try:
            if hasattr(env, "physics"):  # For DeepMind Control Suite
                frame = env.physics.render(
                    height=self.render_size, width=self.render_size, camera_id=0
                )
            else:
                frame = env.render()  # Expecting render_mode="rgb_array"
                if frame is None:
                    raise ValueError("env.render() returned None. Did you set render_mode='rgb_array'?")

            frame = cv2.resize(frame, (self.render_size, self.render_size), interpolation=cv2.INTER_CUBIC)
            self.frames.append(frame)

        except Exception as e:
            print(f"[WARNING] VideoRecorder.record failed: {e}")
            self.enabled = False

    def save(self, file_name):
        if self.enabled and self.frames:
            path = self.save_dir / file_name
            imageio.mimsave(str(path), self.frames, fps=self.fps)
            print(f"[INFO] Video saved to {path}")


class TrainVideoRecorder:
    def __init__(self, root_dir, render_size=256, fps=20):
        self.save_dir = Path(root_dir) / "train_video" if root_dir is not None else None
        if self.save_dir is not None:
            self.save_dir.mkdir(parents=True, exist_ok=True)

        self.render_size = render_size
        self.fps = fps
        self.frames = []
        self.enabled = False

    def init(self, obs, enabled=True):
        self.frames = []
        self.enabled = self.save_dir is not None and enabled
        self.record(obs)

    def record(self, obs):
        if self.enabled:
            frame = cv2.resize(
                obs[-3:].transpose(1, 2, 0),
                dsize=(self.render_size, self.render_size),
                interpolation=cv2.INTER_CUBIC,
            )
            self.frames.append(frame)

    def save(self, file_name):
        if self.enabled and self.frames:
            path = self.save_dir / file_name
            imageio.mimsave(str(path), self.frames, fps=self.fps)
            print(f"[INFO] Training video saved to {path}")
