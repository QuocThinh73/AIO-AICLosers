import json
from app.core.config import get_settings
from functools import lru_cache


class VideoService:
    def __init__(self):
        self.fps = self._load_fps(get_settings().FPS_PATH)

    def _load_fps(self, fps_path):
        with open(fps_path, "r") as f:
            fps = json.load(f)
        return {item["video"]: item["fps"] for item in fps}

    def get_video_fps(self, video):
        video += ".mp4"
        return self.fps[video]


@lru_cache()
def get_video_service():
    return VideoService()
