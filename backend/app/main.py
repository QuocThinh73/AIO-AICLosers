from fastapi import FastAPI
from contextlib import asynccontextmanager
from fastapi.staticfiles import StaticFiles
from app.api.endpoints import search, video
from app.core.config import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup

    # Running
    yield

    # Shutdown


def create_app() -> FastAPI:
    app = FastAPI(lifespan=lifespan)

    app.mount(f"{get_settings().MEDIA_URL_PREFIX}/videos",
              StaticFiles(directory=str(get_settings().MEDIA_VIDEO_DIR)), name="videos")

    app.mount(f"{get_settings().MEDIA_URL_PREFIX}/keyframes",
              StaticFiles(directory=str(get_settings().MEDIA_KEYFRAME_DIR)), name="keyframes")

    app.include_router(search.router, prefix="/api/search", tags=["search"])
    app.include_router(video.router, prefix="/api/video", tags=["video"])

    return app
