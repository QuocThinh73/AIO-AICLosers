from fastapi import APIRouter
from fastapi import Depends
from app.services.video_service import get_video_service


router = APIRouter()


@router.get("/get-video-fps")
def get_video_fps(video: str, video_service=Depends(get_video_service)):
    return video_service.get_video_fps(video)
