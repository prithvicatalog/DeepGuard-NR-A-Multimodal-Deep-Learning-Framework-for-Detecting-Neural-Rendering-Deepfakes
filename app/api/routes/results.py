from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.db.database import get_db
from app.db.models import AnalysisResult, User, Video
from app.db.schemas import ResultResponse

router = APIRouter()


def _to_url(request: Request, path: str) -> str:
    relative = path.replace("\\", "/").split("storage/")[-1]
    return f"{request.base_url}storage/{relative}".replace("///", "//")


@router.get("/{video_id}", response_model=ResultResponse)
def get_result(
    video_id: int,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    video = db.query(Video).filter(Video.id == video_id, Video.user_id == current_user.id).first()
    if not video:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Video not found")

    result = db.query(AnalysisResult).filter(AnalysisResult.video_id == video_id).first()
    if not result:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Result not available")

    heatmap_paths = result.heatmap_paths or []
    heatmap_urls = [_to_url(request, p) for p in heatmap_paths]

    return ResultResponse(
        video_id=video.id,
        filename=video.filename,
        verdict=result.verdict,
        confidence=result.confidence,
        heatmap_url=heatmap_urls[0] if heatmap_urls else None,
        heatmap_urls=heatmap_urls,
        module_summary=result.module_summary,
        created_at=result.created_at,
    )
