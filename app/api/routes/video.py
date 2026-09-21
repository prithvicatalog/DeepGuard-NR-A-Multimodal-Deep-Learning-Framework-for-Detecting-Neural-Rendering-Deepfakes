from pathlib import Path
import traceback

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.db.database import get_db
from app.db.database import SessionLocal
from app.db.models import AnalysisResult, User, Video
from app.db.schemas import AnalyzeRequest, AnalyzeResponse, UploadResponse
from app.services.pipeline_service import analyze_video_pipeline
from app.services.video_service import save_upload_file, validate_video_extension

router = APIRouter()


@router.post("/upload", response_model=UploadResponse, status_code=status.HTTP_201_CREATED)
def upload_video(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if not validate_video_extension(file.filename):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Unsupported file format")

    stored_path = save_upload_file(file)
    video = Video(user_id=current_user.id, filename=file.filename or Path(stored_path).name, stored_path=stored_path)
    db.add(video)
    db.commit()
    db.refresh(video)
    return UploadResponse(video_id=video.id, filename=video.filename, status=video.status)


def _run_analysis_job(video_id: int) -> None:
    db = SessionLocal()
    try:
        video = db.query(Video).filter(Video.id == video_id).first()
        if not video:
            return
        try:
            analysis_data = analyze_video_pipeline(video.stored_path)
        except Exception:
            traceback.print_exc()
            video.status = "failed"
            db.commit()
            return

        result = db.query(AnalysisResult).filter(AnalysisResult.video_id == video_id).first()
        if not result:
            result = AnalysisResult(video_id=video_id, **analysis_data)
            db.add(result)
        else:
            for key, value in analysis_data.items():
                setattr(result, key, value)

        video.status = "completed"
        db.commit()
    finally:
        db.close()


@router.post("/analyze", response_model=AnalyzeResponse)
def analyze_video(
    payload: AnalyzeRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    video = db.query(Video).filter(Video.id == payload.video_id, Video.user_id == current_user.id).first()
    if not video:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Video not found")

    video.status = "processing"
    db.commit()

    background_tasks.add_task(_run_analysis_job, video.id)

    return AnalyzeResponse(video_id=video.id, status=video.status)
