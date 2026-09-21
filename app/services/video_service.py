import shutil
import uuid
from pathlib import Path

from fastapi import UploadFile

from app.core.config import settings


def validate_video_extension(filename: str | None) -> bool:
    if not filename:
        return False
    suffix = Path(filename).suffix.lower()
    return suffix in settings.supported_formats


def save_upload_file(file: UploadFile) -> str:
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    extension = Path(file.filename or "").suffix.lower()
    filename = f"{uuid.uuid4().hex}{extension}"
    destination = upload_dir / filename
    with destination.open("wb") as out:
        shutil.copyfileobj(file.file, out)
    return str(destination)
