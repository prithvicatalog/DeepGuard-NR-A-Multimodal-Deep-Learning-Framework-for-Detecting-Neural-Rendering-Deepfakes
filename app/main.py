from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routes.auth import router as auth_router
from app.api.routes.health import router as health_router
from app.api.routes.results import router as results_router
from app.api.routes.video import router as video_router
from app.core.config import settings
from app.db.database import Base, engine


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        description="DeepGuard-NR multimodal neural-rendering deepfake detection service",
        version="1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health_router, tags=["health"])
    app.include_router(auth_router, prefix="/auth", tags=["auth"])
    app.include_router(video_router, prefix="/video", tags=["video"])
    app.include_router(results_router, prefix="/results", tags=["results"])

    upload_dir = Path(settings.upload_dir)
    heatmap_dir = Path(settings.heatmap_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    heatmap_dir.mkdir(parents=True, exist_ok=True)

    storage_root = upload_dir.parent
    app.mount("/storage", StaticFiles(directory=storage_root), name="storage")

    @app.on_event("startup")
    def startup_event() -> None:
        Base.metadata.create_all(bind=engine)

    return app


app = create_app()
