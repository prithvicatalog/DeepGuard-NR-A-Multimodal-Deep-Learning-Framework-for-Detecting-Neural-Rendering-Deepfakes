from datetime import datetime, timezone

from sqlalchemy import JSON, Column, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from app.db.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(64), unique=True, nullable=False, index=True)
    email = Column(String(128), unique=True, nullable=False, index=True)
    password_hash = Column(String(256), nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    videos = relationship("Video", back_populates="owner", cascade="all, delete-orphan")


class Video(Base):
    __tablename__ = "videos"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    filename = Column(String(256), nullable=False)
    stored_path = Column(String(512), nullable=False)
    status = Column(String(32), default="uploaded", nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    owner = relationship("User", back_populates="videos")
    result = relationship("AnalysisResult", back_populates="video", uselist=False, cascade="all, delete-orphan")


class AnalysisResult(Base):
    __tablename__ = "analysis_results"

    id = Column(Integer, primary_key=True, index=True)
    video_id = Column(Integer, ForeignKey("videos.id"), unique=True, nullable=False, index=True)
    verdict = Column(String(8), nullable=False)
    confidence = Column(Float, nullable=False)
    artifact_score = Column(Float, nullable=False)
    depth_score = Column(Float, nullable=False)
    temporal_score = Column(Float, nullable=False)
    lighting_score = Column(Float, nullable=True)
    heatmap_path = Column(String(512), nullable=True)
    heatmap_paths = Column(JSON, nullable=True)
    module_summary = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    video = relationship("Video", back_populates="result")
