from datetime import datetime
from typing import Dict, Literal

from pydantic import BaseModel, EmailStr, Field


class UserRegisterRequest(BaseModel):
    username: str = Field(min_length=3, max_length=64)
    email: EmailStr
    password: str = Field(min_length=6, max_length=128)


class UserLoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UploadResponse(BaseModel):
    video_id: int
    filename: str
    status: str


class AnalyzeRequest(BaseModel):
    video_id: int


class AnalyzeResponse(BaseModel):
    video_id: int
    status: str
    verdict: Literal["REAL", "FAKE"] | None = None
    confidence: float | None = None


class ResultResponse(BaseModel):
    video_id: int
    filename: str
    verdict: Literal["REAL", "FAKE"]
    confidence: float
    heatmap_url: str | None = None
    heatmap_urls: list[str] = []
    module_summary: Dict[str, float]
    created_at: datetime
