from datetime import timedelta

from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.security import create_access_token, get_password_hash, verify_password
from app.db.models import User
from app.db.schemas import TokenResponse, UserRegisterRequest


def register_user(db: Session, payload: UserRegisterRequest) -> TokenResponse:
    existing = db.query(User).filter((User.username == payload.username) | (User.email == payload.email)).first()
    if existing:
        raise ValueError("User with username/email already exists")

    user = User(
        username=payload.username,
        email=payload.email,
        password_hash=get_password_hash(payload.password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    token = create_access_token(user.username, timedelta(minutes=settings.access_token_expire_minutes))
    return TokenResponse(access_token=token)


def authenticate_user(db: Session, username: str, password: str) -> str | None:
    user = db.query(User).filter(User.username == username).first()
    if not user or not verify_password(password, user.password_hash):
        return None
    return create_access_token(user.username)
