"""
Authentication service — login, token validation, current-user dependency.
"""
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.security import decode_access_token, verify_password
from app.models.operator import Operator

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")


async def authenticate_operator(
    db: AsyncSession, email: str, password: str
) -> Operator | None:
    result = await db.execute(select(Operator).where(Operator.email == email))
    op = result.scalar_one_or_none()
    if op is None or not verify_password(password, op.hashed_password):
        return None
    return op


async def get_current_operator(
    token: str = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db),
) -> Operator:
    credentials_exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    payload = decode_access_token(token)
    if payload is None:
        raise credentials_exc

    operator_id: str | None = payload.get("sub")
    if operator_id is None:
        raise credentials_exc

    result = await db.execute(
        select(Operator).where(Operator.id == operator_id, Operator.is_active == True)  # noqa: E712
    )
    op = result.scalar_one_or_none()
    if op is None:
        raise credentials_exc
    return op
