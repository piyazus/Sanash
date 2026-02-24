"""
Authentication endpoints: login, register.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.security import create_access_token, hash_password
from app.models.operator import Operator
from app.services.auth import authenticate_operator, get_current_operator

router = APIRouter(prefix="/auth", tags=["auth"])


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    operator_id: str
    name: str
    company: str


class RegisterRequest(BaseModel):
    name: str
    email: EmailStr
    password: str
    company: str


class OperatorOut(BaseModel):
    id: str
    name: str
    email: str
    company: str
    is_active: bool

    model_config = {"from_attributes": True}


@router.post("/login", response_model=TokenResponse)
async def login(
    form: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db),
):
    op = await authenticate_operator(db, form.username, form.password)
    if op is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = create_access_token({"sub": str(op.id)})
    return TokenResponse(
        access_token=token,
        operator_id=str(op.id),
        name=op.name,
        company=op.company,
    )


@router.post("/register", response_model=TokenResponse, status_code=201)
async def register(body: RegisterRequest, db: AsyncSession = Depends(get_db)):
    existing = await db.execute(
        select(Operator).where(Operator.email == body.email)
    )
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="Email already registered")

    op = Operator(
        name=body.name,
        email=body.email,
        hashed_password=hash_password(body.password),
        company=body.company,
    )
    db.add(op)
    await db.flush()
    token = create_access_token({"sub": str(op.id)})
    return TokenResponse(
        access_token=token,
        operator_id=str(op.id),
        name=op.name,
        company=op.company,
    )


@router.get("/me", response_model=OperatorOut)
async def get_me(current: Operator = Depends(get_current_operator)):
    return current
