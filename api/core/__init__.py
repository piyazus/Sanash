"""API Core Module"""

from .config import settings, get_settings
from .database import Base, get_db, get_db_context, init_db, close_db
from .security import (
    hash_password,
    verify_password,
    create_access_token,
    create_refresh_token,
    create_token_pair,
    decode_token,
)
from .cache import get_redis, init_redis, close_redis, Cache
from .dependencies import (
    get_current_user,
    get_current_user_optional,
    get_authenticated_user,
    require_role,
)

__all__ = [
    "settings",
    "get_settings",
    "Base",
    "get_db",
    "get_db_context",
    "init_db",
    "close_db",
    "hash_password",
    "verify_password",
    "create_access_token",
    "create_refresh_token",
    "create_token_pair",
    "decode_token",
    "get_redis",
    "init_redis",
    "close_redis",
    "Cache",
    "get_current_user",
    "get_current_user_optional",
    "get_authenticated_user",
    "require_role",
]
