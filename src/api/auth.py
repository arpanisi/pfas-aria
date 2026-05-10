"""
Auth middleware.
Verifies Clerk JWT tokens on protected endpoints.
Public endpoints (health, docs) are excluded.
"""

from __future__ import annotations

import base64
import os
from typing import Annotated

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jwt import PyJWKClient

bearer_scheme = HTTPBearer(auto_error=False)

CLERK_PUBLISHABLE_KEY = os.getenv("CLERK_PUBLISHABLE_KEY", "")
CLERK_SECRET_KEY = os.getenv("CLERK_SECRET_KEY", "")


def _get_jwks_url() -> str:
    """
    Derive the JWKS URL from the Clerk publishable key.
    Publishable key format: pk_test_<base64-encoded-frontend-api>$
    Decoding gives the frontend API domain e.g. super-penguin-94.clerk.accounts.dev
    """
    if not CLERK_PUBLISHABLE_KEY:
        raise ValueError("CLERK_PUBLISHABLE_KEY is not set")

    # Strip the prefix
    raw = CLERK_PUBLISHABLE_KEY
    for prefix in ("pk_live_", "pk_test_"):
        if raw.startswith(prefix):
            raw = raw[len(prefix) :]
            break

    # Add base64 padding
    raw += "=" * (4 - len(raw) % 4)

    # Decode and strip trailing $
    frontend_api = base64.b64decode(raw).decode("utf-8").strip("$").strip()

    return f"https://{frontend_api}/.well-known/jwks.json"


# Cache the JWKS client
_jwks_client: PyJWKClient | None = None


def _get_jwks_client() -> PyJWKClient:
    global _jwks_client
    if _jwks_client is None:
        jwks_url = _get_jwks_url()
        _jwks_client = PyJWKClient(jwks_url)
    return _jwks_client


async def verify_clerk_token(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)],
) -> dict:
    """
    Verify a Clerk JWT token using Clerk's JWKS endpoint.
    Returns the decoded payload on success.
    Raises 401 on failure.
    """
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authorization token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials

    try:
        jwks_client = _get_jwks_client()
        signing_key = jwks_client.get_signing_key_from_jwt(token)

        payload = jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256"],
            options={"verify_exp": True},
        )

        return payload

    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidTokenError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {e}",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Token verification failed: {e}",
            headers={"WWW-Authenticate": "Bearer"},
        )


# Type alias for dependency injection
CurrentUser = Annotated[dict, Depends(verify_clerk_token)]
