"""
Auth middleware.
Verifies Clerk JWT tokens on protected endpoints.
Public endpoints (health, docs) are excluded.
"""

from __future__ import annotations

import os
from typing import Annotated

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jwt import PyJWKClient

bearer_scheme = HTTPBearer(auto_error=False)

CLERK_PUBLISHABLE_KEY = os.getenv("CLERK_PUBLISHABLE_KEY", "")
CLERK_SECRET_KEY = os.getenv("CLERK_SECRET_KEY", "")


# Derived from publishable key: pk_test_xxxx or pk_live_xxxx
# The frontend API is encoded in the publishable key after the prefix
def _get_jwks_url() -> str:
    """Derive JWKS URL from the Clerk secret key."""
    # Extract the instance domain from the secret key
    # sk_test_xxx → instance is embedded in JWT issuer
    # Use Clerk's standard JWKS endpoint
    if not CLERK_SECRET_KEY:
        raise ValueError("CLERK_SECRET_KEY is not set")
    return "https://api.clerk.com/v1/jwks"


# Cache the JWKS client
_jwks_client: PyJWKClient | None = None


def _get_jwks_client() -> PyJWKClient:
    global _jwks_client
    if _jwks_client is None:
        _jwks_client = PyJWKClient(
            _get_jwks_url(),
            headers={"Authorization": f"Bearer {CLERK_SECRET_KEY}"},
        )
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
