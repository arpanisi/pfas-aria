"""
Auth middleware.
Verifies Clerk JWT tokens on protected endpoints.
Public endpoints (health, docs) are excluded.
"""

from __future__ import annotations

import base64
import binascii
import json
import os
import time
from typing import Annotated

import requests
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

bearer_scheme = HTTPBearer(auto_error=False)

CLERK_PUBLISHABLE_KEY = os.getenv("CLERK_PUBLISHABLE_KEY", "")
CLERK_SECRET_KEY = os.getenv("CLERK_SECRET_KEY", "")


def _get_jwks_url() -> str:
    """
    Derive the JWKS URL from the Clerk publishable key.
    Publishable key format: pk_test_<base64-encoded-frontend-api>$
    Decoding gives the frontend API domain
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


_JWKS_TTL_SECONDS = 3600
_jwks_cache: dict[str, object] | None = None
_jwks_cache_loaded_at = 0.0


class InvalidTokenError(ValueError):
    """Raised when a JWT fails validation."""


class ExpiredTokenError(InvalidTokenError):
    """Raised when a JWT has expired."""


def _base64url_decode(value: str) -> bytes:
    value += "=" * (-len(value) % 4)
    try:
        return base64.urlsafe_b64decode(value.encode("ascii"))
    except (UnicodeEncodeError, binascii.Error) as exc:
        raise InvalidTokenError("Token contains invalid base64url") from exc


def _decode_token_part(value: str) -> dict:
    try:
        return json.loads(_base64url_decode(value))
    except (ValueError, TypeError) as e:
        raise InvalidTokenError("Token contains invalid JSON") from e


def _get_jwks() -> dict[str, object]:
    global _jwks_cache, _jwks_cache_loaded_at
    now = time.time()
    if _jwks_cache is None or now - _jwks_cache_loaded_at > _JWKS_TTL_SECONDS:
        response = requests.get(_get_jwks_url(), timeout=10)
        response.raise_for_status()
        jwks = response.json()
        if not isinstance(jwks, dict) or not isinstance(jwks.get("keys"), list):
            raise InvalidTokenError("JWKS endpoint returned an invalid key set")
        _jwks_cache = jwks
        _jwks_cache_loaded_at = now
    return _jwks_cache


def _get_jwk(kid: str) -> dict:
    for jwk in _get_jwks()["keys"]:
        if isinstance(jwk, dict) and jwk.get("kid") == kid:
            return jwk
    raise InvalidTokenError("Token signing key was not found")


def _public_key_from_jwk(jwk: dict) -> rsa.RSAPublicKey:
    try:
        n = int.from_bytes(_base64url_decode(str(jwk["n"])), "big")
        e = int.from_bytes(_base64url_decode(str(jwk["e"])), "big")
    except KeyError as exc:
        raise InvalidTokenError("Token signing key is missing RSA parameters") from exc
    return rsa.RSAPublicNumbers(e=e, n=n).public_key()


def _decode_and_verify_rs256(token: str) -> dict:
    parts = token.split(".")
    if len(parts) != 3:
        raise InvalidTokenError("Token must have three JWT segments")

    signing_input = f"{parts[0]}.{parts[1]}".encode("ascii")
    header = _decode_token_part(parts[0])
    payload = _decode_token_part(parts[1])
    signature = _base64url_decode(parts[2])

    if header.get("alg") != "RS256":
        raise InvalidTokenError("Unsupported token signing algorithm")
    kid = header.get("kid")
    if not isinstance(kid, str) or not kid:
        raise InvalidTokenError("Token is missing a signing key id")

    public_key = _public_key_from_jwk(_get_jwk(kid))
    try:
        public_key.verify(signature, signing_input, padding.PKCS1v15(), hashes.SHA256())
    except InvalidSignature as exc:
        raise InvalidTokenError("Token signature is invalid") from exc

    exp = payload.get("exp")
    if exp is not None:
        try:
            expires_at = float(exp)
        except (TypeError, ValueError) as exc:
            raise InvalidTokenError("Token expiration is invalid") from exc
        if expires_at <= time.time():
            raise ExpiredTokenError("Token has expired")

    return payload


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
        return _decode_and_verify_rs256(token)

    except ExpiredTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except InvalidTokenError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {e}",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception as e:
        # Network errors during JWKS refresh should not invalidate a valid session.
        # Log and raise 503 so the client retries rather than treating it as auth failure.
        from src.utils.logging import get_logger

        get_logger(__name__).warning("JWKS verification error (non-auth): {}", e)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Auth service temporarily unavailable — please retry",
        )


# Type alias for dependency injection
CurrentUser = Annotated[dict, Depends(verify_clerk_token)]
