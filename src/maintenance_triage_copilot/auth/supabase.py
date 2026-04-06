"""Supabase authentication and invite helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import jwt
from jwt import PyJWKClient

from maintenance_triage_copilot.config import SupabaseConfig


@dataclass(frozen=True)
class SupabaseIdentity:
    user_id: str
    email: str | None
    claims: dict[str, Any]


@dataclass(frozen=True)
class SupabaseInviteResult:
    user_id: str
    email: str
    invite_status: str


class SupabaseAuthProvider:
    """Verify Supabase JWTs and send invite magic links."""

    def __init__(self, config: SupabaseConfig):
        self.config = config
        self._jwks_client: PyJWKClient | None = None
        jwks_url = config.jwks_url()
        if jwks_url:
            self._jwks_client = PyJWKClient(jwks_url)

    def configured_for_human_auth(self) -> bool:
        return self.config.configured_for_human_auth()

    def configured_for_invites(self) -> bool:
        return self.config.configured_for_invites()

    def verify_access_token(self, token: str) -> SupabaseIdentity:
        if not self.configured_for_human_auth():
            raise RuntimeError("Supabase human auth is not configured")
        if self._jwks_client is None:
            raise RuntimeError("Supabase JWKS client is unavailable")

        header = jwt.get_unverified_header(token)
        algorithm = str(header.get("alg", "RS256"))
        signing_key = self._jwks_client.get_signing_key_from_jwt(token)
        claims = jwt.decode(
            token,
            signing_key.key,
            algorithms=[algorithm],
            audience=self.config.jwt_audience,
            issuer=self.config.jwt_issuer,
        )
        user_id = claims.get("sub")
        if not isinstance(user_id, str) or not user_id.strip():
            raise ValueError("Supabase token is missing a valid subject")
        email = claims.get("email")
        if email is not None and not isinstance(email, str):
            email = None
        return SupabaseIdentity(
            user_id=user_id,
            email=email.strip().lower() if isinstance(email, str) else None,
            claims=dict(claims),
        )

    def invite_user(
        self,
        *,
        email: str,
        display_name: str,
        redirect_to: str | None = None,
    ) -> SupabaseInviteResult:
        if not self.configured_for_invites():
            raise RuntimeError("Supabase invite flow is not configured")
        project_url = self.config.project_url
        service_role_key = self.config.service_role_key
        assert project_url is not None
        assert service_role_key is not None

        payload: dict[str, Any] = {
            "email": email,
            "data": {"display_name": display_name},
        }
        if redirect_to:
            payload["redirect_to"] = redirect_to

        request = Request(
            url=project_url.rstrip("/") + "/auth/v1/invite",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {service_role_key}",
                "apikey": service_role_key,
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=15) as response:
                body = response.read().decode("utf-8") or "{}"
        except HTTPError as exc:  # pragma: no cover - network path depends on remote service
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(
                f"Supabase invite request failed with status {exc.code}: {detail or exc.reason}"
            ) from exc
        except URLError as exc:  # pragma: no cover - network path depends on remote service
            raise RuntimeError(f"Supabase invite request failed: {exc.reason}") from exc

        data = json.loads(body)
        user = data.get("user", data)
        if not isinstance(user, dict):
            raise RuntimeError("Supabase invite response did not include a user payload")
        user_id = user.get("id")
        if not isinstance(user_id, str) or not user_id.strip():
            raise RuntimeError("Supabase invite response did not include a user id")
        returned_email = user.get("email")
        normalized_email = (
            returned_email.strip().lower()
            if isinstance(returned_email, str) and returned_email.strip()
            else email
        )
        return SupabaseInviteResult(
            user_id=user_id,
            email=normalized_email,
            invite_status="sent",
        )

    def magic_link_url(self, *, email: str, redirect_to: str | None = None) -> str:
        """Builds the browser-facing magic-link request URL for diagnostics."""
        query = {"email": email}
        if redirect_to:
            query["redirect_to"] = redirect_to
        project_url = self.config.project_url
        if not project_url:
            raise RuntimeError("Supabase project URL is not configured")
        return project_url.rstrip("/") + "/auth/v1/invite?" + urlencode(query)
