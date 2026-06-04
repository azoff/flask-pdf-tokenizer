"""Tests for HTTP_WRITE_AUTH basic-auth gating on write endpoints.

Auth is read at request time from the HTTP_WRITE_AUTH env var (format
"user:pass"). Empty/unset disables auth; set requires HTTP Basic on every
write route. `/` (health) is always open.
"""
import base64

import pytest
from fastapi import HTTPException
from fastapi.security import HTTPBasicCredentials
from fastapi.testclient import TestClient

import main

CREDS = "admin:PeruDogmaFinch"


def _basic(user: str, pw: str) -> dict[str, str]:
    token = base64.b64encode(f"{user}:{pw}".encode()).decode()
    return {"Authorization": f"Basic {token}"}


# ── direct unit tests of the dependency ──────────────────────────────────────

def test_disabled_when_unset(monkeypatch):
    monkeypatch.delenv("HTTP_WRITE_AUTH", raising=False)
    # No credentials, no env → passes (returns None, no raise).
    assert main.require_write_auth(None) is None


def test_missing_credentials_401(monkeypatch):
    monkeypatch.setenv("HTTP_WRITE_AUTH", CREDS)
    with pytest.raises(HTTPException) as exc:
        main.require_write_auth(None)
    assert exc.value.status_code == 401


def test_wrong_credentials_403(monkeypatch):
    monkeypatch.setenv("HTTP_WRITE_AUTH", CREDS)
    bad = HTTPBasicCredentials(username="admin", password="nope")
    with pytest.raises(HTTPException) as exc:
        main.require_write_auth(bad)
    assert exc.value.status_code == 403


def test_correct_credentials_pass(monkeypatch):
    monkeypatch.setenv("HTTP_WRITE_AUTH", CREDS)
    ok = HTTPBasicCredentials(username="admin", password="PeruDogmaFinch")
    assert main.require_write_auth(ok) is None


# ── integration tests: the dependency is actually wired to write routes ───────

client = TestClient(main.app)


def test_health_open_no_auth(monkeypatch):
    monkeypatch.setenv("HTTP_WRITE_AUTH", CREDS)
    assert client.get("/").status_code == 200


def test_write_route_requires_auth(monkeypatch):
    monkeypatch.setenv("HTTP_WRITE_AUTH", CREDS)
    assert client.post("/text").status_code == 401


def test_write_route_rejects_wrong(monkeypatch):
    monkeypatch.setenv("HTTP_WRITE_AUTH", CREDS)
    r = client.post("/text", headers=_basic("admin", "wrong"))
    assert r.status_code == 403


def test_write_route_accepts_correct(monkeypatch):
    monkeypatch.setenv("HTTP_WRITE_AUTH", CREDS)
    # Correct creds clear auth; the handler then rejects the empty body (422).
    # The point: it is NOT 401/403, proving auth passed.
    r = client.post("/text", headers=_basic("admin", "PeruDogmaFinch"))
    assert r.status_code not in (401, 403)
