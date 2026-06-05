"""Tests for the /upload endpoint (raw-body file ingestion → reference cache)."""
import base64
import hashlib

import pytest
from fastapi.testclient import TestClient

import main

CREDS = "admin:test-secret"
AUTH = {"Authorization": "Basic " + base64.b64encode(CREDS.encode()).decode()}


class FakeCache:
    """Minimal redis stand-in so tests don't need a live redis."""
    def __init__(self):
        self.store = {}

    def set(self, k, v):
        # main.py's redis client uses decode_responses=True, so stored bytes
        # come back as str. Mirror that so get() returns the right type.
        self.store[k] = v.decode() if isinstance(v, (bytes, bytearray)) else v

    def get(self, k):
        return self.store.get(k)

    def exists(self, k):
        return k in self.store


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("HTTP_WRITE_AUTH", CREDS)
    monkeypatch.setattr(main, "cache", FakeCache())


client = TestClient(main.app)


def test_upload_requires_auth():
    assert client.post("/upload", params={"reference": "r"}, content=b"x").status_code == 401


def test_upload_rejects_empty_body():
    r = client.post("/upload", params={"reference": "r"}, content=b"", headers=AUTH)
    assert r.status_code == 400


def test_upload_returns_expected_key_and_roundtrips():
    body = b"%PDF-1.4 fake pdf bytes"
    r = client.post(
        "/upload",
        params={"reference": "deal-123.pdf"},
        content=body,
        headers={**AUTH, "Content-Type": "application/pdf"},
    )
    assert r.status_code == 200
    key = r.json()["key"]
    # key is the sha256 of the reference seed (matches make_reference_key)
    assert key == hashlib.sha256(b"deal-123.pdf").hexdigest()

    # GET /reference/{key} returns the same bytes + content-type
    got = client.get(f"/reference/{key}")
    assert got.status_code == 200
    assert got.content == body
    assert got.headers["content-type"].startswith("application/pdf")
