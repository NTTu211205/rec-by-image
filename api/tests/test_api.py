"""Test tích hợp API: health + endpoint gợi ý từ ảnh."""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.core.config import Settings
from app.main import create_app


@pytest.fixture(scope="module")
def client() -> TestClient:
    settings = Settings()
    app = create_app(settings)
    with TestClient(app) as c:
        yield c


def _jpeg(size: tuple[int, int] = (640, 480)) -> io.BytesIO:
    buf = io.BytesIO()
    Image.new("RGB", size, (200, 200, 200)).save(buf, format="JPEG")
    buf.seek(0)
    return buf


def test_health_ready(client: TestClient) -> None:
    r = client.get("/health/ready")
    assert r.status_code == 200
    body = r.json()
    assert body["ready"] is True
    assert body["status"] == "ready"
    assert body["detector_classes"] == 45
    assert body["catalog_groups"] == 62
    assert body["room_scores_source"].startswith("artifact category_room_scores.json")
    assert body["cooccurrence_source"].startswith("cooccurrence_matrix.json")
    assert body["catalog_source"].startswith("PostgreSQL mwg.product")
    assert body["model_source"].endswith("(rtdetr)")
    assert body["startup_errors"] == []


def test_recommend_image(client: TestClient) -> None:
    r = client.post(
        "/api/v1/recommendations/image",
        files={"image": ("room.jpg", _jpeg(), "image/jpeg")},
        params={"room_type": "living_room", "top_k": 3},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["room_type"] == "living_room"
    assert "detected_objects" in body
    assert len(body["recommendations"]) <= 3

    for item in body["recommendations"]:
        # Cell 10.10: final_score = cooc + room_score
        assert item["final_score"] == pytest.approx(
            item["cooccurrence_score"] + item["room_score"], abs=1e-9
        )
        assert item["reason"]
        assert item["group_code"] and item["group_name"]


def test_recommend_image_defaults(client: TestClient) -> None:
    """Bỏ trống tham số → dùng mặc định trong Settings."""
    r = client.post(
        "/api/v1/recommendations/image",
        files={"image": ("room.jpg", _jpeg(), "image/jpeg")},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["room_type"] == "living_room"
    assert len(body["recommendations"]) <= 10


def test_recommend_image_top_k(client: TestClient) -> None:
    r = client.post(
        "/api/v1/recommendations/image",
        files={"image": ("room.jpg", _jpeg(), "image/jpeg")},
        params={"top_k": 1, "min_score": 0},
    )
    assert r.status_code == 200, r.text
    assert len(r.json()["recommendations"]) == 1


def test_invalid_room_type(client: TestClient) -> None:
    r = client.post(
        "/api/v1/recommendations/image",
        files={"image": ("room.jpg", _jpeg(), "image/jpeg")},
        params={"room_type": "garage"},
    )
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "not_found"


def test_reject_non_image(client: TestClient) -> None:
    r = client.post(
        "/api/v1/recommendations/image",
        files={"image": ("a.txt", b"not an image", "text/plain")},
    )
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_image"


def test_reject_oversized_image(client: TestClient) -> None:
    r = client.post(
        "/api/v1/recommendations/image",
        files={"image": ("big.jpg", _jpeg((20000, 2000)), "image/jpeg")},
    )
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_image"


def test_reject_tiny_image(client: TestClient) -> None:
    r = client.post(
        "/api/v1/recommendations/image",
        files={"image": ("tiny.jpg", _jpeg((10, 10)), "image/jpeg")},
    )
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_image"


def test_image_required(client: TestClient) -> None:
    r = client.post("/api/v1/recommendations/image")
    assert r.status_code == 422
