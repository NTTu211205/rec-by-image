"""Test tài liệu OpenAPI/Swagger.

Mục tiêu: bảo đảm `/docs`, `/redoc`, `/openapi.json` luôn phục vụ được, schema có
đủ metadata cần thiết, và **chỉ có hai endpoint** được công bố.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app

EXPECTED_OPERATIONS = {
    ("get", "/health/ready"): "healthReady",
    ("post", "/api/v1/recommendations/image"): "recommendFromImage",
}

# Endpoint đã bỏ — phải 404, không được lọt lại vào schema.
REMOVED_PATHS = (
    "/health/live",
    "/api/v1/recommend/labels",
    "/api/v1/rooms",
    "/api/v1/detector/classes",
    "/api/v1/evaluation/cooccurrence",
)


@pytest.fixture(scope="module")
def client() -> TestClient:
    app = create_app(Settings(require_class_match=False))
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def spec(client: TestClient) -> dict:
    r = client.get("/openapi.json")
    assert r.status_code == 200
    return r.json()


def test_docs_page_available(client: TestClient) -> None:
    r = client.get("/docs")
    assert r.status_code == 200
    # Swagger UI được nhúng qua CDN và trỏ tới /openapi.json.
    assert "swagger-ui-bundle.js" in r.text
    assert "/openapi.json" in r.text


def test_redoc_page_available(client: TestClient) -> None:
    r = client.get("/redoc")
    assert r.status_code == 200
    assert "redoc" in r.text.lower()


def test_root_lists_docs(client: TestClient) -> None:
    body = client.get("/").json()
    assert body["docs"] == "/docs"
    assert body["redoc"] == "/redoc"
    assert body["openapi"] == "/openapi.json"
    assert set(body["endpoints"]) == {"health_ready", "recommend_image"}


def test_info_metadata(spec: dict) -> None:
    info = spec["info"]
    assert info["title"] == "Room Recommendation API (v5)"
    assert info["version"] == "5.0.0"
    assert "RT-DETR" in info["summary"]
    # Mô tả ngắn: một tới hai câu, không phải bài viết dài trong Swagger.
    description = info["description"]
    assert description, "info.description không được rỗng"
    assert len(description) <= 300, f"info.description quá dài ({len(description)} ký tự)"
    assert "\n" not in description.strip()
    assert "Không có bước train" in description


def test_only_two_operations(spec: dict) -> None:
    assert spec["openapi"].startswith("3.")
    tags = {t["name"] for t in spec.get("tags", [])}
    assert tags == {"recommendations", "health"}

    actual = {
        (method, path)
        for path, methods in spec["paths"].items()
        for method in methods
    }
    assert actual == set(EXPECTED_OPERATIONS)

    for (method, path), operation_id in EXPECTED_OPERATIONS.items():
        operation = spec["paths"][path][method]
        assert operation["operationId"] == operation_id
        assert operation["summary"], f"{path} thiếu summary"
        assert operation["description"], f"{path} thiếu description"
        assert operation["tags"], f"{path} thiếu tag"
        # Mô tả một dòng, không nhồi công thức/bảng Markdown vào Swagger.
        assert len(operation["description"]) <= 200, f"{path} description quá dài"
        assert "\n" not in operation["description"], f"{path} description nhiều dòng"


@pytest.mark.parametrize("path", REMOVED_PATHS)
def test_removed_endpoints_absent(client: TestClient, spec: dict, path: str) -> None:
    assert path not in spec["paths"], f"{path} vẫn còn trong schema"
    assert client.get(path).status_code == 404


def test_removed_post_endpoints_absent(client: TestClient) -> None:
    assert client.post("/api/v1/recommend/labels", json={}).status_code == 404


def test_removed_schemas_gone(spec: dict) -> None:
    """Model chỉ phục vụ endpoint đã xoá không được còn trong components."""
    names = set(spec["components"]["schemas"])
    for removed in (
        "RecommendRequest",
        "RoomTypesResponse",
        "DetectorClassesResponse",
        "EvaluationResponse",
        "FormulaComparison",
        "FormulaMetrics",
        "RareProxyComparison",
    ):
        assert removed not in names, f"schema {removed} vẫn còn"
    assert {"RecommendationResponse", "HealthResponse", "ErrorResponse"} <= names


def test_error_responses_use_shared_schema(spec: dict) -> None:
    """Phản hồi lỗi phải trỏ tới ErrorResponse, không inline $defs."""
    schemas = spec["components"]["schemas"]
    assert "ErrorResponse" in schemas
    assert "ErrorDetail" in schemas

    operation = spec["paths"]["/api/v1/recommendations/image"]["post"]
    assert "503" in operation["responses"]
    for status in ("400", "404", "429", "503"):
        if status not in operation["responses"]:
            continue
        ref = operation["responses"][status]["content"]["application/json"]["schema"]
        assert ref == {"$ref": "#/components/schemas/ErrorResponse"}


def test_requests_have_examples(spec: dict) -> None:
    assert "example" in spec["components"]["schemas"]["RecommendationResponse"]
    assert "example" in spec["components"]["schemas"]["HealthResponse"]


def test_response_fields_have_descriptions(spec: dict) -> None:
    """Mọi field phải có mô tả, nhưng ngắn gọn (không phải một đoạn văn)."""
    for name in ("RecommendationItem", "Product", "HealthResponse"):
        props = spec["components"]["schemas"][name]["properties"]
        missing = [field for field, body in props.items() if not body.get("description")]
        assert not missing, f"{name} thiếu description: {missing}"
        long = {
            field: len(body["description"])
            for field, body in props.items()
            if len(body.get("description", "")) > 120
        }
        assert not long, f"{name} description quá dài: {long}"


def test_tag_descriptions_are_short(spec: dict) -> None:
    for tag in spec.get("tags", []):
        description = tag.get("description", "")
        assert description, f"tag {tag['name']} thiếu description"
        assert len(description) <= 100, f"tag {tag['name']} description quá dài"


def test_image_endpoint_documents_body_and_query(spec: dict) -> None:
    operation = spec["paths"]["/api/v1/recommendations/image"]["post"]

    body_schema = operation["requestBody"]["content"]["multipart/form-data"]["schema"]
    assert body_schema["$ref"].endswith("Body_recommendFromImage")

    params = {p["name"] for p in operation["parameters"]}
    assert {"room_type", "top_k", "min_score", "products_per_group", "confidence"} <= params
    # `image` nằm trong multipart body, không phải query.
    assert "image" not in params


def test_openapi_disabled_hides_docs() -> None:
    app = create_app(Settings(require_class_match=False, docs_enabled=False))
    with TestClient(app) as c:
        assert c.get("/docs").status_code == 404
        assert c.get("/openapi.json").status_code == 404
        assert c.get("/").json()["docs"] is None
