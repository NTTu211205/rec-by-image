"""Route health."""

from __future__ import annotations

from fastapi import APIRouter, Response

from app.dependencies import AppStateDep
from app.api.openapi import ERROR_CONTENT
from app.schemas.recommendation import HealthResponse

router = APIRouter(tags=["health"])


@router.get(
    "/health/ready",
    response_model=HealthResponse,
    summary="Model + dữ liệu đã sẵn sàng",
    description=(
        "Readiness probe: model, ma trận và catalog đã nạp xong. "
        "Trả 503 khi chưa sẵn sàng."
    ),
    response_description="Trạng thái model và dữ liệu",
    operation_id="healthReady",
    responses={
        200: {"description": "Đã sẵn sàng."},
        503: {"description": "Chưa sẵn sàng.", "content": ERROR_CONTENT},
    },
)
async def ready(response: Response, state: AppStateDep) -> HealthResponse:
    is_ready = state.ready
    if not is_ready:
        response.status_code = 503
    return HealthResponse(
        status="ready" if is_ready else "not_ready",
        ready=is_ready,
        model_family=state.model_family,
        detector_classes=len(state.detector.class_names) if state.detector else 0,
        model_source=state.model_source,
        room_scores_source=state.room_scores_source,
        cooccurrence_source=state.cooccurrence_source,
        catalog_groups=len(state.catalog.groups) if state.catalog else 0,
        catalog_source=state.catalog_source,
        startup_errors=state.startup_errors,
    )
