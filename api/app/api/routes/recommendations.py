"""Route gợi ý: ảnh -> detect (Cell 10.13) -> recommend (Cell 10.10)."""

from __future__ import annotations

import asyncio
import base64
import logging

from fastapi import APIRouter, Depends, File, Query, UploadFile
from fastapi.responses import Response

from app.core.config import PRODUCTS_PER_GROUP_MAX, TOP_K_MAX, Settings, get_settings
from app.core.exceptions import BusyError, ModelNotReadyError, NotFoundError
from app.dependencies import AppStateDep
from app.api.openapi import ERROR_CONTENT
from app.schemas.recommendation import RecommendationResponse
from app.services import group_scores
from app.services.images import load_image_bytes
from app.services.placement import render_room_recommendations

logger = logging.getLogger("app.routes.recommendations")
router = APIRouter(prefix="/api/v1", tags=["recommendations"])


def _validate_room(room_type: str | None, state) -> str:
    allowed = set(state.recommender.room_types)
    room = room_type or state.settings.default_room_type
    if room not in allowed:
        raise NotFoundError(
            f"room_type '{room}' không hợp lệ. Chọn một trong: {', '.join(sorted(allowed))}."
        )
    return room


def _validate_category(category_id: str | None, state: AppState | None = None) -> str | None:
    if category_id is None:
        return None
    valid = set(group_scores.CATEGORY_NAMES)
    if state and state.catalog and getattr(state.catalog, "category_ids", None):
        valid.update(state.catalog.category_ids.values())
    if category_id not in valid:
        raise NotFoundError(
            f"category_id '{category_id}' không hợp lệ. Chọn một trong các ID danh mục của DB hoặc: "
            f"{', '.join(sorted(group_scores.CATEGORY_NAMES))}."
        )
    return category_id


def _price_range(min_price: float | None, max_price: float | None):
    if min_price is None and max_price is None:
        return None
    return (0.0 if min_price is None else min_price, float("inf") if max_price is None else max_price)


@router.post(
    "/recommendations/image",
    response_model=RecommendationResponse,
    summary="Upload ảnh phòng → danh sách nhóm sản phẩm gợi ý",
    description="Upload ảnh phòng (JPEG/PNG/WEBP) → detect → xếp hạng nhóm sản phẩm.",
    response_description="Các nhóm sản phẩm đã xếp hạng",
    operation_id="recommendFromImage",
    responses={
        200: {"description": "Gợi ý thành công."},
        400: {"description": "Ảnh không hợp lệ.", "content": ERROR_CONTENT},
        404: {"description": "Tham số ngoài miền hợp lệ.", "content": ERROR_CONTENT},
        429: {"description": "Quá nhiều yêu cầu đang xử lý.", "content": ERROR_CONTENT},
        503: {"description": "Model chưa sẵn sàng.", "content": ERROR_CONTENT},
    },
)
async def recommend_from_image(
    state: AppStateDep,
    settings: Settings = Depends(get_settings),
    image: UploadFile = File(
        ...,
        description="Ảnh phòng: JPEG/PNG/WEBP.",
    ),
    room_type: str | None = Query(
        None,
        description="Mã phòng; bỏ trống → `DEFAULT_ROOM_TYPE`.",
        examples=["living_room"],
    ),
    top_k: int | None = Query(
        None, ge=1, le=TOP_K_MAX,
        description=f"Số nhóm trả về (1–{TOP_K_MAX}).",
        examples=[5],
    ),
    min_score: float | None = Query(
        None, ge=0,
        description="Ngưỡng `final_score` tối thiểu.",
        examples=[0.05],
    ),
    exclude_detected: bool | None = Query(
        None,
        description="Loại nhóm đã có trong ảnh.",
    ),
    products_per_group: int | None = Query(
        None, ge=1, le=PRODUCTS_PER_GROUP_MAX,
        description=f"Số sản phẩm mỗi nhóm (1–{PRODUCTS_PER_GROUP_MAX}).",
        examples=[3],
    ),
    min_price: float | None = Query(None, ge=0, description="Giá thấp nhất."),
    max_price: float | None = Query(None, ge=0, description="Giá cao nhất."),
    category_id: str | None = Query(
        None,
        description="Chỉ gợi ý trong một category.",
        examples=["electronics"],
    ),
    confidence: float | None = Query(
        None, ge=0, le=1,
        description="Ngưỡng confidence của detector.",
        examples=[0.5],
    ),
) -> RecommendationResponse:
    if not state.ready:
        raise ModelNotReadyError("Model chưa sẵn sàng.")

    room = _validate_room(room_type, state)
    category = _validate_category(category_id, state)
    if top_k is not None and not (1 <= top_k <= TOP_K_MAX):
        raise NotFoundError(f"top_k phải trong [1, {TOP_K_MAX}].")
    if products_per_group is not None and not (1 <= products_per_group <= PRODUCTS_PER_GROUP_MAX):
        raise NotFoundError(f"products_per_group phải trong [1, {PRODUCTS_PER_GROUP_MAX}].")

    data = await image.read()
    if len(data) > settings.max_upload_bytes:
        raise NotFoundError(f"Ảnh vượt {settings.max_upload_mb:g} MB.")
    img = load_image_bytes(
        data, max_pixels=settings.max_image_pixels, min_side=settings.min_image_side
    )

    if not state.try_acquire():
        raise BusyError("Hệ thống đang xử lý quá nhiều yêu cầu, thử lại sau.")
    try:
        detected = await asyncio.get_running_loop().run_in_executor(
            state.executor, state.detector.detect, img,
            confidence if confidence is not None else settings.model_confidence,
        )
        result = state.recommender.recommend(
            detected,
            room_type=room,
            top_k=top_k,
            min_score=min_score,
            exclude_detected=(
                settings.default_exclude_detected if exclude_detected is None else exclude_detected
            ),
            products_per_group=products_per_group,
            price_range=_price_range(min_price, max_price),
            category_id=category,
        )
    finally:
        state.release()

    logger.info(
        "recommend done",
        extra={"event": "recommend", "detected": len(detected),
               "groups": len(result["recommendations"]), "room_type": room},
    )
    return RecommendationResponse(**result)


@router.post(
    "/recommendations/render",
    response_class=Response,
    summary="Upload ảnh phòng → trả về ảnh phòng đã gắn vật thể (Placement Render)",
    description=(
        "Upload ảnh phòng (JPEG/PNG/WEBP), nhận diện vật thể và gợi ý sản phẩm phù hợp. "
        "Lấy top 1 sản phẩm ở mỗi category gợi ý, xóa nền và gắn lên phòng theo hình học và ánh sáng không gian (V8.20). "
        "Trả về duy nhất ảnh PNG đã gắn vật thể (media_type: image/png)."
    ),
    response_description="Ảnh PNG phòng đã gắn vật thể",
    operation_id="recommendAndRenderFromImage",
    responses={
        200: {
            "content": {
                "image/png": {
                    "schema": {"type": "string", "format": "binary"}
                },
            },
            "description": "Ảnh PNG phòng đã gắn vật thể.",
        },
        400: {"description": "Ảnh không hợp lệ.", "content": ERROR_CONTENT},
        404: {"description": "Tham số ngoài miền hợp lệ.", "content": ERROR_CONTENT},
        429: {"description": "Quá nhiều yêu cầu đang xử lý.", "content": ERROR_CONTENT},
        503: {"description": "Model chưa sẵn sàng.", "content": ERROR_CONTENT},
    },
)
@router.post(
    "/recommendations/image/render",
    response_class=Response,
    include_in_schema=False,
)
async def recommend_and_render_from_image(
    state: AppStateDep,
    settings: Settings = Depends(get_settings),
    image: UploadFile = File(
        ...,
        description="Ảnh phòng: JPEG/PNG/WEBP.",
    ),
    room_type: str | None = Query(
        None,
        description="Mã phòng; bỏ trống → `DEFAULT_ROOM_TYPE`.",
        examples=["living_room"],
    ),
    top_k: int | None = Query(
        None, ge=1, le=TOP_K_MAX,
        description=f"Số nhóm trả về (1–{TOP_K_MAX}).",
        examples=[5],
    ),
    min_score: float | None = Query(
        None, ge=0,
        description="Ngưỡng `final_score` tối thiểu.",
        examples=[0.05],
    ),
    exclude_detected: bool | None = Query(
        None,
        description="Loại nhóm đã có trong ảnh.",
    ),
    products_per_group: int | None = Query(
        None, ge=1, le=PRODUCTS_PER_GROUP_MAX,
        description=f"Số sản phẩm mỗi nhóm (1–{PRODUCTS_PER_GROUP_MAX}). Mặc định 3.",
        examples=[3],
    ),
    min_price: float | None = Query(None, ge=0, description="Giá thấp nhất."),
    max_price: float | None = Query(None, ge=0, description="Giá cao nhất."),
    category_id: str | None = Query(
        None,
        description="Chỉ gợi ý trong một category.",
        examples=["electronics"],
    ),
    confidence: float | None = Query(
        None, ge=0, le=1,
        description="Ngưỡng confidence của detector.",
        examples=[0.5],
    ),
) -> Response:
    if not state.ready:
        raise ModelNotReadyError("Model chưa sẵn sàng.")

    room = _validate_room(room_type, state)
    category = _validate_category(category_id, state)
    if top_k is not None and not (1 <= top_k <= TOP_K_MAX):
        raise NotFoundError(f"top_k phải trong [1, {TOP_K_MAX}].")
    if products_per_group is not None and not (1 <= products_per_group <= PRODUCTS_PER_GROUP_MAX):
        raise NotFoundError(f"products_per_group phải trong [1, {PRODUCTS_PER_GROUP_MAX}].")

    data = await image.read()
    if len(data) > settings.max_upload_bytes:
        raise NotFoundError(f"Ảnh vượt {settings.max_upload_mb:g} MB.")
    img = load_image_bytes(
        data, max_pixels=settings.max_image_pixels, min_side=settings.min_image_side
    )

    if not state.try_acquire():
        raise BusyError("Hệ thống đang xử lý quá nhiều yêu cầu, thử lại sau.")
    try:
        detected = await asyncio.get_running_loop().run_in_executor(
            state.executor, state.detector.detect, img,
            confidence if confidence is not None else settings.model_confidence,
        )
        rec_result = state.recommender.recommend(
            detected,
            room_type=room,
            top_k=top_k,
            min_score=min_score,
            exclude_detected=(
                settings.default_exclude_detected if exclude_detected is None else exclude_detected
            ),
            products_per_group=products_per_group or settings.products_per_group,
            price_range=_price_range(min_price, max_price),
            category_id=category,
        )
        # Spatial placement and photometric render (V8.20)
        render_result = await asyncio.get_running_loop().run_in_executor(
            state.executor,
            render_room_recommendations,
            img,
            rec_result,
            settings.remove_bg_api_url,
        )
    finally:
        state.release()

    logger.info(
        "render done",
        extra={
            "event": "render",
            "detected": len(detected),
            "rendered": render_result.get("rendered_count", 0),
            "total_items": len(render_result.get("items", [])),
            "room_type": room,
        },
    )

    image_bytes = render_result.get("image_bytes")
    if not image_bytes:
        b64 = (render_result.get("image") or "").split(",", 1)[-1]
        image_bytes = base64.b64decode(b64)

    return Response(
        content=image_bytes,
        media_type="image/png",
        headers={
            "X-Room-Type": str(render_result.get("room_type") or ""),
            "X-Rendered-Count": str(render_result.get("rendered_count", 0)),
            "X-Recommended-Count": str(render_result.get("recommended_count", 0)),
        },
    )
