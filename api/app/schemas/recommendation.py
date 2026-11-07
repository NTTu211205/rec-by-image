"""Hợp đồng dữ liệu của API (Pydantic) — kèm ví dụ cho Swagger UI.

Mọi `model_config` ở đây chỉ phục vụ tài liệu (`/docs`, `/redoc`):
Swagger UI dùng chúng để điền sẵn request body và hiển thị response mẫu.
Mô tả field giữ ngắn (một cụm từ); công thức chi tiết nằm trong README.
"""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, ConfigDict, Field

# --- Ví dụ dùng lại nhiều nơi ---

_EXAMPLE_PRODUCT = {
    "product_id": "000001",
    "product_code": "000000000001",
    "product_name": "Samsung Smart TV UA32H5000F",
    "brand": "Samsung",
    "price": 4090000.0,
    "currency": "VND",
    "image_url": "https://cdn.tgdd.vn/2026/10/timerseo/361163-600x600-8.jpg",
    "product_url": "https://www.dienmayxanh.com/tivi/smart-tivi-samsung-32-inch-ua32h5000f",
    "category_id": "56",
}

_EXAMPLE_DETECTED = [
    {"label": "Sofa", "confidence": 0.96},
    {"label": "Rug", "confidence": 0.88},
    {"label": "TV", "confidence": 0.81},
    {"label": "Coffee Table", "confidence": 0.74},
]

_EXAMPLE_RESPONSE = {
    "room_type": "living_room",
    "detected_objects": _EXAMPLE_DETECTED,
    "recommendations": [
        {
            "group_code": "speaker",
            "group_name": "Loa",
            "category_id": "electronics",
            "cooccurrence_score": 0.4123,
            "room_score": 1.0,
            "final_score": 1.4123,
            "product_count": 25,
            "reason": "Ảnh có TV — thường đi kèm Loa (41%); điểm phòng 1.0.",
            "products": [_EXAMPLE_PRODUCT],
        }
    ],
}


class Product(BaseModel):
    """Một sản phẩm lấy từ catalog (Cell 10.8)."""

    model_config = ConfigDict(json_schema_extra={"example": _EXAMPLE_PRODUCT})

    product_id: str = Field(..., description="Mã sản phẩm (dãy 6 số)", examples=["000001"])
    product_code: str | None = Field(None, description="Mã sản phẩm (dãy 12 số)", examples=["000000000001"])
    product_name: str = Field(..., description="Tên sản phẩm", examples=["Samsung Smart TV UA32H5000F"])
    brand: str = Field(..., description="Thương hiệu", examples=["Samsung"])
    price: float = Field(..., description="Giá niêm yết", examples=[4090000.0])
    currency: str = Field(..., description="Đơn vị tiền tệ", examples=["VND"])
    image_url: str = Field(..., description="Ảnh sản phẩm")
    product_url: str = Field(..., description="Trang sản phẩm")
    category_id: str | None = Field(None, description="Mã category_id trong DB", examples=["56"])


class RecommendationItem(BaseModel):
    """Một nhóm sản phẩm được gợi ý, kèm điểm và lý do (Cell 10.10)."""

    group_code: str = Field(..., description="Mã nhóm (= subcategory)", examples=["tv"])
    group_name: str = Field(..., description="Tên nhóm hiển thị", examples=["TV"])
    category_id: str | None = Field(
        None, description="Category ID trong DB", examples=["56"]
    )
    cooccurrence_score: float = Field(
        ..., description="`P_cooc(c | D)` — Noisy-OR (Cell 10.9).", examples=[0.4123]
    )
    room_score: float = Field(
        ..., description="`W(c, r)` — điểm phù hợp với phòng.", examples=[0.9]
    )
    final_score: float = Field(
        ..., description="`P_cooc + W(c, r)`.", examples=[1.3123]
    )
    product_count: int = Field(..., description="Số sản phẩm của nhóm.", examples=[25])
    reason: str = Field(..., description="Câu giải thích.")
    products: list[Product] = Field(..., description="Sản phẩm còn hàng.")


class DetectedObject(BaseModel):
    """Vật thể detector phát hiện được; nhãn đã gộp trùng giữ confidence cao nhất."""

    label: str = Field(..., description="Tên class detector", examples=["Sofa"])
    confidence: float = Field(..., ge=0, le=1, description="Độ tin cậy [0, 1]", examples=[0.96])
    bbox: list[float] | None = Field(None, description="Tọa độ [x1, y1, x2, y2]", examples=[[10.0, 20.0, 100.0, 200.0]])


class PlacementRenderItem(BaseModel):
    """Chi tiết kết quả gắn từng sản phẩm lên ảnh."""

    rank: int = Field(..., description="Thứ tự ưu tiên gợi ý", examples=[1])
    category_id: str | None = Field(None, description="Mã danh mục", examples=["37"])
    group_name: str | None = Field(None, description="Tên nhóm hiển thị", examples=["TV"])
    placement_group: str | None = Field(None, description="Nhóm không gian placement", examples=["TV"])
    product_type: str | None = Field(None, description="Loại sản phẩm cho API xóa nền", examples=["tv"])
    product_id: str | None = Field(None, description="Mã sản phẩm", examples=["341445"])
    product_name: str | None = Field(None, description="Tên sản phẩm", examples=["Smart Tivi OLED LG 55 inch"])
    price: float | None = Field(None, description="Giá sản phẩm", examples=[25000000.0])
    currency: str | None = Field("VND", description="Đơn vị tiền tệ", examples=["VND"])
    image_url: str | None = Field(None, description="URL ảnh sản phẩm")
    status: str = Field(..., description="Trạng thái: rendered, api_or_render_failed, unmapped_category, no_placement, etc.")
    reason: str = Field("", description="Chi tiết lỗi nếu thất bại")
    surface: str | None = Field(None, description="Bề mặt đặt: wall, floor, table, ceiling")
    bbox_px: dict[str, float | int] | None = Field(None, description="Tọa độ vùng đặt dự kiến {x1, y1, x2, y2}")
    rendered_bbox_px: dict[str, float | int] | None = Field(None, description="Tọa độ thực tế sau khi ghép {x1, y1, x2, y2}")


class PlacementRenderResponse(BaseModel):
    """Kết quả gợi ý kèm ảnh render đã ghép sản phẩm (V8.20)."""

    room_type: str = Field(..., description="Phòng đã dùng.", examples=["living_room"])
    detected_objects: list[DetectedObject] = Field(
        ..., description="Vật thể đưa vào chấm điểm."
    )
    recommendations: list[RecommendationItem] = Field(
        ..., description="Đã sắp giảm dần theo `final_score`."
    )
    items: list[PlacementRenderItem] = Field(
        default_factory=list, description="Chi tiết từng sản phẩm được gắn lên ảnh hoặc lỗi kèm theo."
    )
    image: str | None = Field(
        None, description="Ảnh phòng đã ghép sản phẩm dưới dạng Data URI PNG (data:image/png;base64,...)"
    )
    rendered_image: str | None = Field(
        None, description="Ảnh phòng đã ghép sản phẩm dưới dạng Data URI PNG (data:image/png;base64,...)"
    )
    rendered_count: int = Field(0, description="Số sản phẩm ghép thành công.")
    recommended_count: int = Field(0, description="Tổng số sản phẩm gợi ý được xử lý.")
    diagnostics: dict[str, Any] | None = Field(
        default_factory=dict, description="Thông tin chẩn đoán (segmentation, depth, lighting)."
    )


class RecommendationResponse(BaseModel):
    """Kết quả của `recommend()` (Cell 10.10)."""

    model_config = ConfigDict(json_schema_extra={"example": _EXAMPLE_RESPONSE})

    room_type: str = Field(..., description="Phòng đã dùng.", examples=["living_room"])
    detected_objects: list[DetectedObject] = Field(
        ..., description="Vật thể đưa vào chấm điểm."
    )
    recommendations: list[RecommendationItem] = Field(
        ..., description="Đã sắp giảm dần theo `final_score`."
    )


class HealthResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "ready",
                "ready": True,
                "model_family": "rtdetr",
                "detector_classes": 45,
                "model_source": "best.pt (rtdetr)",
                "room_scores_source": "artifact category_room_scores.json (62 nhóm × 7 phòng) + mwg.categories",
                "cooccurrence_source": "cooccurrence_matrix.json (45 class, 1692 ảnh train)",
                "catalog_groups": 62,
                "catalog_source": "PostgreSQL mwg.product (1413 sản phẩm)",
                "startup_errors": [],
            }
        }
    )

    status: str = Field(..., description="`ready` hoặc `not_ready`.")
    ready: bool = Field(..., description="Đã nạp xong chưa.")
    model_family: str | None = Field(None, description="`rfdetr`, `rtdetr` hoặc `yolo`.")
    detector_classes: int = Field(0, description="Số class của checkpoint.")
    model_source: str | None = Field(None, description="Nguồn model detector.")
    room_scores_source: str | None = Field(
        None, description="Nguồn `W(c, r)`."
    )
    cooccurrence_source: str | None = Field(
        None, description="Nguồn ma trận đồng xuất hiện."
    )
    catalog_groups: int = Field(0, description="Số nhóm trong catalog.")
    catalog_source: str | None = Field(None, description="Nguồn catalog sản phẩm.")
    startup_errors: list[str] = Field(
        default_factory=list, description="Lỗi không nghiêm trọng khi khởi động."
    )


class ErrorDetail(BaseModel):
    code: str = Field(..., description="Mã lỗi.", examples=["invalid_image"])
    message: str = Field(..., description="Thông báo.")


class ErrorResponse(BaseModel):
    """Mọi lỗi nghiệp vụ đều trả cùng một phong bì."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {"error": {"code": "invalid_image", "message": "Dữ liệu không phải ảnh hợp lệ."}}
        }
    )

    error: ErrorDetail
