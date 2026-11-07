"""Cấu hình đọc từ biến môi trường / `api/.env`.

Mọi tham số ở đây ánh xạ 1-1 tới hằng số trong notebook
`yolo_train_predict_v5_spatial_relationships_rtdetr.ipynb`, cell 1 → 10.14.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[2]  # .../api

TOP_K_MAX = 100
PRODUCTS_PER_GROUP_MAX = 50

# Cell 8 của notebook (train) KHÔNG được port: API chỉ đọc checkpoint .pt có sẵn.
ALLOWED_IMAGE_FORMATS = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=("settings_",),
    )

    # --- Ứng dụng ---
    app_name: str = "Room Recommendation API (v5)"
    app_version: str = "5.0.0"
    environment: str = "development"
    log_level: str = "INFO"
    docs_enabled: bool = True

    # --- Detector model (RF-DETR Medium pretrain, RT-DETR hoặc YOLO) ---
    model_path: str = "artifacts/rfdetr_medium"
    model_device: str = "auto"  # auto | cpu | mps | 0
    model_confidence: float = Field(0.5, ge=0, le=1)
    # Tái tạo bước nén JPEG trong bộ nhớ nếu cần khớp notebook
    detect_jpeg_roundtrip: bool = True
    # Kiểm tra khớp cứng tên class với ma trận đồng xuất hiện.
    # Với RF-DETR pretrained COCO thì tự động bỏ qua kiểm tra khớp cứng.
    require_class_match: bool = False

    # --- Cell 10.3 / 10.5: ma trận đồng xuất hiện học từ COCO train ---
    cooccurrence_matrix_path: str = "artifacts/matrix/cooccurrence_matrix.json"
    # Cell 10.7: bảng điểm W(c, r), khoá theo TÊN nhóm hiển thị.
    category_room_scores_path: str = "artifacts/matrix/category_room_scores.json"
    # Đường dẫn dữ liệu COCO — chỉ dùng để đối chiếu class của checkpoint.
    # Để trống = bỏ qua bước so class (khi không mang theo annotation).
    data_root: str = ""

    # --- Cell 10.1 / 10.6: catalog sản phẩm ---
    # Catalog CHỈ đọc từ PostgreSQL schema `mwg` (3 bảng: product, categories, brand).
    # Không còn fallback CSV: DB không tới được là API khởi động thất bại.
    db_host: str = "127.0.0.1"
    db_port: int = 55432
    db_name: str = "mwgshop"
    db_user: str = "mwg"
    db_password: str = "mwg_secret"
    db_schema: str = "mwg"
    db_connect_timeout_s: int = Field(5, ge=1)

    # --- Tham số gợi ý (Cell 10.1) ---
    default_room_type: str = "living_room"
    default_top_k: int = Field(10, ge=1, le=TOP_K_MAX)
    default_min_score: float = Field(0.05, ge=0)
    products_per_group: int = Field(3, ge=1, le=PRODUCTS_PER_GROUP_MAX)
    default_exclude_detected: bool = True
    # Cell 10.9: hệ số Laplace smoothing.
    alpha_smooth: float = Field(1.0, ge=0)
    smooth_k: int = Field(2, ge=1)

    # --- Spatial Placement & Render ---
    placement_rules_path: str = "artifacts/placement_rules.json"
    remove_bg_base_url: str = ""
    remove_bg_api_url: str = ""
    remove_bg_timeout_s: float = Field(120.0, gt=0)

    @model_validator(mode="after")
    def _resolve_remove_bg_url(self) -> Settings:
        if not self.remove_bg_api_url and self.remove_bg_base_url:
            base = self.remove_bg_base_url.rstrip("/")
            if "/remove-background" in base:
                self.remove_bg_api_url = base
            else:
                self.remove_bg_api_url = f"{base}/remove-background"
        return self

    # --- Giới hạn đầu vào ---
    max_upload_mb: float = Field(10, gt=0)
    max_image_pixels: int = Field(25_000_000, gt=0)
    min_image_side: int = Field(32, ge=1)
    max_pending_requests: int = Field(4, ge=1)

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mb * 1024 * 1024)

    def resolve(self, path: str) -> Path:
        p = Path(path).expanduser()
        return p if p.is_absolute() else (BASE_DIR / p).resolve()


@lru_cache
def get_settings() -> Settings:
    return Settings()
