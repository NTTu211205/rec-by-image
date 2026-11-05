"""Khởi động: dựng toàn bộ trạng thái từ artifact có sẵn (không train).

Bốn nguồn dữ liệu, **không có bảng dự phòng nào trong code**. Thiếu hoặc hỏng
một trong bốn là khởi động thất bại với `ArtifactError` / `DataSourceError`
(HTTP 503), và log ghi rõ đường dẫn đã thử:

    MODEL_PATH                        checkpoint .pt (RT-DETR hoặc YOLO)
    COOCCURRENCE_MATRIX_PATH          ma trận đồng xuất hiện (45 class)
    CATEGORY_ROOM_SCORES_PATH         bảng W(c, r) theo tên nhóm
    PostgreSQL schema `mwg`           catalog sản phẩm (3 bảng)
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from app.core.config import Settings
from app.core.exceptions import ArtifactError
from app.services.cooccurrence import (
    DROP_CLASSES,
    NAME_REMAP,
    build_class_map,
    load_cooccurrence,
)
from app.services.detector import Detector
from app.services.product_catalog import Catalog, load_catalog_from_db, load_category_names
from app.services.recommender import Recommender
from app.services.room_scores import ROOM_CODES, RoomScoreTable, load_room_scores

logger = logging.getLogger("app.bootstrap")


@dataclass
class AppState:
    settings: Settings
    detector: Detector | None = None
    recommender: Recommender | None = None
    catalog: Catalog | None = None
    catalog_source: str | None = None
    room_scores_source: str | None = None
    cooccurrence_source: str | None = None
    model_source: str | None = None
    model_family: str | None = None
    detector_classes: list[str] = field(default_factory=list)
    startup_errors: list[str] = field(default_factory=list)
    executor: ThreadPoolExecutor = field(
        default_factory=lambda: ThreadPoolExecutor(max_workers=1, thread_name_prefix="infer")
    )
    pending: int = 0
    _pending_lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def ready(self) -> bool:
        return (
            self.detector is not None
            and self.detector.loaded
            and self.recommender is not None
        )

    def try_acquire(self) -> bool:
        """Giữ một suất xử lý; False nếu đã đạt `max_pending_requests`."""
        with self._pending_lock:
            if self.pending >= self.settings.max_pending_requests:
                return False
            self.pending += 1
            return True

    def release(self) -> None:
        with self._pending_lock:
            self.pending = max(0, self.pending - 1)


def expected_detector_classes(data_root: Path) -> list[str]:
    """Cell 6/9: bộ class detector theo `data.yaml` = kết quả `build_class_map`.

    Chỉ dùng cho `scripts/build_artifacts.py` (học lại ma trận từ annotation).
    Lúc chạy, API đối chiếu checkpoint với `cooccurrence_matrix.json["classes"]`
    nên **không cần** thư mục COCO.
    """
    id_to_idx, idx_to_name = build_class_map([Path(data_root) / "train"], NAME_REMAP, DROP_CLASSES)
    if not idx_to_name:  # pragma: no cover - dataset thiếu annotation
        raise ArtifactError(f"Không đọc được class map từ {data_root}/train")
    return [idx_to_name[i] for i in sorted(idx_to_name)]


def db_config(settings: Settings) -> dict:
    """Tham số `psycopg2.connect` từ Settings."""
    return {
        "host": settings.db_host,
        "port": settings.db_port,
        "dbname": settings.db_name,
        "user": settings.db_user,
        "password": settings.db_password,
        "connect_timeout": settings.db_connect_timeout_s,
    }


def _require_file(settings: Settings, path_str: str, what: str) -> Path:
    """Đường dẫn artifact phải tồn tại — báo đúng file thiếu, không chỉ lỗi IO."""
    path = settings.resolve(path_str)
    if not path.is_file():
        raise ArtifactError(f"Không tìm thấy {what}: {path}")
    return path


def build_state(settings: Settings) -> AppState:
    """Nạp ma trận + điểm phòng + checkpoint .pt + catalog PostgreSQL.

    Không có bước train nào. Checkpoint được đọc từ `MODEL_PATH` và tự nhận diện
    YOLO/RT-DETR.
    """
    state = AppState(settings=settings)

    # --- 1) Ma trận đồng xuất hiện (Cell 10.3/10.5) ---
    cooc_path = _require_file(
        settings, settings.cooccurrence_matrix_path, "ma trận đồng xuất hiện"
    )
    cooc = load_cooccurrence(cooc_path)
    state.cooccurrence_source = (
        f"{cooc_path.name} ({len(cooc['classes'])} class, "
        f"{cooc['n_images']} ảnh train)"
    )

    # --- 2) W(c, r) từ artifact, ghép tên nhóm -> mã qua DB ---
    scores_path = _require_file(
        settings, settings.category_room_scores_path, "file điểm phòng"
    )
    score_table: RoomScoreTable = load_room_scores(scores_path)
    state.room_scores_source = f"artifact {score_table.source}"

    # --- 3) Catalog PostgreSQL (không fallback) ---
    cfg = db_config(settings)
    name_to_code = load_category_names(cfg, settings.db_schema)
    room_scores = score_table.bind(name_to_code)
    state.room_scores_source = (
        f"artifact {score_table.source} + {settings.db_schema}.categories"
    )

    catalog = load_catalog_from_db(cfg, settings.db_schema)
    state.catalog = catalog
    state.catalog_source = catalog.source

    if settings.default_room_type not in ROOM_CODES:
        raise ArtifactError(
            f"DEFAULT_ROOM_TYPE='{settings.default_room_type}' không có trong "
            f"danh sách phòng {list(ROOM_CODES)}."
        )

    state.recommender = Recommender(
        cooc,
        catalog,
        room_scores=room_scores,
        alpha=settings.alpha_smooth,
        smooth_k=settings.smooth_k,
        default_room_type=settings.default_room_type,
        default_top_k=settings.default_top_k,
        default_min_score=settings.default_min_score,
        default_products_per_group=settings.products_per_group,
    )

    # --- 4) Checkpoint .pt ---
    model_path = _require_file(settings, settings.model_path, "checkpoint model")
    detector = Detector(model_path, settings.model_device, settings.detect_jpeg_roundtrip)
    detector.load()

    # Cell 9: checkpoint phải khớp bộ class của ma trận — nếu lệch thì `P_cooc`
    # tra sai cột mà không báo gì. Đối chiếu với ma trận (đã có sẵn) thay vì
    # annotation COCO, nên không cần `data/coco` khi chạy.
    if settings.require_class_match:
        detector.check_classes(cooc["classes"])
        state.detector_classes = list(cooc["classes"])
    detector.warmup()

    state.detector = detector
    state.model_family = detector.family
    state.model_source = f"{model_path.name} ({detector.family})"
    logger.info(
        "startup complete",
        extra={
            "event": "startup",
            "model_family": detector.family,
            "classes": len(detector.class_names),
            "catalog_source": catalog.source,
            "catalog_groups": len(catalog.groups),
            "room_scores_source": state.room_scores_source,
            "cooccurrence_source": state.cooccurrence_source,
        },
    )
    return state
