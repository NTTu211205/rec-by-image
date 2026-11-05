"""Tầng 0 — detector. Cell 9 + Cell 10.13 của notebook (KHÔNG có Cell 8).

Cell 8 (train) không được port. Model chỉ được **đọc từ file `.pt`**:

* `RTDETRDetectionModel` (sản phẩm của Cell 8 trong notebook RT-DETR) -> `RTDETR`
* `DetectionModel` (YOLO cũ) -> `YOLO`

Kiến trúc được nhận diện bằng cách mở checkpoint trước, nên cùng một
`MODEL_PATH` dùng được cho cả hai phiên bản mà không cần cấu hình thêm.
"""

from __future__ import annotations

import logging
import threading
from io import BytesIO
from pathlib import Path

from PIL import Image

from app.core.exceptions import ArtifactError, ModelNotReadyError

logger = logging.getLogger("app.detector")


def resolve_device(requested: str) -> str:
    """auto -> cuda(0) > mps > cpu. Giá trị khác giữ nguyên."""
    if requested != "auto":
        return requested
    try:
        import torch

        if torch.cuda.is_available():
            return "0"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:  # noqa: BLE001
        pass
    return "cpu"


def _peek_architecture(path: Path) -> str:
    """Đọc tên class của module trong checkpoint: 'rtdetr' | 'yolo'."""
    try:
        import torch

        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001
        raise ArtifactError(f"Không đọc được checkpoint: {path.name}") from exc

    if not isinstance(ckpt, dict):
        raise ArtifactError(f"Checkpoint không phải dict: {type(ckpt).__name__}")

    # Ultralytics lưu nhiều slot: `model` (weights hiện tại), `ema` (bản EMA của lần
    # train cuối). Checkpoint RT-DETR đã train thường chỉ còn `ema`, hoặc `model`
    # là None — phải thử lần lượt thay vì chỉ đọc `model`.
    for key in ("model", "ema"):
        model = ckpt.get(key)
        if model is None:
            continue
        name = type(model).__name__
        if name == "RTDETRDetectionModel":
            return "rtdetr"
        if name == "DetectionModel":
            return "yolo"

    raise ArtifactError(
        "Checkpoint không phải YOLO/RT-DETR: "
        f"model={type(ckpt.get('model')).__name__}, ema={type(ckpt.get('ema')).__name__}"
    )


class Detector:
    """Bọc model Ultralytics. Một model dùng chung -> predict phải tuần tự."""

    def __init__(self, model_path: Path, device: str = "auto", jpeg_roundtrip: bool = True):
        self.model_path = model_path
        self.device = resolve_device(device)
        self.jpeg_roundtrip = jpeg_roundtrip
        self.family: str | None = None
        self._model = None
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def class_names(self) -> list[str]:
        if not self.loaded:
            return []
        return [self._model.names[i] for i in sorted(self._model.names)]

    def load(self) -> None:
        if not self.model_path.is_file():
            raise ArtifactError(f"Không tìm thấy file weights: {self.model_path.name}")

        self.family = _peek_architecture(self.model_path)
        try:
            if self.family == "rtdetr":
                from ultralytics import RTDETR as ModelClass
            else:
                from ultralytics import YOLO as ModelClass
            self._model = ModelClass(str(self.model_path))
        except Exception as exc:  # noqa: BLE001
            raise ArtifactError(
                f"Không nạp được model ({self.family}): {type(exc).__name__}"
            ) from exc

        logger.info(
            "Đã nạp model",
            extra={"event": "model_loaded", "family": self.family, "classes": len(self._model.names)},
        )

    def check_classes(self, expected: list[str]) -> None:
        """Cell 9: checkpoint phải có đúng bộ class như ma trận đồng xuất hiện."""
        actual = set(self.class_names)
        if actual != set(expected):
            missing = sorted(set(expected) - actual)
            extra = sorted(actual - set(expected))
            raise ArtifactError(
                "Class của checkpoint không khớp ma trận đồng xuất hiện "
                f"(thiếu {missing[:5]}, thừa {extra[:5]})."
            )

    def warmup(self) -> None:
        if self.loaded:
            self.detect(Image.new("RGB", (64, 64)), confidence=0.99)

    def detect(self, img: Image.Image, confidence: float) -> list[dict]:
        """Cell 10.13: chạy detector, gộp nhãn trùng giữ confidence cao nhất."""
        if self._model is None:
            raise ModelNotReadyError("Model chưa được nạp.")

        # Notebook lưu ảnh ra .jpg rồi predict; tái tạo trong RAM để khớp kết quả.
        if self.jpeg_roundtrip:
            buf = BytesIO()
            img.save(buf, format="JPEG")
            img = Image.open(BytesIO(buf.getvalue())).convert("RGB")

        with self._lock:
            results = self._model.predict(
                source=img, conf=confidence, device=self.device, verbose=False
            )
        result = results[0]

        best: dict[str, dict] = {}
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            cls_name = self._model.names.get(cls_id, f"class_{cls_id}")
            xyxy = [float(round(v, 2)) for v in box.xyxy[0].tolist()]
            if cls_name not in best or conf > best[cls_name]["confidence"]:
                best[cls_name] = {"label": cls_name, "confidence": conf, "bbox": xyxy}
        return list(best.values())
