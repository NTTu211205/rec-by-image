"""Tầng 0 — detector.

Hỗ trợ:
* `rfdetr`: Roboflow RF-DETR (ví dụ: `Roboflow/rf-detr-medium` hoặc local directory) qua Hugging Face transformers.
"""

from __future__ import annotations

import logging
import threading
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image

from app.core.exceptions import ArtifactError, ModelNotReadyError

logger = logging.getLogger("app.detector")

# Bảng ánh xạ nhãn COCO của RF-DETR sang tên class nội thất trong hệ thống gợi ý
COCO_TO_ROOM_MAP: dict[str, str] = {
    "chair": "Chair",
    "couch": "Sofa",
    "sofa": "Sofa",
    "dining table": "Table",
    "table": "Table",
    "tv": "TV",
    "television": "TV",
    "laptop": "Laptop",
    "mouse": "Mouse",
    "remote": "Remote",
    "keyboard": "Keyboard",
    "cell phone": "Phone",
    "phone": "Phone",
    "book": "Books",
    "books": "Books",
    "clock": "Clock",
    "vase": "Vase",
    "potted plant": "Plant",
    "plant": "Plant",
    "bed": "Bed",
    "bench": "Bench/Ottoman",
    "microwave": "Microwave",
    "oven": "Oven",
    "toaster": "Toaster",
    "sink": "Sink",
    "refrigerator": "Refrigerator",
}


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


def _peek_architecture(path: Path | str) -> str:
    """Xác định kiến trúc model: 'rfdetr' | 'rtdetr' | 'yolo'."""
    str_path = str(path)

    # 1. Nhận diện RF-DETR qua tên hoặc cấu trúc thư mục
    if "rf-detr" in str_path.lower() or "rfdetr" in str_path.lower():
        return "rfdetr"

    p = Path(path)
    if p.is_dir() and ((p / "model.safetensors").is_file() or (p / "config.json").is_file()):
        return "rfdetr"

    # Nếu không phải file tồn tại nhưng có dạng org/repo trên Hugging Face
    if not p.is_file():
        if "/" in str_path and not str_path.startswith((".", "/")):
            return "rfdetr"
        raise ArtifactError(f"Không tìm thấy checkpoint RF-DETR: {str_path}")

    if p.suffix == ".pt":
        raise ArtifactError(
            f"Checkpoint {p.name} thuộc định dạng cũ (Ultralytics .pt). "
            "Dự án đã chuẩn hóa sang RF-DETR safetensors (Apache-2.0)."
        )

    return "rfdetr"


class Detector:
    """Bọc model object detection (RF-DETR hoặc Ultralytics). Một model dùng chung -> predict tuần tự."""

    def __init__(self, model_path: Path | str, device: str = "auto", jpeg_roundtrip: bool = True):
        self.model_path = model_path
        self.device = resolve_device(device)
        self.jpeg_roundtrip = jpeg_roundtrip
        self.family: str | None = None
        self._model: Any = None
        self._processor: Any = None
        self._torch_device: Any = None
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def class_names(self) -> list[str]:
        if not self.loaded:
            return []
        if self.family == "rfdetr":
            id2label = getattr(self._model.config, "id2label", {})
            return [v for k, v in sorted(id2label.items()) if v and v != "N/A"]
        return [self._model.names[i] for i in sorted(self._model.names)]

    def load(self) -> None:
        self.family = _peek_architecture(self.model_path)
        try:
            if self.family == "rfdetr":
                import torch
                from transformers import AutoImageProcessor, AutoModelForObjectDetection

                self._processor = AutoImageProcessor.from_pretrained(str(self.model_path))
                self._model = AutoModelForObjectDetection.from_pretrained(str(self.model_path))

                if self.device in ("0", "cuda", "cuda:0"):
                    self._torch_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                elif self.device == "mps":
                    mps_avail = getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
                    self._torch_device = torch.device("mps" if mps_avail else "cpu")
                else:
                    self._torch_device = torch.device("cpu")

                self._model.to(self._torch_device)
                self._model.eval()
            else:
                raise ArtifactError(f"Kiến trúc không được hỗ trợ: {self.family}")
        except Exception as exc:  # noqa: BLE001
            raise ArtifactError(
                f"Không nạp được model ({self.family}): {type(exc).__name__}: {exc}"
            ) from exc

        logger.info(
            "Đã nạp model",
            extra={"event": "model_loaded", "family": self.family, "classes": len(self.class_names)},
        )

    def check_classes(self, expected: list[str]) -> None:
        """Kiểm tra đối chiếu bộ class. Với RF-DETR pretrained COCO thì bỏ qua kiểm tra khớp cứng."""
        if self.family == "rfdetr":
            logger.info("RF-DETR pretrained COCO: bỏ qua kiểm tra khớp cứng 45 class nội thất.")
            return

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
        """Chạy detector, gộp nhãn trùng giữ confidence cao nhất."""
        if self._model is None:
            raise ModelNotReadyError("Model chưa được nạp.")

        # Tái tạo bước nén JPEG trong RAM nếu được yêu cầu
        if self.jpeg_roundtrip:
            buf = BytesIO()
            img.save(buf, format="JPEG")
            img = Image.open(BytesIO(buf.getvalue())).convert("RGB")
        else:
            img = img.convert("RGB")

        if self.family == "rfdetr":
            import torch

            width, height = img.size
            inputs = self._processor(images=img, return_tensors="pt")
            inputs = {k: v.to(self._torch_device) for k, v in inputs.items()}

            with self._lock:
                with torch.no_grad():
                    outputs = self._model(**inputs)

            results = self._processor.post_process_object_detection(
                outputs, target_sizes=[(height, width)], threshold=confidence
            )[0]

            best: dict[str, dict] = {}
            for score, label_id, box in zip(results["scores"], results["labels"], results["boxes"]):
                conf = float(score.item())
                raw_label = self._model.config.id2label.get(int(label_id.item()), f"class_{label_id.item()}")
                if raw_label == "N/A":
                    continue
                cls_name = COCO_TO_ROOM_MAP.get(raw_label.lower(), raw_label)
                xyxy = [float(round(v.item(), 2)) for v in box]
                if cls_name not in best or conf > best[cls_name]["confidence"]:
                    best[cls_name] = {"label": cls_name, "confidence": conf, "bbox": xyxy}
            return list(best.values())

        # Ultralytics RT-DETR / YOLO
        with self._lock:
            results = self._model.predict(
                source=img, conf=confidence, device=self.device, verbose=False
            )
        result = results[0]

        best = {}
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            cls_name = self._model.names.get(cls_id, f"class_{cls_id}")
            xyxy = [float(round(v, 2)) for v in box.xyxy[0].tolist()]
            if cls_name not in best or conf > best[cls_name]["confidence"]:
                best[cls_name] = {"label": cls_name, "confidence": conf, "bbox": xyxy}
        return list(best.values())
