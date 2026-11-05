"""Cell 10.12: nạp ảnh từ bytes / đường dẫn / URL, kèm kiểm tra an toàn."""

from __future__ import annotations

import io
import logging

from PIL import Image, UnidentifiedImageError

from app.core.config import ALLOWED_IMAGE_FORMATS
from app.core.exceptions import ImageError

logger = logging.getLogger("app.images")

# Chữ ký nhận dạng nhanh theo magic bytes — tránh phải nhờ Pillow đoán.
_MAGIC: tuple[tuple[bytes, str], ...] = (
    (b"\xff\xd8\xff", "JPEG"),
    (b"\x89PNG\r\n\x1a\n", "PNG"),
    (b"RIFF", "WEBP"),  # kiểm tra thêm "WEBP" ở offset 8
)


def _sniff_format(data: bytes) -> str | None:
    for magic, fmt in _MAGIC:
        if not data.startswith(magic):
            continue
        if fmt == "WEBP" and data[8:12] != b"WEBP":
            continue
        return fmt
    return None


def load_image_bytes(data: bytes, *, max_pixels: int, min_side: int) -> Image.Image:
    """Mở ảnh từ bytes. Kiểm tra định dạng thật, không tin content-type."""
    if not data:
        raise ImageError("Ảnh rỗng.")

    # Ultralytics patch `PIL.Image.open` toàn cục: khi gặp bytes không nhận dạng được
    # nó cố `pip install pi-heif` rồi ném lỗi không phải `UnidentifiedImageError`.
    # Sniff magic bytes trước để file lạ bị từ chối ở đây, không lọt vào nhánh đó.
    sniffed = _sniff_format(data)
    if sniffed is None:
        raise ImageError(
            "Dữ liệu không phải ảnh hợp lệ. "
            f"Chấp nhận: {', '.join(sorted(ALLOWED_IMAGE_FORMATS))}."
        )

    try:
        img = Image.open(io.BytesIO(data))
        fmt = (img.format or "").upper()
        img.load()  # giải mã ngay để lỗi ảnh hỏng lộ ra ở đây, không phải lúc convert
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
        raise ImageError("Dữ liệu không phải ảnh hợp lệ.") from exc
    except Exception as exc:  # noqa: BLE001 - nhánh pi-heif của ultralytics
        logger.warning(
            "Pillow không mở được ảnh: %s", type(exc).__name__,
            extra={"event": "image_open_failed"},
        )
        raise ImageError("Dữ liệu không phải ảnh hợp lệ.") from exc

    if fmt not in ALLOWED_IMAGE_FORMATS:
        raise ImageError(
            f"Định dạng {fmt or 'không xác định'} không hỗ trợ. "
            f"Chấp nhận: {', '.join(sorted(ALLOWED_IMAGE_FORMATS))}."
        )

    width, height = img.size
    if width * height > max_pixels:
        raise ImageError(f"Ảnh quá lớn ({width}×{height}); giới hạn {max_pixels:,} điểm ảnh.")
    if min(width, height) < min_side:
        raise ImageError(f"Ảnh quá nhỏ ({width}×{height}); cạnh ngắn nhất phải ≥ {min_side}px.")

    return img.convert("RGB")
