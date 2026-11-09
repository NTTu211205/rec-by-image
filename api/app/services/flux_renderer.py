"""FLUX.2 Klein 4B Generative Render & Harmonization Engine.

Tích hợp mô hình FLUX.2 [klein] 4B (Black Forest Labs - Apache 2.0)
vào quy trình phối cảnh không gian phòng:
- Nhận diện phối cảnh sơ bộ (geometric placement từ SAM 2 & Depth Anything V2).
- Sử dụng FLUX.2 Klein 4B để tái tạo ánh sáng tự nhiên (photorealistic relighting),
  khử viền cắt dán và tạo phản xạ, bóng đổ chân thực (contact shadow harmonization).
- Tự động fallback về photometric composite nếu không đủ VRAM hoặc thiếu thư viện.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, List, Optional

import torch
from PIL import Image

logger = logging.getLogger("app.flux_renderer")

_FLUX_PIPELINE = None
_FLUX_LOADED = False
_FLUX_LOAD_ATTEMPTED = False
_FLUX_LOAD_ERROR = None

ROOM_NAME_MAP = {
    "living_room": "phòng khách (living room)",
    "bedroom": "phòng ngủ (bedroom)",
    "kitchen": "nhà bếp (kitchen)",
    "dining_room": "phòng ăn (dining room)",
    "bathroom": "phòng tắm (bathroom)",
    "home_office": "phòng làm việc (home office)",
    "balcony": "ban công (balcony)",
}


def get_default_device() -> str:
    """Xác định device tối ưu cho FLUX.2 Klein."""
    env_dev = os.environ.get("FLUX_DEVICE", "auto").strip().lower()
    if env_dev in ("cuda", "mps", "cpu"):
        return env_dev
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_flux_klein_pipeline(
    model_id: str = "black-forest-labs/FLUX.2-klein-4B",
    device: Optional[str] = None,
) -> Any:
    """Tải và khởi tạo pipeline FLUX.2 Klein 4B với cơ chế lazy load."""
    global _FLUX_PIPELINE, _FLUX_LOADED, _FLUX_LOAD_ATTEMPTED, _FLUX_LOAD_ERROR

    if _FLUX_LOADED and _FLUX_PIPELINE is not None:
        return _FLUX_PIPELINE

    if _FLUX_LOAD_ATTEMPTED and _FLUX_LOAD_ERROR is not None:
        return None

    _FLUX_LOAD_ATTEMPTED = True
    device = device or get_default_device()

    # Kiểm tra an toàn thiết bị: Diffusion 4B không được chạy trên CPU trong production
    if device == "cpu":
        _FLUX_LOAD_ERROR = (
            "FLUX.2 Klein 4B yêu cầu GPU (CUDA) hoặc Apple Silicon (MPS). "
            "Hệ thống tự động sử dụng Photometric Composite để đảm bảo phản hồi tức thì."
        )
        logger.warning(_FLUX_LOAD_ERROR)
        return None

    # Kiểm tra xem có thư mục weights local tại api/artifacts/flux2_klein_4b không
    local_dir = Path(__file__).resolve().parents[2] / "artifacts" / "flux2_klein_4b"
    if not local_dir.is_dir():
        _FLUX_LOAD_ERROR = (
            "Chưa có thư mục trọng số api/artifacts/flux2_klein_4b trên đĩa. "
            "Để tránh nghẽn mạng và tải 12GB trong request, hệ thống sử dụng Photometric Composite."
        )
        logger.warning(_FLUX_LOAD_ERROR)
        return None

    target_model_id = str(local_dir)

    try:
        from diffusers import DiffusionPipeline  # type: ignore

        logger.info(
            "Đang khởi tạo FLUX.2 Klein 4B Pipeline từ '%s' trên thiết bị '%s'...",
            target_model_id,
            device,
        )
        t0 = time.perf_counter()

        dtype = torch.bfloat16 if device in ("cuda", "mps") else torch.float32

        pipe = DiffusionPipeline.from_pretrained(
            target_model_id,
            torch_dtype=dtype,
            local_files_only=True,
        )

        if device == "cuda":
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(device)

        _FLUX_PIPELINE = pipe
        _FLUX_LOADED = True
        logger.info(
            "Tải thành công FLUX.2 Klein 4B trong %.2f giây (device: %s, dtype: %s)",
            time.perf_counter() - t0,
            device,
            dtype,
        )
        return _FLUX_PIPELINE

    except ImportError as imp_err:
        _FLUX_LOAD_ERROR = f"Thiếu thư viện diffusers/accelerate: {imp_err}"
        logger.warning(
            "Chưa cài đặt 'diffusers' hoặc 'accelerate'. Bỏ qua FLUX.2 Klein 4B và sử dụng Photometric Composite."
        )
        return None
    except Exception as exc:
        _FLUX_LOAD_ERROR = f"Lỗi tải FLUX.2 Klein 4B: {exc}"
        logger.warning(
            "Không thể tải FLUX.2 Klein 4B (%s). Tự động fallback về Photometric Composite.",
            exc,
        )
        return None


def build_harmonization_prompt(
    room_type: str,
    placed_items: List[dict],
) -> str:
    """Tạo text prompt dẫn dắt FLUX.2 Klein hòa hợp ánh sáng và chất liệu đồ gia dụng."""
    room_desc = ROOM_NAME_MAP.get(room_type, room_type or "căn phòng")
    
    product_names = []
    for item in placed_items:
        pname = item.get("product_name") or item.get("group_name")
        if pname and pname not in product_names:
            product_names.append(pname)

    items_str = ", ".join(product_names[:4]) if product_names else "thiết bị điện máy gia dụng"

    prompt = (
        f"A masterfully shot, photorealistic interior photograph of a modern {room_desc}, "
        f"featuring seamlessly integrated {items_str}. "
        "Hyperrealistic studio lighting, natural contact shadows on the floor and walls, "
        "consistent color balance, soft ambient occlusion, high end architectural digest style, "
        "8k uhd, sharp details, no clipping, photorealistic material reflections."
    )
    return prompt


def refine_with_flux_klein(
    composite_image: Image.Image,
    room_type: str = "living_room",
    placed_items: Optional[List[dict]] = None,
    denoising_strength: float = 0.38,
    num_inference_steps: int = 4,
    guidance_scale: float = 2.5,
    model_id: str = "black-forest-labs/FLUX.2-klein-4B",
    device: Optional[str] = None,
) -> tuple[Image.Image, dict[str, Any]]:
    """Dùng FLUX.2 Klein 4B để hoàn thiện (harmonize / relight) ảnh phối cảnh phòng.

    Args:
        composite_image: Ảnh PIL đã qua định vị hình học và ghép vật thể.
        room_type: Loại phòng (living_room, bedroom,...).
        placed_items: Danh sách sản phẩm đã được đặt thành công.
        denoising_strength: Độ biến đổi (0.35 - 0.45 để giữ trọn vẹn kết cấu gốc).
        num_inference_steps: Số bước inference (mặc định 4 bước siêu nhanh của Klein).
        guidance_scale: Mức độ bám sát prompt.
        model_id: ID hoặc đường dẫn mô hình.
        device: Thiết bị chạy (auto/cuda/mps/cpu).

    Returns:
        tuple[Image.Image, dict]: (Ảnh hoàn thiện, thông tin chẩn đoán FLUX).
    """
    placed_items = placed_items or []
    pipe = load_flux_klein_pipeline(model_id=model_id, device=device)

    diagnostics = {
        "flux_applied": False,
        "model_id": model_id,
        "device": device or get_default_device(),
        "steps": num_inference_steps,
        "strength": denoising_strength,
        "error": _FLUX_LOAD_ERROR,
    }

    if pipe is None:
        return composite_image, diagnostics

    prompt = build_harmonization_prompt(room_type, placed_items)
    negative_prompt = (
        "blurry, artificial, 2d sticker, floating objects, wrong perspective, "
        "harsh cutoff, bad lighting, low quality, cartoon, distorted lines"
    )

    t0 = time.perf_counter()
    try:
        # Resize nhẹ nếu kích thước quá lớn để tăng tốc inference và tiết kiệm VRAM
        orig_size = composite_image.size
        work_img = composite_image.convert("RGB")
        max_dim = 1024
        if max(orig_size) > max_dim:
            scale = max_dim / max(orig_size)
            work_img = work_img.resize(
                (int(orig_size[0] * scale), int(orig_size[1] * scale)),
                Image.Resampling.LANCZOS,
            )

        logger.info(
            "Bắt đầu FLUX.2 Klein 4B Generative Harmonization (steps: %d, strength: %.2f)...",
            num_inference_steps,
            denoising_strength,
        )

        with torch.inference_mode():
            # Hỗ trợ image-to-image / multi-reference editing của FLUX.2 Klein
            result = pipe(
                prompt=prompt,
                image=work_img,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                strength=denoising_strength,
                negative_prompt=negative_prompt,
            )
            out_img = result.images[0]

        # Khôi phục kích thước ban đầu nếu đã resize
        if out_img.size != orig_size:
            out_img = out_img.resize(orig_size, Image.Resampling.LANCZOS)

        duration = time.perf_counter() - t0
        logger.info(
            "FLUX.2 Klein 4B hoàn tất làm mịn và hòa hợp ánh sáng thành công sau %.2f giây",
            duration,
        )

        diagnostics["flux_applied"] = True
        diagnostics["latency_s"] = round(duration, 3)
        diagnostics["prompt"] = prompt
        return out_img, diagnostics

    except Exception as exc:
        logger.error(
            "Lỗi trong quá trình chạy FLUX.2 Klein 4B: %s. Giữ nguyên ảnh composite ban đầu.",
            exc,
            exc_info=True,
        )
        diagnostics["error"] = str(exc)
        return composite_image, diagnostics

