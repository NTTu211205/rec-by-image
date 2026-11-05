"""Tầng xếp hạng gợi ý: P_cooc(c | D) + W(c, r), rồi gắn sản phẩm.

Đơn vị xếp hạng là **group_code** (`tv`, `speaker`, `air_conditioner`, ...) — trùng
`mwg.product.category_id`. Tên hiển thị tiếng Việt lấy từ `group_scores.GROUP_NAMES`.

Công thức giữ nguyên notebook v5 (Cell 10.4 / 10.9 / 10.10):

    P̂(d_target | d_source) = (N(d_source ∩ d_target) + α·P(d_target)) / (N(d_source) + α·K)
    P_cooc(c | D)          = 1 - Π_{d∈D} (1 - P̂(π(c) | d))
    final_score            = P_cooc(c | D) + W(c, r)

Nguồn dữ liệu (không có bảng dự phòng trong code):

1. `W(c, r)` — `artifacts/matrix/category_room_scores.json`, nạp bởi
   `app/services/room_scores.py` rồi ghép tên -> mã qua `mwg.categories`.
2. Ma trận đồng xuất hiện — `artifacts/matrix/cooccurrence_matrix.json`.
3. Sản phẩm — PostgreSQL schema `mwg` (`catalog.products`).

`GROUP_TO_DETECTOR` phủ 22 nhóm: nhóm không có detector proxy thì `P_cooc = 0`
và xếp hạng hoàn toàn bằng `W(c, r)` — giống cách notebook xử lý
`GROUPS_WITHOUT_DETECTOR`.
"""

from __future__ import annotations

import logging
import random
from typing import Any, Mapping, Sequence

from app.services import group_scores
from app.services.product_catalog import Catalog
from app.services.room_scores import ROOM_CODES

logger = logging.getLogger("app.recommender")

# ---------------------------------------------------------------------------
# Nhóm catalog -> class detector (Cell 10.2).
#
# Detector của repo này là RT-DETR huấn luyện trên ảnh nội thất COCO (45 class:
# TV, Speaker, Laptop, Monitor, Fan, Air Conditioner, ...), KHÔNG phải detector
# đồ điện máy. Vì vậy chỉ nhóm có proxy hợp lý mới được cộng P_cooc; phần còn
# lại dựa hoàn toàn vào W(c, r). Mọi giá trị dưới đây đều có trong
# `artifacts/matrix/cooccurrence_matrix.json`.
# ---------------------------------------------------------------------------
GROUP_TO_DETECTOR: dict[str, str] = {
    # MWGShop category ID mappings
    "38": "TV",                           # Tivi (cũ) / Tủ lạnh (mới)
    "37": "TV",                           # Tivi (mới)
    "147": "TV",                          # Máy chiếu (mới)
    "1498": "TV",                         # Máy chiếu & Màn hình chiếu (cũ)
    "980": "Entertainment Center",        # Tivi box & Đầu thu kỹ thuật số
    "132": "Speaker",                     # Loa (mới)
    "1933": "Speaker",                    # Loa (cũ)
    "1536": "Speaker",                    # Tai nghe nhét tai & chụp tai
    "2555": "Speaker",                    # Micro karaoke
    "28": "Laptop",                       # Laptop (mới)
    "1772": "Laptop",                     # Laptop (cũ)
    "72": "Laptop",                       # Máy tính để bàn (mới)
    "1481": "Laptop",                     # Bộ Máy Tính Bàn (cũ)
    "30": "Laptop",                       # Máy tính bảng (mới)
    "2116": "Laptop",                     # Máy tính bảng (cũ)
    "73": "Monitor",                      # Màn hình máy tính (mới)
    "1763": "Monitor",                    # Màn hình máy tính (cũ)
    "36": "Air Conditioner",              # Máy lạnh (mới)
    "1082": "Air Conditioner",            # Máy lạnh (cũ)
    "1083": "Air Conditioner",            # Máy sưởi
    "198": "Fan",                         # Quạt (mới)
    "59": "Fan",                          # Quạt (cũ)
    "69": "Fan",                          # Quạt mini (mới)
    "170": "Fan",                         # Quạt sưởi (mới)
    "199": "Fan",                         # Quạt điều hoà (mới)
    "200": "Fan",                         # Quạt trần (mới)
    "1215": "Fan",                        # Quạt Cầm Tay (cũ)
    "152": "Light",                       # Đèn điện, đèn sạc (mới)
    "297": "Light",                       # Đèn thông minh (mới)
    "301": "Light",                       # Đèn năng lượng mặt trời (mới)
    "317": "Light",                       # Bóng đèn led (mới)
    "2066": "Light",                      # Đèn & Đèn Pin (cũ)
    "253": "Power Surge Protector",       # Ổ cắm - Phích cắm (mới)
    "148": "Power Surge Protector",       # Ổ cắm, bóng đèn thông minh (mới)
    "1510": "Power Surge Protector",      # Ổ cắm điện & Dây nối (cũ)
    "257": "Remote",                      # Điều khiển Tivi (mới)
    "127": "Remote",                      # Camera giám sát (mới)
    "129": "Remote",                      # Camera hành trình / hành động (mới)
    "320": "Remote",                      # Chuông cửa camera (mới)
    "2539": "Remote",                     # Điều khiển tivi (cũ)
    "2056": "Remote",                     # Camera giám sát kết nối internet (cũ)
    "1341": "Remote",                     # Camera hành trình & Camera lùi (cũ)
    "75": "Printer",                      # Máy in (mới)
    "1487": "Printer",                    # Máy In, Máy Scan & Máy Photo (cũ)
    "27": "Phone",                        # Điện thoại (mới)
    "300": "Phone",                       # Điện thoại (cũ)
    "1129": "Organizer Storage",          # Hộp đựng, giỏ đựng đồ
    "941": "Rug",                         # Thảm trải sàn
    "166": "Basket/Bin",                  # Robot hút bụi (mới)
    "1709": "Basket/Bin",                 # Bộ & Cây lau nhà (cũ)
    "333": "Chair",                       # Ghế gaming, ghế công thái học (mới)
    "2671": "Chair",                      # Ghế gaming, ghế công thái học (cũ)
    "2662": "Accent Chair",               # Ghế Massage
    "1844": "Pillow",                     # Gối, nệm
    "913": "Clock",                       # Cân sức khỏe và phân tích cơ thể

    # Legacy group codes
    "tv": "TV",
    "projector": "TV",
    "speaker": "Speaker",
    "karaoke_speaker": "Speaker",
    "monitor": "Monitor",
    "laptop": "Laptop",
    "desktop_pc": "Laptop",
    "game_console": "Entertainment Center",
    "indoor_camera": "Remote",
    "outdoor_camera": "Remote",
    "air_conditioner": "Air Conditioner",
    "ceiling_fan": "Fan",
    "fan": "Fan",
    "air_cooler": "Fan",
    "heater_fan": "Fan",
    "lamp": "Floor Lamp",
    "led_bulb": "Light",
    "storage_box": "Organizer Storage",
    "storage_rack": "Shelf",
    "trash_bin": "Basket/Bin",
    "power_strip": "Power Surge Protector",
}

# Detector -> nhóm cần loại khỏi gợi ý vì đã có trong ảnh (Cell 10.2).
DETECTOR_TO_GROUP: dict[str, list[str]] = {
    "TV": ["37", "38", "147", "1498", "tv", "projector"],
    "Laptop": ["28", "30", "72", "1772", "1481", "2116", "laptop", "desktop_pc"],
    "Monitor": ["73", "1763", "monitor"],
    "Air Conditioner": ["36", "1082", "1083", "air_conditioner"],
    "Speaker": ["132", "1933", "1536", "2555", "speaker", "karaoke_speaker"],
    "Power Surge Protector": ["148", "253", "1510", "power_strip"],
    "Shelf": ["storage_rack"],
    "Basket/Bin": ["166", "1709", "trash_bin"],
    "Fan": ["198", "59", "69", "170", "199", "200", "1215", "fan", "ceiling_fan", "air_cooler", "heater_fan"],
    "Floor Lamp": ["lamp"],
    "Light": ["152", "297", "301", "317", "2066", "led_bulb"],
    "Organizer Storage": ["1129", "storage_box"],
    "Remote": ["257", "127", "129", "320", "2539", "2056", "1341", "indoor_camera", "outdoor_camera"],
    "Entertainment Center": ["980", "game_console"],
    "Printer": ["75", "1487"],
    "Phone": ["27", "300"],
    "Rug": ["941"],
    "Chair": ["333", "2671"],
    "Accent Chair": ["2662"],
    "Pillow": ["1844"],
    "Clock": ["913"],
}

# Nhóm không có detector tương ứng — P_cooc luôn 0, chỉ xếp theo W(c, r).
GROUPS_WITHOUT_DETECTOR: frozenset[str] = frozenset(
    set(group_scores.GROUP_NAMES) - set(GROUP_TO_DETECTOR)
)


class Recommender:
    """Bất biến sau khi tạo, an toàn khi dùng đa luồng."""

    def __init__(
        self,
        cooc: Mapping[str, Any],
        catalog: Catalog,
        *,
        room_scores: Mapping[str, Mapping[str, float]],
        alpha: float = 1.0,
        smooth_k: int = 2,
        default_room_type: str = "living_room",
        default_top_k: int = 10,
        default_min_score: float = 0.05,
        default_products_per_group: int = 3,
    ):
        self.classes: list[str] = list(cooc["classes"])
        self.matrix: list[list[float]] = cooc["matrix"]
        self.support: dict[str, int] = cooc["support"]
        self.counts: dict[str, dict[str, int]] = cooc["counts"]
        self.base_rate: dict[str, float] = cooc["base_rate"]
        self.n_images: int = cooc["n_images"]

        self.catalog = catalog
        # W(c, r) do `bootstrap` nạp từ artifact rồi truyền vào — Recommender
        # không tự đọc file/database để dùng lại được với bảng khác trong test.
        self.room_scores: dict[str, dict[str, float]] = {
            code: dict(scores) for code, scores in room_scores.items()
        }
        self.alpha = alpha
        self.smooth_k = smooth_k
        self.default_room_type = default_room_type
        self.default_top_k = default_top_k
        self.default_min_score = default_min_score
        self.default_products_per_group = default_products_per_group
        self._index = {c: i for i, c in enumerate(self.classes)}

        # Ứng viên = nhóm có trong catalog VÀ có điểm phòng; tính một lần.
        self.candidate_groups: list[str] = sorted(
            code for code in self.room_scores if code in catalog.groups
        )
        missing = sorted(set(self.catalog.groups) - set(self.room_scores))
        if missing:
            logger.warning(
                "Nhóm có sản phẩm nhưng chưa có W(c, r) nên không được gợi ý: %s",
                ", ".join(missing),
                extra={"event": "groups_without_scores"},
            )

    @property
    def room_types(self) -> list[str]:
        return list(ROOM_CODES)

    # ---- Tầng 1: P̂(d_target | d_source) ----
    def p_smoothed(
        self, d_source: str, d_target: str, alpha: float | None = None, k: int | None = None
    ) -> float:
        """P̂(d_target | d_source) = (N(d_source ∩ d_target) + α·P(d_target)) / (N(d_source) + α·K)."""
        alpha = self.alpha if alpha is None else alpha
        k = self.smooth_k if k is None else k
        if d_source not in self._index or d_target not in self._index:
            return 0.0
        n_source = self.support.get(d_source, 0)
        if n_source == 0:
            return self.base_rate.get(d_target, 0.0)
        if alpha <= 0:
            return self.matrix[self._index[d_source]][self._index[d_target]]
        n_both = self.counts[d_source][d_target]
        prior = self.base_rate.get(d_target, 0.0)
        return (n_both + alpha * prior) / (n_source + alpha * k)

    # ---- Tầng 2: W(c, r) ----
    def w(self, group_code: str, room_type: str) -> float:
        """W(c, r) — đọc bảng điểm nạp từ `artifacts/matrix/category_room_scores.json`."""
        return float(self.room_scores.get(group_code, {}).get(room_type, 0.0))

    # ---- Tầng 3: P_cooc(c | D) ----
    def cooccurrence_score(
        self, detected_labels: Sequence[str], group_code: str, alpha: float | None = None
    ) -> float:
        """P_cooc(c | D) = 1 - Π_{d∈D} (1 - P̂(π(c) | d)); 0 nếu nhóm không có proxy."""
        if not detected_labels:
            return 0.0
        proxy = GROUP_TO_DETECTOR.get(group_code)
        if proxy is None:
            return 0.0
        not_present = 1.0
        for label in detected_labels:
            not_present *= 1.0 - self.p_smoothed(label, proxy, alpha)
        return 1.0 - not_present

    def build_reason(
        self,
        group_code: str,
        detected_labels: Sequence[str],
        room_score: float,
        cooc_score: float = 0.0,
    ) -> str:
        """Câu giải thích: ưu tiên nêu vật thể trong ảnh nếu có đóng góp P_cooc.

        Nêu cả `cooc_score` (Noisy-OR tổng hợp trên mọi vật thể) chứ không chỉ
        `P̂(π(c) | best)` của một vật thể, để con số trong câu khớp với
        `cooccurrence_score` mà response trả về.
        """
        name = self.catalog.category_names.get(group_code) or group_scores.group_name(group_code)
        proxy = GROUP_TO_DETECTOR.get(group_code)
        if not detected_labels or proxy is None:
            return f"{name} phù hợp với phòng đang xét (điểm phòng {room_score:.1f})."

        best_label, best_p = None, 0.0
        for label in detected_labels:
            if label == proxy:
                continue
            p = self.p_smoothed(label, proxy)
            if p > best_p:
                best_label, best_p = label, p

        if best_label is None or best_p <= 0 or cooc_score <= 0:
            return f"{name} phù hợp với phòng đang xét (điểm phòng {room_score:.1f})."

        return (
            f"Ảnh có {best_label} — thường đi kèm {name} "
            f"(kết hợp các vật thể: {cooc_score:.0%}); điểm phòng {room_score:.1f}."
        )

    # ---- Xếp hạng ----
    def recommend(
        self,
        detected_objects: Sequence[dict[str, Any]],
        *,
        room_type: str | None = None,
        top_k: int | None = None,
        min_score: float | None = None,
        exclude_detected: bool = True,
        products_per_group: int | None = None,
        price_range: tuple[float, float] | None = None,
        alpha: float | None = None,
        category_id: str | None = None,
    ) -> dict[str, Any]:
        """Ghép P_cooc với W(c, r), lọc, xếp hạng, rồi gắn sản phẩm (Cell 10.10)."""
        room_type = room_type or self.default_room_type
        top_k = self.default_top_k if top_k is None else top_k
        min_score = self.default_min_score if min_score is None else min_score
        products_per_group = (
            self.default_products_per_group if products_per_group is None else products_per_group
        )

        detected_labels = [o["label"] for o in detected_objects]

        already_present: set[str] = set()
        if exclude_detected:
            for label in detected_labels:
                already_present.update(DETECTOR_TO_GROUP.get(label, ()))

        candidates: list[dict[str, Any]] = []
        for group_code in self.candidate_groups:
            if group_code in already_present:
                continue
            if (
                category_id is not None
                and self.catalog.category_ids.get(group_code) != category_id
                and group_code != category_id
                and group_scores.GROUP_CATEGORY.get(group_code) != category_id
            ):
                continue

            cooc = self.cooccurrence_score(detected_labels, group_code, alpha)
            room_score = self.w(group_code, room_type)
            final_score = cooc + room_score
            if final_score < min_score:
                continue

            candidates.append({
                "group_code": group_code,
                "group_name": self.catalog.category_names.get(group_code) or group_scores.group_name(group_code),
                "category_id": self.catalog.category_ids.get(group_code) or group_code,
                "cooccurrence_score": cooc,
                "room_score": room_score,
                "final_score": final_score,
                "product_count": self.catalog.counts.get(group_code, 0),
                "reason": self.build_reason(group_code, detected_labels, room_score, cooc),
            })

        # Sắp giảm dần theo `final_score`. Điểm phòng gán cứng chỉ có ~11 mức
        # (0.0 → 1.0 bước 0.1) nên rất nhiều nhóm đồng điểm; tie-break theo
        # P_cooc rồi số sản phẩm để kết quả ổn định và có ý nghĩa (nhóm nhiều
        # lựa chọn hơn đứng trước) thay vì rơi về thứ tự bảng chữ cái.
        candidates.sort(
            key=lambda c: (
                c["final_score"],
                c["cooccurrence_score"],
                c["product_count"],
                c["group_code"],
            ),
            reverse=True,
        )

        # Gắn sản phẩm rồi CHỈ nhận nhóm còn ít nhất một sản phẩm thoả lọc.
        # Nhóm có `products: []` là vô nghĩa với người dùng (nhất là khi lọc
        # `min_price`/`max_price`), nên bỏ qua và lấy tiếp ứng viên phía sau để
        # vẫn đủ `top_k`.
        top: list[dict[str, Any]] = []
        for item in candidates:
            products = list(self.catalog.available.get(item["group_code"], ()))
            if price_range is not None:
                lo, hi = price_range
                products = [p for p in products if lo <= p["price"] <= hi]
            products = [p for p in products if (p.get("image_url") or "").strip()]
            if not products:
                continue
            # Lấy ngẫu nhiên sản phẩm từ mỗi nhóm/category
            k = min(len(products), products_per_group)
            item["products"] = random.sample(products, k)
            top.append(item)
            if len(top) >= top_k:
                break

        return {
            "room_type": room_type,
            "detected_objects": [dict(o) for o in detected_objects],
            "recommendations": top,
        }
