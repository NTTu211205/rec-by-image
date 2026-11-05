"""Test tầng logic thuần Python — bảng W(c,r), Noisy-OR, xếp hạng.

Không cần model, không cần ảnh, không cần DB: dựng `Catalog` giả + ma trận đồng
xuất hiện nhỏ + bảng điểm phòng giả để kiểm tra công thức và thứ tự.
"""

from __future__ import annotations

import pytest

from app.services import group_scores
from app.services.product_catalog import Catalog
from app.services.room_scores import ROOM_CODES
from app.services.recommender import (
    DETECTOR_TO_GROUP,
    GROUPS_WITHOUT_DETECTOR,
    GROUP_TO_DETECTOR,
    Recommender,
)


def _catalog() -> Catalog:
    """Catalog nhỏ: 3 nhóm, nhóm `tv` có 2 sản phẩm (1 hết hàng)."""
    products = {
        "tv": [
            {"product_id": "T1", "product_name": "TV đắt", "brand": "A",
             "price": 9_000_000.0, "currency": "VND", "image_url": "",
             "product_url": "", "availability": True},
            {"product_id": "T2", "product_name": "TV rẻ", "brand": "B",
             "price": 3_000_000.0, "currency": "VND", "image_url": "",
             "product_url": "", "availability": True},
            {"product_id": "T3", "product_name": "TV hết hàng", "brand": "C",
             "price": 1_000_000.0, "currency": "VND", "image_url": "",
             "product_url": "", "availability": False},
        ],
        "speaker": [
            {"product_id": "S1", "product_name": "Loa", "brand": "D",
             "price": 2_000_000.0, "currency": "VND", "image_url": "",
             "product_url": "", "availability": True},
        ],
        "air_conditioner": [
            {"product_id": "A1", "product_name": "Điều hoà", "brand": "E",
             "price": 8_000_000.0, "currency": "VND", "image_url": "",
             "product_url": "", "availability": True},
        ],
    }
    return Catalog(products, source="fixture")


def _cooc() -> dict:
    """Ma trận 3 class: TV, Speaker, Air Conditioner. Mỗi class có 10 ảnh."""
    classes = ["TV", "Speaker", "Air Conditioner"]
    # P(TV|TV)=1, P(Speaker|TV)=0.4, P(AC|TV)=0.1 ...
    matrix = [
        [1.0, 0.4, 0.1],
        [0.5, 1.0, 0.1],
        [0.2, 0.2, 1.0],
    ]
    support = {c: 10 for c in classes}
    counts = {
        c: {d: int(round(matrix[i][j] * support[c])) for j, d in enumerate(classes)}
        for i, c in enumerate(classes)
    }
    return {
        "classes": classes,
        "matrix": matrix,
        "support": support,
        "counts": counts,
        "base_rate": {c: 10 / 50 for c in classes},
        "n_images": 50,
    }


def _room_scores() -> dict[str, dict[str, float]]:
    """Bảng điểm giả cho 3 nhóm trong `_catalog()` — đủ 7 phòng mỗi nhóm.

    Chỉ `living_room` khác 0 để dễ suy luận khi kiểm tra xếp hạng.
    """
    def row(living_room: float) -> dict[str, float]:
        return {room: 0.0 for room in ROOM_CODES} | {"living_room": living_room}

    return {
        "tv": row(0.9),
        "speaker": row(0.7),
        "air_conditioner": row(0.5),
    }


def _recommender(**kwargs) -> Recommender:
    kwargs.setdefault("room_scores", _room_scores())
    return Recommender(_cooc(), _catalog(), **kwargs)


# --- Metadata nhóm ----------------------------------------------------------

def test_group_metadata_complete() -> None:
    codes = set(group_scores.GROUP_NAMES)
    assert codes == set(group_scores.GROUP_CATEGORY)
    assert set(group_scores.GROUP_CATEGORY.values()) <= set(group_scores.CATEGORY_NAMES)


def test_detector_proxies_exist_in_matrix() -> None:
    """Mọi proxy phải có trong ma trận THẬT; nếu không P_cooc luôn 0 mà không báo lỗi."""
    from pathlib import Path

    from app.core.config import BASE_DIR
    from app.services.cooccurrence import load_cooccurrence

    path = Path(BASE_DIR) / "artifacts" / "matrix" / "cooccurrence_matrix.json"
    classes = set(load_cooccurrence(path)["classes"])
    for code, detector in GROUP_TO_DETECTOR.items():
        assert code in group_scores.GROUP_NAMES, code
        assert detector in classes, f"{code} -> {detector} không có trong ma trận"


def test_groups_without_detector() -> None:
    assert GROUPS_WITHOUT_DETECTOR == set(group_scores.GROUP_NAMES) - set(GROUP_TO_DETECTOR)
    assert "air_fryer" in GROUPS_WITHOUT_DETECTOR
    assert "tv" not in GROUPS_WITHOUT_DETECTOR


def test_detector_to_group_covers_proxies() -> None:
    """Mỗi proxy dùng để loại trùng phải ánh xạ ngược về chính nhóm đó."""
    for code, detector in GROUP_TO_DETECTOR.items():
        assert code in DETECTOR_TO_GROUP.get(detector, ()), f"{detector} thiếu {code}"


# --- Công thức --------------------------------------------------------------

def test_w_reads_injected_scores() -> None:
    rec = _recommender()
    assert rec.w("tv", "living_room") == 0.9
    assert rec.w("speaker", "living_room") == 0.7
    assert rec.w("khong_ton_tai", "living_room") == 0.0
    assert rec.w("tv", "garage") == 0.0


def test_room_types_from_contract() -> None:
    assert _recommender().room_types == list(ROOM_CODES)


def test_recommender_does_not_mutate_injected_scores() -> None:
    scores = _room_scores()
    Recommender(_cooc(), _catalog(), room_scores=scores)
    scores["tv"]["living_room"] = 0.0
    # Recommender đã copy — thay đổi dict bên ngoài không ảnh hưởng.
    assert _recommender().w("tv", "living_room") == 0.9


def test_p_smoothed_zero_for_unknown_class() -> None:
    rec = _recommender()
    assert rec.p_smoothed("Sofa", "TV") == 0.0
    assert rec.p_smoothed("TV", "KhongCo") == 0.0


def test_p_smoothed_laplace() -> None:
    """(N + α·P) / (N + α·K) với N=4, P=0.2, α=1, K=2 → (4+0.2)/12."""
    rec = _recommender(alpha=1.0, smooth_k=2)
    expected = (4 + 1.0 * 0.2) / (10 + 1.0 * 2)
    assert rec.p_smoothed("TV", "Speaker") == pytest.approx(expected)


def test_cooccurrence_noisy_or() -> None:
    """P_cooc = 1 - Π(1 - P̂(d | label)); nhiều vật thể không thể giảm điểm."""
    rec = _recommender()
    single = rec.cooccurrence_score(["TV"], "speaker")
    double = rec.cooccurrence_score(["TV", "Air Conditioner"], "speaker")
    assert single > 0
    assert double >= single
    assert rec.cooccurrence_score([], "speaker") == 0.0


def test_cooccurrence_zero_without_proxy() -> None:
    rec = _recommender()
    assert rec.cooccurrence_score(["TV"], "air_fryer") == 0.0


# --- Xếp hạng ---------------------------------------------------------------

def test_final_score_is_sum() -> None:
    rec = _recommender()
    out = rec.recommend([{"label": "Sofa", "confidence": 0.9}], room_type="living_room", top_k=5)
    for item in out["recommendations"]:
        assert item["final_score"] == pytest.approx(
            item["cooccurrence_score"] + item["room_score"], abs=1e-12
        )


def test_sorted_descending() -> None:
    rec = _recommender()
    out = rec.recommend([{"label": "TV", "confidence": 0.9}], room_type="living_room", top_k=10)
    scores = [i["final_score"] for i in out["recommendations"]]
    assert scores == sorted(scores, reverse=True)


def test_excludes_detected_groups() -> None:
    """Có TV trong ảnh thì không gợi ý thêm TV."""
    rec = _recommender()
    out = rec.recommend([{"label": "TV", "confidence": 0.9}], room_type="living_room", top_k=10)
    codes = {i["group_code"] for i in out["recommendations"]}
    assert "tv" not in codes
    assert "speaker" in codes


def test_keeps_detected_when_disabled() -> None:
    rec = _recommender()
    out = rec.recommend(
        [{"label": "TV", "confidence": 0.9}], room_type="living_room",
        top_k=10, exclude_detected=False,
    )
    assert "tv" in {i["group_code"] for i in out["recommendations"]}


def test_min_score_filters() -> None:
    rec = _recommender()
    out = rec.recommend([], room_type="bathroom", top_k=10, min_score=0.5)
    assert out["recommendations"] == []


def test_products_exclude_unavailable_and_random_sample() -> None:
    rec = _recommender()
    out = rec.recommend([{"label": "TV", "confidence": 0.9}], room_type="living_room",
                        top_k=10, exclude_detected=False, min_score=0)
    tv = next(i for i in out["recommendations"] if i["group_code"] == "tv")
    names = [p["product_name"] for p in tv["products"]]
    assert set(names) == {"TV rẻ", "TV đắt"}  # hết hàng bị loại, lấy ngẫu nhiên


def test_price_range_filter() -> None:
    rec = _recommender()
    out = rec.recommend([{"label": "TV", "confidence": 0.9}], room_type="living_room",
                        top_k=10, exclude_detected=False, min_score=0,
                        price_range=(5_000_000.0, float("inf")))
    tv = next(i for i in out["recommendations"] if i["group_code"] == "tv")
    assert [p["product_name"] for p in tv["products"]] == ["TV đắt"]


def test_groups_without_products_are_dropped() -> None:
    """Lọc giá không khớp gì thì nhóm phải biến mất, không trả `products: []`."""
    rec = _recommender()
    out = rec.recommend([], room_type="living_room", top_k=10, min_score=0,
                        price_range=(100_000_000.0, float("inf")))
    assert out["recommendations"] == []


def test_price_filter_keeps_filling_top_k() -> None:
    """Bỏ nhóm rỗng rồi lấy tiếp ứng viên sau, không trả thiếu so với `top_k`."""
    rec = _recommender()
    # TV bị loại khỏi ứng viên ở đây vì đã có trong ảnh; 2 nhóm còn lại đều có
    # sản phẩm >= 1,5tr nên vẫn đủ 2.
    out = rec.recommend(
        [{"label": "TV", "confidence": 0.9}], room_type="living_room",
        top_k=2, min_score=0, price_range=(1_500_000.0, float("inf")),
    )
    assert len(out["recommendations"]) == 2
    assert all(i["products"] for i in out["recommendations"])


def test_category_filter() -> None:
    rec = _recommender()
    out = rec.recommend([], room_type="living_room", top_k=20, min_score=0,
                        category_id="electronics")
    assert out["recommendations"]
    assert all(i["category_id"] == "electronics" for i in out["recommendations"])


def test_product_count_matches_catalog() -> None:
    rec = _recommender()
    out = rec.recommend([], room_type="living_room", top_k=20, min_score=0)
    counts = {i["group_code"]: i["product_count"] for i in out["recommendations"]}
    assert counts["tv"] == 3  # gồm cả sản phẩm hết hàng — đếm theo catalog


def test_reason_mentions_detected_object() -> None:
    rec = _recommender()
    out = rec.recommend([{"label": "TV", "confidence": 0.9}], room_type="living_room",
                        top_k=10, min_score=0)
    speaker = next(i for i in out["recommendations"] if i["group_code"] == "speaker")
    assert "TV" in speaker["reason"]


def test_reason_without_detections() -> None:
    rec = _recommender()
    out = rec.recommend([], room_type="living_room", top_k=1, min_score=0)
    assert out["recommendations"][0]["reason"]
    assert "phù hợp với phòng" in out["recommendations"][0]["reason"]


def test_empty_catalog_is_safe() -> None:
    """Nhóm có trong bảng điểm nhưng không có sản phẩm thì không xuất hiện."""
    rec = Recommender(_cooc(), Catalog({}, source="empty"), room_scores=_room_scores())
    out = rec.recommend([], room_type="living_room", top_k=10, min_score=0)
    assert out["recommendations"] == []


def test_response_shape() -> None:
    rec = _recommender()
    detected = [{"label": "Sofa", "confidence": 0.9}]
    out = rec.recommend(detected, room_type="bedroom", top_k=2, min_score=0)
    assert set(out) == {"room_type", "detected_objects", "recommendations"}
    assert out["detected_objects"] == detected
    item = out["recommendations"][0]
    assert set(item) == {
        "group_code", "group_name", "category_id", "cooccurrence_score",
        "room_score", "final_score", "product_count", "reason", "products",
    }
