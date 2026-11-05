"""Cell 6 → 10.5: gộp/lọc class COCO và học ma trận đồng xuất hiện.

Bản port trung thực của:

* Cell 6  — `CLASS_MERGE_MAP`, `DROP_CLASSES`, `NAME_REMAP`, `build_class_map`
* Cell 10.3 — `build_cooccurrence_matrix`
* Cell 10.4 — `BASE_RATE`, `P`, `P_smoothed`, `N`
* Cell 10.5 — lưu `cooccurrence_matrix.json`

Không train gì cả: đây chỉ là thống kê đếm trên annotation COCO của split train.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

from app.core.exceptions import DataSourceError

logger = logging.getLogger("app.cooccurrence")

# --- Cell 6: gộp category gốc trùng nghĩa về một class ---
CLASS_MERGE_MAP: dict[str, list[str]] = {
    "Pillow": ["Pillows", "Throw Pillow"],
    "Picture Frame": ["Picture Frame", "Photo Frame"],
    "Plant": ["Plants", "Plant", "Artificial Plants"],
    "Sofa": ["Sofa", "Sectional Sofa"],
    "Rug": ["Rug", "Area Rug", "Rug Runner"],
    "Curtain": ["Curtains", "Curtain Panel"],
    "Side Table": ["End Side Tables", "Snack Table"],
    "Bench/Ottoman": ["Ottomas and Benches", "Ottoman", "Bench", "Storage Bench"],
    "Chair": ["Chair", "Dining Chair"],
    "Table": ["Table", "Dining Table"],
    "Mirror": ["Wall Mirror", "Full Length Mirror"],
    "Clock": ["Wall Clock", "Chime Wall Clock", "Grandfather Clock", "Mantle Clock"],
    "Decor": ["Decor", "Statue", "Globe"],
    "Tableware": ["Bowl", "Serving Tray", "Cake Tray", "Coaster"],
    "Fireplace": ["Fireplace", "Fireplace Screen", "Fireplace Tool Set", "Fireplace Wood Holder"],
    "Candle": ["Candle", "Candle Holder", "Lantern"],
    "Planter": ["Planter", "Plant Stand"],
    "Basket/Bin": ["Basket", "Laundry Basket", "Storage Bin", "Plastic Storage Tote"],
    "Shelf": ["Book Shelf", "Shelf", "Corner Shelf", "Storage Shelf", "Cube Storage"],
    "Cabinet": ["Cabinet", "Dresser", "Hutch"],
    "Stool": ["Stool", "Bar Stool"],
    "TV": ["TV", "TV Stand"],
    "Speaker": ["Speaker", "Sound Bar", "Subwoofer", "Audio Receiver", "Radio", "DVD Player"],
    "Fan": ["Table Fan", "Ceiling Fan"],
    "Light": ["Chandeliers", "Light"],
}

DROP_CLASSES: frozenset[str] = frozenset({
    "object", "Backpack", "Umbrella", "Hairbrush", "Lysol", "Tape Measure",
    "Ping Pong Table", "Christmas Tree", "Dog Bed", "Coat Rack", "Toy",
    "Guitar", "Piano", "Bed", "Desk", "Bar Cart", "Wine Rack", "Table Runner",
    "Grill", "Refrigerator", "Microwave", "Dishwasher", "Wine Cooler",
})

NAME_REMAP: dict[str, str] = {
    original: merged for merged, originals in CLASS_MERGE_MAP.items() for original in originals
}


def build_class_map(
    split_dirs: list[Path],
    name_remap: dict[str, str] | None = None,
    drop_classes: set[str] | frozenset[str] | None = None,
) -> tuple[dict[int, int], dict[int, str]]:
    """Cell 6 `build_class_map`: COCO category id -> class index sau gộp/lọc."""
    name_remap = name_remap or {}
    drop_classes = drop_classes or frozenset()

    cat_id_to_orig_name: dict[int, str] = {}
    for d in split_dirs:
        p = Path(d) / "_annotations.coco.json"
        if p.exists():
            with open(p, encoding="utf-8") as f:
                for cat in json.load(f).get("categories", []):
                    cat_id_to_orig_name.setdefault(cat["id"], cat["name"])

    cat_id_to_final = {
        cid: name_remap.get(name, name)
        for cid, name in cat_id_to_orig_name.items()
        if name not in drop_classes
    }

    final_names_sorted = sorted(set(cat_id_to_final.values()))
    name_to_idx = {name: i for i, name in enumerate(final_names_sorted)}
    id_to_idx = {cid: name_to_idx[name] for cid, name in cat_id_to_final.items()}
    idx_to_name = {idx: name for name, idx in name_to_idx.items()}
    return id_to_idx, idx_to_name


def read_split_labels(
    coco_path: Path,
    name_remap: dict[str, str] | None = None,
    drop_classes: set[str] | frozenset[str] | None = None,
) -> tuple[list[int], dict[int, set[str]]]:
    """Đọc COCO -> (danh sách image id, nhãn đã gộp/lọc của từng ảnh).

    Dùng chung cho Cell 10.3 (học ma trận) và `scripts/build_artifacts.py`.
    """
    path = Path(coco_path)
    if not path.is_file():
        raise DataSourceError(f"Không tìm thấy annotation COCO: {path}")

    name_remap = name_remap or {}
    drop_classes = drop_classes or frozenset()

    try:
        with open(path, encoding="utf-8") as f:
            coco = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise DataSourceError(f"Không đọc được annotation COCO: {path}") from exc

    cid_to_name = {c["id"]: c["name"] for c in coco.get("categories", [])}
    cid_to_final = {
        cid: name_remap.get(name, name)
        for cid, name in cid_to_name.items()
        if name not in drop_classes
    }

    labels: dict[int, set[str]] = defaultdict(set)
    for ann in coco.get("annotations", []):
        final = cid_to_final.get(ann["category_id"])
        if final is not None:
            labels[ann["image_id"]].add(final)

    return [img["id"] for img in coco.get("images", [])], labels


def build_cooccurrence_matrix(
    coco_path: Path,
    name_remap: dict[str, str] | None = None,
    drop_classes: set[str] | frozenset[str] | None = None,
) -> dict:
    """Cell 10.3: P(d | c) = N(c ∩ d) / N(c), chuẩn hoá theo CỘT."""
    image_ids, labels = read_split_labels(coco_path, name_remap, drop_classes)

    classes = sorted({obj for objs in labels.values() for obj in objs})
    image_sets: dict[str, set[int]] = {c: set() for c in classes}
    for image_id, objs in labels.items():
        for obj in objs:
            image_sets[obj].add(image_id)

    index = {c: i for i, c in enumerate(classes)}
    matrix = [[0.0] * len(classes) for _ in classes]
    support: dict[str, int] = {}
    counts: dict[str, dict[str, int]] = {}

    for source in classes:
        src_images = image_sets[source]
        n_source = len(src_images)
        support[source] = n_source
        row_counts: dict[str, int] = {}
        for target in classes:
            n_both = len(src_images & image_sets[target])
            row_counts[target] = n_both
            matrix[index[source]][index[target]] = (n_both / n_source) if n_source else 0.0
        counts[source] = row_counts

    return {
        "classes": classes,
        "matrix": matrix,
        "support": support,
        "counts": counts,
        "n_images": len(image_ids),
    }


def base_rate_table(cooc: dict) -> dict[str, float]:
    """Cell 10.4: P(e) = N(e) / N(train)."""
    n_images = cooc["n_images"]
    return {
        cls: (support / n_images) if n_images else 0.0
        for cls, support in cooc["support"].items()
    }


def save_cooccurrence(cooc: dict, out_path: Path) -> Path:
    """Cell 10.5: lưu ma trận + base_rate để API đọc lại."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "classes": cooc["classes"],
        "matrix": cooc["matrix"],
        "support": cooc["support"],
        "base_rate": base_rate_table(cooc),
        "n_images": cooc["n_images"],
        "note": (
            f"matrix[c][d] = P(d|c) = N(c ∩ d) / N(c), tính trên {cooc['n_images']} ảnh train. "
            "base_rate[c] = P(c) = N(c) / n_images, dùng làm prior cho smoothing tầng 3."
        ),
    }
    # `counts` cần cho smoothing nên lưu kèm (khác notebook, không đổi công thức).
    payload["counts"] = cooc["counts"]
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def _derive_counts(matrix: list[list[float]], support: dict[str, int], classes: list[str]) -> dict[str, dict[str, int]]:
    """Tái tạo N(c ∩ d) = P(d | c) · N(c) từ ma trận đã chuẩn hoá theo hàng.

    File ma trận do notebook sinh ra có thể chỉ lưu `matrix` + `support` (Cell 10.5);
    `counts` suy ra được chính xác vì `P(d|c)` chính là `N(c ∩ d) / N(c)`.
    """
    counts: dict[str, dict[str, int]] = {}
    for i, c in enumerate(classes):
        n_c = int(support.get(c, 0))
        counts[c] = {
            d: int(round(matrix[i][j] * n_c)) for j, d in enumerate(classes)
        }
    return counts


def load_cooccurrence(path: Path) -> dict:
    """Nạp ma trận JSON đã lưu, kiểm tra tính nhất quán trước khi dùng."""
    path = Path(path)
    if not path.is_file():
        raise DataSourceError(f"Không tìm thấy ma trận đồng xuất hiện: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataSourceError(f"Không đọc được ma trận đồng xuất hiện: {path}") from exc

    # `counts` là tuỳ chọn: suy ra từ matrix × support nếu file không lưu sẵn.
    required = ("classes", "matrix", "support", "base_rate", "n_images")
    missing = [k for k in required if k not in raw]
    if missing:
        raise DataSourceError(f"Ma trận đồng xuất hiện thiếu trường: {', '.join(missing)}")

    classes = list(raw["classes"])
    n = len(classes)
    matrix = raw["matrix"]
    if n == 0 or len(matrix) != n or any(len(row) != n for row in matrix):
        raise DataSourceError("Ma trận đồng xuất hiện không vuông khớp danh sách class.")

    matrix_f = [[float(v) for v in row] for row in matrix]
    support = {k: int(v) for k, v in raw["support"].items()}
    if raw.get("counts"):
        counts = {k: {t: int(v) for t, v in row.items()} for k, row in raw["counts"].items()}
    else:
        logger.info(
            "Ma trận không có `counts` — suy ra từ matrix × support",
            extra={"event": "counts_derived", "classes": n},
        )
        counts = _derive_counts(matrix_f, support, classes)

    return {
        "classes": classes,
        "matrix": matrix_f,
        "support": support,
        "counts": counts,
        "base_rate": {k: float(v) for k, v in raw["base_rate"].items()},
        "n_images": int(raw["n_images"]),
    }
