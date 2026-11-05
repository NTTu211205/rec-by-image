#!/usr/bin/env python3
"""Sinh `api/artifacts/matrix/category_room_scores.json` từ `data/dienmayxanh_products.csv`.

Đây là artifact Cell 10.7 của notebook — bảng W(c, r) mà API đọc lúc chạy. Khoá
ngoài là **tên nhóm hiển thị** (`"TV"`, `"Loa"`, `"Máy điều hòa"`…), khoá trong là
7 mã phòng, vì notebook gốc ghi bảng này theo tên (`category_name`) chứ không
theo mã nhóm.

Chạy lại khi CSV đổi (điểm phòng thay đổi):

    python3 scripts/gen_group_scores.py
    python3 scripts/gen_group_scores.py --csv data/khac.csv --out /tmp/scores.json

Script kiểm tra trước khi ghi: đủ 7 phòng mỗi nhóm, giá trị trong [0, 1], và một
nhóm chỉ có **một** bộ điểm (nếu CSV cho hai bộ khác nhau thì dừng, không tự chọn).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT / "data" / "dienmayxanh_products.csv"
DEFAULT_OUT = ROOT / "api" / "artifacts" / "matrix" / "category_room_scores.json"

ROOM_CODES: tuple[str, ...] = (
    "living_room",
    "bedroom",
    "kitchen",
    "bathroom",
    "dining_room",
    "home_office",
    "balcony",
)


def build(csv_path: Path) -> dict[str, dict[str, float]]:
    with open(csv_path, encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"CSV rỗng: {csv_path}")

    scores: dict[str, dict[str, float]] = {}

    for row in rows:
        code = row["subcategory"].strip()
        display = row["group_name"].strip()
        if not code or not display:
            raise SystemExit(f"Dòng thiếu subcategory/group_name: {row.get('product_id')}")

        try:
            weights = json.loads(row["room_weights"])
        except json.JSONDecodeError as exc:
            raise SystemExit(
                f"room_weights không phải JSON ({row.get('product_id')}): {exc}"
            ) from exc

        if set(weights) != set(ROOM_CODES):
            raise SystemExit(
                f"Nhóm '{code}' không đủ 7 phòng: {sorted(weights)}"
            )

        normalised = {room: float(weights[room]) for room in ROOM_CODES}
        for room, value in normalised.items():
            if not 0.0 <= value <= 1.0:
                raise SystemExit(f"Nhóm '{code}' phòng '{room}' = {value} ngoài [0, 1]")

        # Cùng một nhóm phải cùng một bộ điểm; khác nhau là CSV không nhất quán.
        if code in scores and scores[code] != normalised:
            raise SystemExit(
                f"Nhóm '{code}' có hai bộ room_weights khác nhau — sửa CSV trước."
            )
        scores[code] = normalised

    # Đổi khoá sang tên hiển thị, đúng định dạng artifact notebook sinh ra.
    by_name: dict[str, dict[str, float]] = {}
    for code, weights in scores.items():
        row = next(r for r in rows if r["subcategory"].strip() == code)
        name = row["group_name"].strip()
        if name in by_name:
            raise SystemExit(f"Hai nhóm trùng tên hiển thị '{name}' — không ghép được.")
        by_name[name] = weights

    return dict(sorted(by_name.items()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=str(DEFAULT_CSV))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    csv_path = Path(args.csv).expanduser()
    if not csv_path.is_file():
        raise SystemExit(f"❌ không thấy CSV: {csv_path}")

    by_name = build(csv_path)
    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(by_name, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    values = [v for row in by_name.values() for v in row.values()]
    print(f"✅ {out_path}")
    print(
        f"   {len(by_name)} nhóm × {len(ROOM_CODES)} phòng "
        f"({min(values):.2f}–{max(values):.2f})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
