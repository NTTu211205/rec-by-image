#!/usr/bin/env python
"""Sinh artifact cho API — tương đương Cell 10.3 + 10.5 của notebook.

  python scripts/build_artifacts.py

Ghi `artifacts/matrix/cooccurrence_matrix.json`. KHÔNG train model: checkpoint `.pt`
phải được copy/trỏ tới sẵn (xem README).
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings  # noqa: E402
from app.services.cooccurrence import (  # noqa: E402
    DROP_CLASSES,
    NAME_REMAP,
    build_cooccurrence_matrix,
    save_cooccurrence,
)


def main() -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--coco", default=None,
        help="đường dẫn _annotations.coco.json của train (mặc định: DATA_ROOT/train/...)",
    )
    parser.add_argument(
        "--out", default=None,
        help="file JSON đầu ra (mặc định: COOCCURRENCE_MATRIX_PATH)",
    )
    parser.add_argument(
        "--model", default=None,
        help="copy checkpoint .pt vào artifacts/ (mặc định: không copy)",
    )
    args = parser.parse_args()

    coco = Path(args.coco) if args.coco else (
        settings.resolve(settings.data_root) / "train" / "_annotations.coco.json"
    )
    out = Path(args.out) if args.out else settings.resolve(settings.cooccurrence_matrix_path)

    print(f"COCO train : {coco}")
    cooc = build_cooccurrence_matrix(coco, NAME_REMAP, DROP_CLASSES)
    print(f"Ảnh train  : {cooc['n_images']:,}")
    print(f"Class      : {len(cooc['classes'])}")

    support_sorted = sorted(cooc["support"].items(), key=lambda kv: -kv[1])
    print("\nSupport (12 class phổ biến nhất):")
    for cls, cnt in support_sorted[:12]:
        print(f"  {cls:<24} {cnt:>5} ảnh  ({cnt / cooc['n_images']:.1%})")

    saved = save_cooccurrence(cooc, out)
    print(f"\n💾 {saved}")

    if args.model:
        src = Path(args.model).expanduser()
        if not src.is_file():
            print(f"❌ không thấy checkpoint: {src}", file=sys.stderr)
            return 1
        # Copy vào đúng chỗ MODEL_PATH trỏ tới, không hard-code `artifacts/best.pt`.
        dst = settings.resolve(settings.model_path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"💾 {dst}  (từ {src})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
