"""W(c, r) — điểm phù hợp phòng, đọc từ `artifacts/matrix/category_room_scores.json`.

Đây là bản port Cell 10.7 của notebook: file JSON do bước huấn luyện/chuẩn bị
artifact sinh ra, khoá ngoài là **tên hiển thị nhóm** (`"TV"`, `"Loa"`,
`"Máy chiếu"`…), khoá trong là 7 mã phòng.

    {
      "TV": {"living_room": 1.0, "bedroom": 0.8, "kitchen": 0.1, ...},
      "Loa": {...}
    }

Vì khoá là tên chứ không phải `group_code`, module này KHÔNG tự ánh xạ: nó đọc
bảng theo tên rồi để `product_catalog` ghép với `mwg.categories`
(`category_name` -> `category_id`) — DB là nguồn duy nhất biết mã nhóm.

Không có bảng điểm dự phòng trong code: thiếu/sai file là lỗi khởi động.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Mapping

from app.core.exceptions import ArtifactError

logger = logging.getLogger("app.room_scores")

# 7 phòng API chấp nhận. Đây là hợp đồng của endpoint (`?room_type=`), không phải
# dữ liệu huấn luyện, nên giữ trong code; file JSON phải có đủ cả 7 cho mọi nhóm.
ROOM_CODES: tuple[str, ...] = (
    "living_room",
    "bedroom",
    "kitchen",
    "bathroom",
    "dining_room",
    "home_office",
    "balcony",
)


class RoomScoreTable:
    """Bảng điểm theo tên nhóm, kèm ánh xạ sang `group_code` sau khi ghép DB."""

    def __init__(self, by_name: dict[str, dict[str, float]], source: str):
        self.by_name = by_name
        self.source = source

    def __len__(self) -> int:
        return len(self.by_name)

    def bind(self, name_to_code: Mapping[str, str]) -> dict[str, dict[str, float]]:
        """Đổi khoá tên -> `group_code`.

        Lệch giữa file điểm và catalog là lỗi cấu hình, không phải chuyện bỏ qua
        im lặng: nhóm không có điểm sẽ vĩnh viễn không được gợi ý.
        """
        missing = sorted(set(name_to_code) - set(self.by_name))
        if missing:
            raise ArtifactError(
                f"{self.source} thiếu điểm phòng cho {len(missing)} nhóm: "
                f"{', '.join(missing[:8])}{'…' if len(missing) > 8 else ''}. "
                "Sinh lại artifact hoặc sửa tên nhóm cho khớp `mwg.categories`."
            )

        unknown = sorted(set(self.by_name) - set(name_to_code))
        if unknown:
            logger.warning(
                "File điểm có nhóm không tồn tại trong catalog (bỏ qua): %s",
                ", ".join(unknown[:8]),
                extra={"event": "room_scores_extra", "count": len(unknown)},
            )

        return {
            name_to_code[name]: scores
            for name, scores in self.by_name.items()
            if name in name_to_code
        }


def _validate(by_name: dict, path: Path) -> dict[str, dict[str, float]]:
    """Mọi nhóm phải có đủ 7 phòng, giá trị số trong [0, 1]."""
    expected = set(ROOM_CODES)
    checked: dict[str, dict[str, float]] = {}

    for name, scores in by_name.items():
        if not isinstance(scores, dict):
            raise ArtifactError(
                f"{path.name}: nhóm '{name}' phải là object {{room: score}}."
            )
        if set(scores) != expected:
            missing = sorted(expected - set(scores))
            extra = sorted(set(scores) - expected)
            raise ArtifactError(
                f"{path.name}: nhóm '{name}' không đúng 7 phòng "
                f"(thiếu {missing}, thừa {extra})."
            )
        row: dict[str, float] = {}
        for room, value in scores.items():
            try:
                score = float(value)
            except (TypeError, ValueError) as exc:
                raise ArtifactError(
                    f"{path.name}: điểm '{name}/{room}' không phải số ({value!r})."
                ) from exc
            if not 0.0 <= score <= 1.0:
                raise ArtifactError(
                    f"{path.name}: điểm '{name}/{room}' = {score} ngoài [0, 1]."
                )
            row[room] = score
        checked[str(name)] = row

    if not checked:
        raise ArtifactError(f"{path.name} không có nhóm nào.")
    return checked


def load_room_scores(path: Path) -> RoomScoreTable:
    """Đọc `category_room_scores.json`; thiếu file hoặc sai cấu trúc đều raise."""
    path = Path(path)
    if not path.is_file():
        raise ArtifactError(f"Không tìm thấy file điểm phòng: {path}")

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"Không đọc được file điểm phòng: {path}") from exc

    if not isinstance(raw, dict):
        raise ArtifactError(
            f"{path.name} phải là object {{tên nhóm: {{room: score}}}}, "
            f"nhận được {type(raw).__name__}."
        )

    by_name = _validate(raw, path)
    logger.info(
        "nạp W(c, r) từ %s: %d nhóm × %d phòng",
        path.name, len(by_name), len(ROOM_CODES),
        extra={"event": "room_scores_loaded", "groups": len(by_name)},
    )
    return RoomScoreTable(by_name, source=f"{path.name} ({len(by_name)} nhóm × {len(ROOM_CODES)} phòng)")
