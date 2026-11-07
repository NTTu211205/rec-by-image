"""Test nguồn dữ liệu: PostgreSQL (chỉ DB), file điểm phòng, ma trận, checkpoint.

Không còn nhánh fallback CSV — mọi trường hợp thiếu nguồn đều phải raise lỗi có
thông điệp chỉ đúng file/bảng. Test DB tự skip nếu PostgreSQL chưa lên.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.core.config import BASE_DIR, Settings
from app.core.exceptions import ArtifactError, DataSourceError
from app.services import group_scores
from app.services.cooccurrence import load_cooccurrence
from app.services.product_catalog import (
    Catalog,
    load_catalog_from_db,
    load_category_names,
)
from app.services.room_scores import ROOM_CODES, load_room_scores

ARTIFACTS = Path(BASE_DIR) / "artifacts" / "matrix"
SCORES_PATH = ARTIFACTS / "category_room_scores.json"
MATRIX_PATH = ARTIFACTS / "cooccurrence_matrix.json"
MODEL_PATH = Path(BASE_DIR) / "artifacts" / "rfdetr_medium" / "model.safetensors"
SAM2_PATH = Path(BASE_DIR) / "artifacts" / "sam2_small" / "model.safetensors"

# 1.413 sản phẩm / 62 nhóm / 157 brand — số liệu cố định của catalog hiện tại.
EXPECTED_PRODUCTS = 1413
EXPECTED_GROUPS = 62
EXPECTED_ROOMS = 7


def _db_config() -> dict:
    s = Settings()
    return {
        "host": s.db_host,
        "port": s.db_port,
        "dbname": s.db_name,
        "user": s.db_user,
        "password": s.db_password,
        "connect_timeout": 3,
    }


@pytest.fixture(scope="module")
def db_catalog() -> Catalog:
    if not Settings().db_host:
        pytest.skip("DB_HOST rỗng — không có PostgreSQL trong cấu hình")
    try:
        return load_catalog_from_db(_db_config(), Settings().db_schema)
    except DataSourceError as exc:
        pytest.skip(f"PostgreSQL chưa sẵn sàng: {exc}")


# --- Artifact có mặt --------------------------------------------------------

def test_artifacts_exist() -> None:
    """Các artifact bắt buộc: thiếu một cái là API không khởi động được."""
    for path in (SCORES_PATH, MATRIX_PATH, MODEL_PATH, SAM2_PATH):
        assert path.is_file(), f"thiếu artifact: {path}"


# --- W(c, r) từ file --------------------------------------------------------

def test_scores_load_from_json() -> None:
    table = load_room_scores(SCORES_PATH)
    assert len(table) == EXPECTED_GROUPS
    assert table.source.startswith("category_room_scores.json")
    for name, scores in table.by_name.items():
        assert set(scores) == set(ROOM_CODES), name
        assert all(0.0 <= v <= 1.0 for v in scores.values()), name


def test_scores_json_matches_known_values() -> None:
    """Giá trị cụ thể để phát hiện file bị ghi đè nhầm."""
    table = load_room_scores(SCORES_PATH)
    assert table.by_name["TV"]["living_room"] == 1.0
    assert table.by_name["TV"]["bathroom"] == 0.0
    assert table.by_name["Loa"]["living_room"] == 1.0
    assert table.by_name["Tủ lạnh"]["kitchen"] == 1.0


def test_room_codes_count() -> None:
    assert len(ROOM_CODES) == EXPECTED_ROOMS
    assert "living_room" in ROOM_CODES


def test_missing_scores_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ArtifactError, match="Không tìm thấy file điểm phòng"):
        load_room_scores(tmp_path / "khong-co.json")


def test_corrupt_scores_file_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(ArtifactError, match="Không đọc được file điểm phòng"):
        load_room_scores(bad)


def test_scores_missing_room_raises(tmp_path: Path) -> None:
    bad = tmp_path / "thieu_phong.json"
    bad.write_text(json.dumps({"TV": {"living_room": 1.0}}), encoding="utf-8")
    with pytest.raises(ArtifactError, match="không đúng 7 phòng"):
        load_room_scores(bad)


def test_scores_out_of_range_raises(tmp_path: Path) -> None:
    bad = tmp_path / "ngoai_mien.json"
    row = {room: 0.5 for room in ROOM_CODES} | {"living_room": 2.0}
    bad.write_text(json.dumps({"TV": row}), encoding="utf-8")
    with pytest.raises(ArtifactError, match="ngoài \\[0, 1\\]"):
        load_room_scores(bad)


def test_scores_not_object_raises(tmp_path: Path) -> None:
    bad = tmp_path / "list.json"
    bad.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    with pytest.raises(ArtifactError, match="phải là object"):
        load_room_scores(bad)


def test_bind_reports_missing_groups(tmp_path: Path) -> None:
    """File điểm thiếu nhóm so với DB -> lỗi rõ ràng, không bỏ qua im lặng."""
    table = load_room_scores(SCORES_PATH)
    with pytest.raises(ArtifactError, match="thiếu điểm phòng"):
        table.bind({"tv": "tv", "nhom_khong_co": "nhom_khong_co"})


def test_bind_maps_name_to_code() -> None:
    table = load_room_scores(SCORES_PATH)
    mapped = table.bind({"TV": "tv", "Loa": "speaker"})
    assert set(mapped) == {"tv", "speaker"}
    assert mapped["tv"]["living_room"] == 1.0


# --- Ma trận đồng xuất hiện -------------------------------------------------

def test_matrix_loads() -> None:
    cooc = load_cooccurrence(MATRIX_PATH)
    assert len(cooc["classes"]) == 45
    assert cooc["n_images"] > 0
    assert set(cooc["counts"]) == set(cooc["classes"])


def test_checkpoint_classes_match_matrix() -> None:
    """Kiểm tra artifact RF-DETR có config id2label hợp lệ."""
    import json

    config_path = MODEL_PATH.parent / "config.json"
    assert config_path.is_file(), f"thiếu config: {config_path}"
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)
    id2label = config.get("id2label", {})
    assert len(id2label) > 0


# --- Metadata nhóm ----------------------------------------------------------

def test_group_metadata_complete() -> None:
    codes = set(group_scores.GROUP_NAMES)
    assert codes == set(group_scores.GROUP_CATEGORY)
    assert set(group_scores.GROUP_CATEGORY.values()) <= set(group_scores.CATEGORY_NAMES)
    assert len(codes) == EXPECTED_GROUPS


def test_name_to_code_is_one_to_one() -> None:
    assert len(group_scores.NAME_TO_GROUP_CODE) == len(group_scores.GROUP_NAMES)
    assert group_scores.NAME_TO_GROUP_CODE["TV"] == "tv"


def test_scores_names_match_group_names() -> None:
    """File điểm khoá theo tên hiển thị, nên tên phải khớp `GROUP_NAMES`."""
    table = load_room_scores(SCORES_PATH)
    assert set(table.by_name) == set(group_scores.NAME_TO_GROUP_CODE)


# --- PostgreSQL (không fallback) -------------------------------------------

def test_db_catalog_counts(db_catalog: Catalog) -> None:
    assert db_catalog.total == EXPECTED_PRODUCTS
    assert len(db_catalog.groups) == EXPECTED_GROUPS
    assert db_catalog.source.startswith("PostgreSQL mwg.product")


def test_db_matches_scores_groups(db_catalog: Catalog) -> None:
    table = load_room_scores(SCORES_PATH)
    mapped = table.bind(load_category_names(_db_config(), Settings().db_schema))
    assert set(mapped) == set(db_catalog.groups)


def test_db_products_are_available(db_catalog: Catalog) -> None:
    """Nguồn hiện tại toàn bộ availability=TRUE; danh sách còn hàng phải đầy đủ."""
    assert sum(len(v) for v in db_catalog.available.values()) == EXPECTED_PRODUCTS


def test_db_product_shape(db_catalog: Catalog) -> None:
    product = db_catalog.groups["tv"][0]
    assert set(product) == {
        "product_id", "product_name", "brand", "price", "currency",
        "image_url", "product_url", "availability",
    }
    assert isinstance(product["price"], float)
    assert isinstance(product["availability"], bool)


def test_db_unreachable_raises() -> None:
    """Không còn fallback: DB chết thì phải raise, không trả catalog khác."""
    broken = {
        "host": "127.0.0.1", "port": 1, "dbname": "none", "user": "none",
        "password": "none", "connect_timeout": 1,
    }
    with pytest.raises(DataSourceError, match="Không kết nối được PostgreSQL"):
        load_catalog_from_db(broken, "mwg")


def test_db_missing_schema_raises(db_catalog: Catalog) -> None:
    """Schema sai tên -> báo thiếu bảng, không trả danh sách rỗng."""
    with pytest.raises(DataSourceError, match="thiếu bảng"):
        load_catalog_from_db(_db_config(), "schema_khong_ton_tai")


def test_category_names_are_unique(db_catalog: Catalog) -> None:
    mapping = load_category_names(_db_config(), Settings().db_schema)
    assert len(mapping) == EXPECTED_GROUPS
    assert mapping["TV"] == "tv"


def test_db_schema_defaults_to_mwg() -> None:
    assert Settings().db_schema == "mwg"
