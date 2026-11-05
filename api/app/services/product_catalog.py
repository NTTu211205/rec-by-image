"""Catalog sản phẩm: đọc PostgreSQL schema `mwg` — **chỉ** DB, không fallback CSV.

Hỗ trợ cả:
1. Schema mới (MWGShop 2 bảng `categories` và `products`):
    categories (id, name)
    products   (product_code, product_id, category_id, name, brand, sale_price, ...)
2. Schema cũ (3 bảng `categories`, `brand`, `product`):
    categories (category_id, category_tag, category_name, parent_category_code)
    brand      (brand_id, brand_name)
    product    (category_id -> categories, brand_id -> brand)
"""

from __future__ import annotations

import logging
from collections import defaultdict

from app.core.exceptions import DataSourceError

logger = logging.getLogger("app.catalog")

# Từ điển một sản phẩm.
Product = dict


class Catalog:
    """Gom sản phẩm theo `group_code` và giữ sẵn danh sách còn hàng."""

    def __init__(
        self,
        groups: dict[str, list[Product]],
        source: str,
        category_ids: dict[str, str] | None = None,
        category_names: dict[str, str] | None = None,
    ):
        self.groups: dict[str, list[Product]] = groups
        self.source = source
        self.counts: dict[str, int] = {g: len(v) for g, v in groups.items()}
        # Notebook lọc `availability` mỗi lần gợi ý (Cell 10.10); tính sẵn một lần.
        self.available: dict[str, list[Product]] = {
            g: [p for p in items if p.get("availability", True)] for g, items in groups.items()
        }
        self.category_ids: dict[str, str] = category_ids or {}
        self.category_names: dict[str, str] = category_names or {}

    @property
    def group_codes(self) -> list[str]:
        return sorted(self.groups)

    @property
    def total(self) -> int:
        return sum(self.counts.values())


def _row_to_product(row: dict) -> Product:
    """Chuẩn hoá một dòng DB về shape mà recommender dùng."""
    raw_price = row.get("price")
    if raw_price is None:
        raw_price = row.get("sale_price", 0.0)
    try:
        price_val = float(raw_price or 0.0)
    except (ValueError, TypeError):
        price_val = 0.0

    return {
        "product_id": str(row["product_id"]).strip(),
        "product_code": str(row["product_code"]).strip() if row.get("product_code") is not None else None,
        "product_name": str(row.get("product_name") or row.get("name") or "").strip(),
        "brand": str(row.get("brand_name") or row.get("brand") or "").strip(),
        "price": price_val,
        "currency": str(row.get("currency") or "VND").strip(),
        "image_url": str(row.get("image_url") or "").strip(),
        "product_url": str(row.get("product_url") or "").strip(),
        "availability": bool(row.get("availability", True)),
        "category_id": str(row["category_id"]).strip() if row.get("category_id") is not None else None,
    }


def load_catalog_from_db(cfg: dict, schema: str = "mwg") -> Catalog:
    """Đọc toàn bộ sản phẩm từ PostgreSQL. Lỗi kết nối -> raise.

    Tự động nhận diện cấu trúc bảng (products & categories hoặc product & categories & brand).
    """
    try:
        import psycopg2
        from psycopg2.extras import RealDictCursor
    except ImportError as exc:  # pragma: no cover
        raise DataSourceError(
            "Thiếu driver psycopg2 — không đọc được PostgreSQL."
        ) from exc

    if not schema.replace("_", "").isalnum():
        raise DataSourceError(f"DB_SCHEMA='{schema}' không hợp lệ.")

    try:
        conn = psycopg2.connect(**cfg)
    except Exception as exc:  # noqa: BLE001
        raise DataSourceError(
            f"Không kết nối được PostgreSQL ({cfg.get('host')}:{cfg.get('port')}/"
            f"{cfg.get('dbname')}): {type(exc).__name__}: {exc}"
        ) from exc

    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = %s
                """,
                (schema,),
            )
            existing = {row["table_name"] for row in cur.fetchall()}

            # 1. Schema mới: products + categories
            if "products" in existing and "categories" in existing:
                cur.execute(
                    f"""
                    SELECT p.product_id,
                           p.product_code,
                           p.name AS product_name,
                           p.brand AS brand_name,
                           COALESCE(p.sale_price, p.original_price, 0) AS price,
                           'VND' AS currency,
                           p.image_url,
                           p.product_url,
                           TRUE AS availability,
                           p.category_id,
                           c.name AS category_name,
                           p.category_id AS group_code
                    FROM {schema}.products AS p
                    JOIN {schema}.categories AS c ON c.id = p.category_id
                    ORDER BY p.category_id, p.sale_price
                    """
                )
                rows = cur.fetchall()

                cur.execute(f"SELECT id, name FROM {schema}.categories")
                cat_rows = cur.fetchall()
                category_ids = {str(r["id"]): str(r["id"]) for r in cat_rows}
                category_names = {str(r["id"]): str(r["name"]) for r in cat_rows}
                table_name = "products"

            # 2. Schema cũ: product + categories + brand
            elif "product" in existing and "categories" in existing and "brand" in existing:
                cur.execute(
                    f"""
                    SELECT p.product_id,
                           p.product_code,
                           p.product_name,
                           b.brand_name,
                           p.price,
                           p.currency,
                           p.image_url,
                           p.product_url,
                           p.availability,
                           p.category_id,
                           c.category_tag AS group_code,
                           c.category_name
                    FROM {schema}.product AS p
                    JOIN {schema}.categories AS c ON c.category_id = p.category_id
                    JOIN {schema}.brand AS b ON b.brand_id = p.brand_id
                    ORDER BY c.category_tag, p.price
                    """
                )
                rows = cur.fetchall()

                cur.execute(
                    f"SELECT category_tag, category_id, category_name FROM {schema}.categories"
                )
                cat_rows = cur.fetchall()
                category_ids = {
                    str(r["category_tag"]): str(r["category_id"]) for r in cat_rows
                }
                category_names = {
                    str(r["category_tag"]): str(r["category_name"]) for r in cat_rows
                }
                table_name = "product"
            else:
                raise DataSourceError(
                    f"Schema '{schema}' thiếu bảng catalog (đang có: {sorted(existing) or 'rỗng'}). "
                    "Cần có bảng 'products' & 'categories'. Chạy `make import`."
                )
    finally:
        conn.close()

    if not rows:
        raise DataSourceError(
            f"{schema}.{table_name} rỗng — chưa có sản phẩm nào. Chạy `make import`."
        )

    groups: dict[str, list[Product]] = defaultdict(list)
    for row in rows:
        groups[row["group_code"]].append(_row_to_product(row))

    logger.info("nạp catalog từ PostgreSQL: %d sản phẩm / %d nhóm",
                len(rows), len(groups), extra={"event": "catalog_db"})
    return Catalog(
        dict(groups),
        source=f"PostgreSQL {schema}.{table_name} ({len(rows)} sản phẩm)",
        category_ids=category_ids,
        category_names=category_names,
    )


def load_category_names(cfg: dict, schema: str = "mwg") -> dict[str, str]:
    """`category_name` -> `group_code` (category_id / category_tag) từ `{schema}.categories`.

    Cần vì `category_room_scores.json` khoá theo tên hiển thị (`"Bút cảm ứng"`, `"TV"`),
    còn catalog khoá theo mã id/tag (`1427`, `tv`).
    """
    try:
        import psycopg2
        from psycopg2.extras import RealDictCursor
    except ImportError as exc:  # pragma: no cover
        raise DataSourceError("Thiếu driver psycopg2.") from exc

    try:
        conn = psycopg2.connect(**cfg)
    except Exception as exc:  # noqa: BLE001
        raise DataSourceError(
            f"Không kết nối được PostgreSQL để đọc tên nhóm: {type(exc).__name__}: {exc}"
        ) from exc

    try:
        with conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = %s AND table_name = 'categories'
                """,
                (schema,),
            )
            cols = {r["column_name"] for r in cur.fetchall()}

            # 1. Schema mới: id, name
            if "name" in cols and "id" in cols:
                cur.execute(f"SELECT name AS category_name, id AS category_code FROM {schema}.categories")
            # 2. Schema cũ: category_name, category_tag
            elif "category_name" in cols and "category_tag" in cols:
                cur.execute(f"SELECT category_name, category_tag AS category_code FROM {schema}.categories")
            # 3. Fallback: category_name, category_id
            elif "category_name" in cols and "category_id" in cols:
                cur.execute(f"SELECT category_name, category_id AS category_code FROM {schema}.categories")
            else:
                cur.execute(f"SELECT name AS category_name, id AS category_code FROM {schema}.categories")

            rows = cur.fetchall()
    finally:
        conn.close()

    if not rows:
        raise DataSourceError(
            f"{schema}.categories rỗng — không ghép được điểm phòng theo tên."
        )

    mapping = {str(r["category_name"]): str(r["category_code"]) for r in rows}
    duplicates = len(rows) - len(mapping)
    if duplicates:
        raise DataSourceError(
            f"{schema}.categories có {duplicates} tên trùng — không ghép được "
            "điểm phòng theo tên."
        )
    return mapping
