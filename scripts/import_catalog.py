#!/usr/bin/env python3
"""Nạp dữ liệu từ api_crawl_infor_mwgshop.json vào PostgreSQL (schema categories & products).

    python3 scripts/import_catalog.py
    python3 scripts/import_catalog.py --dry-run
    python3 scripts/import_catalog.py --dsn postgresql://mwg:mwg_secret@localhost:55432/mwgshop
    python3 scripts/import_catalog.py --dsn postgresql://thanhtu@localhost:5432/mwgshop
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
import psycopg2
from psycopg2.extras import execute_values

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JSON_CANDIDATES = [
    ROOT / "data" / "api_crawl_infor_new.json",
    Path("/Users/thanhtu/Downloads/Data-superapp-dmx/api_crawl_infor_new.json"),
    ROOT / "data" / "api_crawl_infor_mwgshop.json",
    Path("/Users/thanhtu/Downloads/Data-superapp-dmx/api_crawl_infor_mwgshop.json"),
]
SCHEMA_PATH = ROOT / "db" / "init" / "01_schema.sql"

DEFAULT_DSN = os.getenv(
    "DATABASE_URL",
    "postgresql://mwg:mwg_secret@localhost:55432/mwgshop"
)


def find_json_path(specified: str | None) -> Path:
    if specified:
        p = Path(specified).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Không tìm thấy file JSON: {p}")
        return p
    for c in DEFAULT_JSON_CANDIDATES:
        if c.is_file():
            return c
    raise FileNotFoundError(
        "Không tìm thấy file JSON catalog tại data/ hoặc ~/Downloads/Data-superapp-dmx/"
    )


def clean_str(s: object) -> str | None:
    if s is None:
        return None
    s = str(s).strip()
    return s if s else None


def clean_num(n: object) -> float | None:
    if n is None or n == "":
        return None
    try:
        return float(n)
    except (ValueError, TypeError):
        return None


def clean_int(n: object) -> int | None:
    if n is None or n == "":
        return None
    try:
        return int(n)
    except (ValueError, TypeError):
        return None


def load_and_parse(json_path: Path):
    print(f"📖 Đang đọc JSON từ {json_path} ...")
    t0 = time.time()
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"   Đã nạp {len(data):,} bản ghi trong {time.time() - t0:.2f}s")

    categories: dict[str, str] = {}
    products: list[tuple] = []

    for item in data:
        cid = clean_str(item.get("category_id") or item.get("categoryId"))
        cname = clean_str(item.get("category_name") or item.get("categoryName"))
        if cid and cid not in categories:
            categories[cid] = cname or f"Category {cid}"

        spec = item.get("spec_product") if "spec_product" in item else item.get("specProduct")
        spec_json = json.dumps(spec) if isinstance(spec, (dict, list)) else "{}"

        prog = item.get("gia_theo_chuong_trinh")
        prog_json = json.dumps(prog) if isinstance(prog, (dict, list)) else "[]"

        p = (
            clean_str(item.get("productcode") or item.get("productCode")),
            clean_str(item.get("product_id") or item.get("productId")),
            cid,
            clean_str(item.get("tên sản phẩm")),
            clean_str(item.get("producttype") or item.get("productType")),
            clean_str(item.get("brand")),
            clean_str(item.get("brand_code") or item.get("brandCode")),
            clean_str(item.get("url_image") or item.get("urlImage")),
            clean_num(item.get("Giá gốc")),
            clean_num(item.get("Giá khuyến mãi")),
            clean_num(item.get("rating_vote") or item.get("ratingVote")),
            clean_int(item.get("quantity_sold") or item.get("quantitySold")),
            clean_str(item.get("màu sắc")),
            clean_str(item.get("Phụ kiện đi kèm")),
            clean_str(item.get("chính sách bảo hành")),
            clean_str(item.get("promotion")),
            clean_str(item.get("outstanding")),
            spec_json,
            item.get("onlineSaleOnly") if isinstance(item.get("onlineSaleOnly"), bool) else None,
            clean_str(item.get("url")),
            clean_str(item.get("model_code") or item.get("modelCode")),
            clean_int(item.get("sale_program_id") or item.get("saleProgramId")),
            prog_json,
            clean_num(item.get("tong_gia_tri_qua")),
            clean_int(item.get("seller_id") or item.get("sellerId")),
            clean_str(item.get("seller_name") or item.get("sellerName")),
            clean_int(item.get("total_rating") or item.get("totalRating")),
            clean_int(item.get("online_status_id") or item.get("onlineStatusId")),
            clean_str(item.get("time_crawler") or item.get("timeCrawler")),
        )
        products.append(p)

    return categories, products


def run_import(
    dsn: str,
    categories: dict[str, str],
    products: list[tuple],
    recreate_schema: bool = True,
    batch_size: int = 2000,
):
    print(f"🔌 Kết nối PostgreSQL ({dsn}) ...")
    conn = psycopg2.connect(dsn)
    conn.autocommit = False

    try:
        with conn.cursor() as cur:
            if recreate_schema:
                print(f"⚙️ Khởi tạo schema từ {SCHEMA_PATH} ...")
                schema_sql = SCHEMA_PATH.read_text(encoding="utf-8")
                cur.execute(schema_sql)
            else:
                print("🧹 Dọn dẹp dữ liệu cũ (TRUNCATE) ...")
                cur.execute("TRUNCATE TABLE products, categories CASCADE;")

            print(f"📥 Đang nạp {len(categories)} categories ...")
            cat_tuples = [(cid, name) for cid, name in categories.items()]
            insert_cats_sql = """
                INSERT INTO categories (id, name)
                VALUES %s
                ON CONFLICT (id) DO UPDATE SET name = EXCLUDED.name;
            """
            execute_values(cur, insert_cats_sql, cat_tuples)

            print(f"📥 Đang nạp {len(products):,} products (batch size {batch_size}) ...")
            insert_products_sql = """
                INSERT INTO products (
                    product_code,
                    product_id,
                    category_id,
                    name,
                    product_type,
                    brand,
                    brand_code,
                    image_url,
                    original_price,
                    sale_price,
                    rating_vote,
                    quantity_sold,
                    color,
                    accessories,
                    warranty_policy,
                    promotion,
                    outstanding,
                    spec_product,
                    online_sale_only,
                    product_url,
                    model_code,
                    sale_program_id,
                    program_prices,
                    total_gift_value,
                    seller_id,
                    seller_name,
                    total_rating,
                    online_status_id,
                    time_crawler
                ) VALUES %s
                ON CONFLICT (product_code) DO UPDATE SET
                    product_id = EXCLUDED.product_id,
                    category_id = EXCLUDED.category_id,
                    name = EXCLUDED.name,
                    product_type = EXCLUDED.product_type,
                    brand = EXCLUDED.brand,
                    brand_code = EXCLUDED.brand_code,
                    image_url = EXCLUDED.image_url,
                    original_price = EXCLUDED.original_price,
                    sale_price = EXCLUDED.sale_price,
                    rating_vote = EXCLUDED.rating_vote,
                    quantity_sold = EXCLUDED.quantity_sold,
                    color = EXCLUDED.color,
                    accessories = EXCLUDED.accessories,
                    warranty_policy = EXCLUDED.warranty_policy,
                    promotion = EXCLUDED.promotion,
                    outstanding = EXCLUDED.outstanding,
                    spec_product = EXCLUDED.spec_product,
                    online_sale_only = EXCLUDED.online_sale_only,
                    product_url = EXCLUDED.product_url,
                    model_code = EXCLUDED.model_code,
                    sale_program_id = EXCLUDED.sale_program_id,
                    program_prices = EXCLUDED.program_prices,
                    total_gift_value = EXCLUDED.total_gift_value,
                    seller_id = EXCLUDED.seller_id,
                    seller_name = EXCLUDED.seller_name,
                    total_rating = EXCLUDED.total_rating,
                    online_status_id = EXCLUDED.online_status_id,
                    time_crawler = EXCLUDED.time_crawler;
            """
            for i in range(0, len(products), batch_size):
                batch = products[i : i + batch_size]
                execute_values(cur, insert_products_sql, batch)
                print(f"   -> Đã ghi {min(i + batch_size, len(products)):,}/{len(products):,} sản phẩm")

        conn.commit()
        print("💾 Transaction committed thành công!")

        # Verify
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM categories;")
            cat_count = cur.fetchone()[0]
            cur.execute("SELECT count(*) FROM products;")
            prod_count = cur.fetchone()[0]
            cur.execute("""
                SELECT count(*) FROM products p
                LEFT JOIN categories c ON c.id = p.category_id
                WHERE c.id IS NULL;
            """)
            orphan_count = cur.fetchone()[0]
            cur.execute("SELECT min(sale_price), max(sale_price), count(DISTINCT brand) FROM products;")
            min_p, max_p, brand_cnt = cur.fetchone()

        print("\n" + "=" * 50)
        print("📊 KẾT QUẢ KIỂM TRA TOÀN VẸN:")
        print(f"  ✅ Categories: {cat_count} (mong đợi {len(categories)})")
        print(f"  ✅ Products:   {prod_count:,} (mong đợi {len(products):,})")
        print(f"  ✅ Khóa ngoại mồ côi: {orphan_count}")
        print(f"  ✅ Khoảng giá: {min_p:,.0f}đ - {max_p:,.0f}đ")
        print(f"  ✅ Số brand:   {brand_cnt}")
        print("=" * 50 + "\n")

    except Exception as exc:
        conn.rollback()
        print(f"❌ Lỗi: {exc}")
        raise
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description="Nạp mwgshop JSON vào PostgreSQL")
    parser.add_argument("--json", dest="json_path", help="Đường dẫn file JSON")
    parser.add_argument("--dsn", default=DEFAULT_DSN, help=f"PostgreSQL DSN (mặc định: {DEFAULT_DSN})")
    parser.add_argument("--dry-run", action="store_true", help="Chỉ đọc JSON và kiểm tra dữ liệu, không ghi DB")
    parser.add_argument("--no-recreate", action="store_true", help="Không recreate schema, chỉ TRUNCATE")
    parser.add_argument("--batch-size", type=int, default=2000, help="Batch size khi INSERT")
    args = parser.parse_args()

    json_file = find_json_path(args.json_path)
    categories, products = load_and_parse(json_file)

    if args.dry_run:
        print(f"🔍 [DRY-RUN] Hợp lệ! Đã chuẩn bị {len(categories)} categories và {len(products)} products.")
        return

    run_import(
        dsn=args.dsn,
        categories=categories,
        products=products,
        recreate_schema=not args.no_recreate,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
