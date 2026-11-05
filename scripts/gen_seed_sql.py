#!/usr/bin/env python3
"""Sinh `db/init/02_seed.sql` từ `api_crawl_infor_mwgshop.json`.

Dùng cho Docker: container `postgres` chạy các file trong `db/init` khi volume
còn trống. File seed này đảm bảo container tự khởi tạo đầy đủ 243 categories
và 25,034 sản phẩm mà không cần chạy lệnh nạp thủ công.

Chạy:
    python3 scripts/gen_seed_sql.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JSON = ROOT / "data" / "api_crawl_infor_new.json"
DEFAULT_OUT = ROOT / "db" / "init" / "02_seed.sql"
BATCH_SIZE = 500


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


def lit_str(s: str | None) -> str:
    if s is None:
        return "NULL"
    escaped = s.replace("'", "''")
    return f"'{escaped}'"


def lit_num(n: float | None) -> str:
    if n is None:
        return "NULL"
    return str(n)


def lit_int(n: int | None) -> str:
    if n is None:
        return "NULL"
    return str(n)


def lit_bool(b: bool | None) -> str:
    if b is None:
        return "NULL"
    return "TRUE" if b else "FALSE"


def lit_json(obj: object, default: str) -> str:
    if not isinstance(obj, (dict, list)):
        obj_str = default
    else:
        obj_str = json.dumps(obj, ensure_ascii=False)
    escaped = obj_str.replace("'", "''")
    return f"'{escaped}'::jsonb"


def main():
    parser = argparse.ArgumentParser(description="Sinh seed SQL từ JSON")
    parser.add_argument("--json", dest="json_path", default=str(DEFAULT_JSON), help="Đường dẫn file JSON")
    parser.add_argument("--out", dest="out_path", default=str(DEFAULT_OUT), help="Đường dẫn file SQL đầu ra")
    args = parser.parse_args()

    json_path = Path(args.json_path).resolve()
    out_path = Path(args.out_path).resolve()

    print(f"📖 Đang đọc JSON từ {json_path} ...")
    t0 = time.time()
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    categories: dict[str, str] = {}
    products_rows: list[str] = []

    for item in data:
        cid = clean_str(item.get("category_id") or item.get("categoryId"))
        cname = clean_str(item.get("category_name") or item.get("categoryName"))
        if cid and cid not in categories:
            categories[cid] = cname or f"Category {cid}"

        spec = item.get("spec_product") if "spec_product" in item else item.get("specProduct")
        prog = item.get("gia_theo_chuong_trinh")

        p_vals = [
            lit_str(clean_str(item.get("productcode") or item.get("productCode"))),
            lit_str(clean_str(item.get("product_id") or item.get("productId"))),
            lit_str(cid),
            lit_str(clean_str(item.get("tên sản phẩm"))),
            lit_str(clean_str(item.get("producttype") or item.get("productType"))),
            lit_str(clean_str(item.get("brand"))),
            lit_str(clean_str(item.get("brand_code") or item.get("brandCode"))),
            lit_str(clean_str(item.get("url_image") or item.get("urlImage"))),
            lit_num(clean_num(item.get("Giá gốc"))),
            lit_num(clean_num(item.get("Giá khuyến mãi"))),
            lit_num(clean_num(item.get("rating_vote") or item.get("ratingVote"))),
            lit_int(clean_int(item.get("quantity_sold") or item.get("quantitySold"))),
            lit_str(clean_str(item.get("màu sắc"))),
            lit_str(clean_str(item.get("Phụ kiện đi kèm"))),
            lit_str(clean_str(item.get("chính sách bảo hành"))),
            lit_str(clean_str(item.get("promotion"))),
            lit_str(clean_str(item.get("outstanding"))),
            lit_json(spec, "{}"),
            lit_bool(item.get("onlineSaleOnly") if isinstance(item.get("onlineSaleOnly"), bool) else None),
            lit_str(clean_str(item.get("url"))),
            lit_str(clean_str(item.get("model_code") or item.get("modelCode"))),
            lit_int(clean_int(item.get("sale_program_id") or item.get("saleProgramId"))),
            lit_json(prog, "[]"),
            lit_num(clean_num(item.get("tong_gia_tri_qua"))),
            lit_int(clean_int(item.get("seller_id") or item.get("sellerId"))),
            lit_str(clean_str(item.get("seller_name") or item.get("sellerName"))),
            lit_int(clean_int(item.get("total_rating") or item.get("totalRating"))),
            lit_int(clean_int(item.get("online_status_id") or item.get("onlineStatusId"))),
            lit_str(clean_str(item.get("time_crawler") or item.get("timeCrawler"))),
        ]
        products_rows.append("(" + ", ".join(p_vals) + ")")

    print(f"✍️ Đang ghi file {out_path} ...")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("-- Seed data generated from api_crawl_infor_mwgshop.json\n")
        f.write("-- 243 categories, 25,034 products\n\n")
        f.write("BEGIN;\n\n")

        # Categories
        f.write("-- 1. Categories\n")
        cat_lines = []
        for cid, cname in categories.items():
            cat_lines.append(f"({lit_str(cid)}, {lit_str(cname)})")
        f.write("INSERT INTO public.categories (id, name) VALUES\n")
        f.write(",\n".join(f"  {line}" for line in cat_lines))
        f.write("\nON CONFLICT (id) DO UPDATE SET name = EXCLUDED.name;\n\n")

        # Products
        f.write("-- 2. Products (in batches)\n")
        cols = (
            "product_code, product_id, category_id, name, product_type, brand, "
            "brand_code, image_url, original_price, sale_price, rating_vote, "
            "quantity_sold, color, accessories, warranty_policy, promotion, "
            "outstanding, spec_product, online_sale_only, product_url, "
            "model_code, sale_program_id, program_prices, total_gift_value, "
            "seller_id, seller_name, total_rating, online_status_id, time_crawler"
        )
        for i in range(0, len(products_rows), BATCH_SIZE):
            chunk = products_rows[i : i + BATCH_SIZE]
            f.write(f"INSERT INTO public.products ({cols}) VALUES\n")
            f.write(",\n".join(f"  {row}" for row in chunk))
            f.write("\nON CONFLICT (product_code) DO NOTHING;\n\n")

        f.write("COMMIT;\n")

    print(f"✅ Hoàn tất trong {time.time() - t0:.2f}s. File size: {out_path.stat().st_size / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()
