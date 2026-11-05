# Gợi ý đồ điện máy từ ảnh phòng

PostgreSQL 16 (catalog 1.413 sản phẩm) + FastAPI (RT-DETR + ma trận đồng xuất hiện).

Ảnh phòng → detector RT-DETR → xếp hạng 62 nhóm sản phẩm → trả về sản phẩm kèm giá.

| Service  | URL                    | Credentials (từ `.env`)          |
| -------- | ---------------------- | -------------------------------- |
| API      | http://localhost:8000  | Swagger: `/docs`                 |
| Postgres | `localhost:55432`      | `mwg` / `mwg_secret` / `mwgshop` |
| Adminer  | http://localhost:58080 | server `postgres`, cùng creds    |

## Quick start

```bash
docker compose up -d --build     # postgres (tự nạp 1.413 sản phẩm) + api + adminer
curl -s localhost:8000/health/ready | python3 -m json.tool
```

`/health/ready` báo rõ nguồn của cả bốn thành phần:

```json
{
  "model_source": "best.pt (rtdetr)",
  "cooccurrence_source": "cooccurrence_matrix.json (45 class, 1717 ảnh train)",
  "room_scores_source": "artifact category_room_scores.json (62 nhóm × 7 phòng) + mwg.categories",
  "catalog_source": "PostgreSQL mwg.product (1413 sản phẩm)"
}
```

Không có nhánh dự phòng nào: nguồn nào thiếu thì API dừng khởi động và log ghi
đúng đường dẫn đã thử.

```bash
curl -X POST "localhost:8000/api/v1/recommendations/image?room_type=living_room&top_k=5" \
     -F "image=@phong-khach.jpg"
```

## Layout

```
docker-compose.yml            postgres + api + adminer, volume pgdata
db/init/01_schema.sql         schema `mwg`: categories, brand, product (đúng 3 bảng)
db/init/02_seed.sql           1.413 sản phẩm (SINH từ CSV — đừng sửa tay)
data/dienmayxanh_products.csv nguồn: 1.413 sản phẩm, 62 nhóm, 157 brand, 5 category
scripts/import_catalog.py     CSV -> PostgreSQL (xoá sạch + nạp, một transaction)
scripts/gen_seed_sql.py       CSV -> db/init/02_seed.sql
scripts/gen_group_scores.py   CSV -> api/artifacts/matrix/category_room_scores.json
scripts/verify.sh             kiểm tra toàn vẹn catalog + artifact
api/artifacts/                model .pt + 2 file ma trận (bắt buộc khi chạy)
api/                          FastAPI: app/ + services/ + tests/
```

API đọc **bốn nguồn**, không có dự phòng cho nguồn nào:

| Nguồn | Đường dẫn | Thiếu thì |
| --- | --- | --- |
| Checkpoint | `MODEL_PATH` (`artifacts/rtdetr_room_detection/weights/best.pt`) | dừng, báo đường dẫn |
| Ma trận đồng xuất hiện | `COOCCURRENCE_MATRIX_PATH` | dừng, báo đường dẫn |
| Bảng W(c, r) | `CATEGORY_ROOM_SCORES_PATH` | dừng, báo đường dẫn |
| Catalog | PostgreSQL `mwg` | dừng, báo host/port/lỗi |

Trước đây API có nhánh dự phòng đọc CSV khi DB chết. Nhánh đó đã bị bỏ: chạy với
catalog khác nguồn chuẩn mà không ai biết còn tệ hơn là không chạy.

Hai file trong `db/init` chỉ chạy **một lần** khi volume còn trống. Đã có dữ liệu
rồi thì `make import` (nạp lại, không xoá volume) hoặc `make reset` (xoá volume).

## Dữ liệu

`data/dienmayxanh_products.csv` — 1.413 sản phẩm, 25 sản phẩm/nhóm (vài nhóm ít
hơn), không có `product_id` trùng, không thiếu `brand`. `api/data/` giữ một bản sao
để image Docker có CSV dự phòng khi không kết nối được DB.

### Schema `mwg` — 3 bảng

```
categories ──< product >── brand
```

| Bảng | Khoá | Nội dung |
| --- | --- | --- |
| `categories` | `category_id` (= cột `subcategory`: `tv`, `speaker`, …) | 62 nhóm, `category_name` (= `group_name`), `parent_category_code` (5 mục lớn) |
| `brand` | `brand_id` (IDENTITY) | 157 hãng, `brand_name` UNIQUE |
| `product` | `product_id` | 1.413 sản phẩm, FK `category_id` + `brand_id`, `colors`/`tags`/`style`/`placement`/`dimensions` dạng JSONB |

`category_id` của `mwg.product` là **subcategory** của CSV (62 nhóm), không phải 5
mục lớn. 5 mục lớn nằm ở `categories.parent_category_code`, nhờ vậy một nhóm vừa
là đơn vị xếp hạng của recommender vừa giữ được phân cấp ngành hàng.

`room_weights` **không** có trong DB — schema 3 bảng không có cột này. Điểm
W(c, r) đọc từ artifact `api/artifacts/matrix/category_room_scores.json`.

Cột JSON giữ đúng dạng của CSV — `colors` là mảng object `{color_name, color_code}`,
`tags`/`style`/`placement` là mảng chuỗi, `dimensions` là object (nguồn đang rỗng).
Kiểm tra bằng JSONB containment:

```sql
SELECT product_name FROM mwg.product
WHERE tags @> '["tv"]' LIMIT 5;
```

### Điểm phù hợp phòng W(c, r) — đọc từ artifact

`api/artifacts/matrix/category_room_scores.json` (Cell 10.7 của notebook) chứa 62
nhóm × 7 phòng, khoá ngoài là **tên hiển thị** (`"TV"`, `"Loa"`, `"Máy điều hòa"`…).
Vì khoá là tên, `bootstrap` ghép nó với `mwg.categories` (`category_name` ->
`category_id`) để có bảng theo `group_code`.

```json
{
  "TV": {"living_room": 1.0, "bedroom": 0.8, "kitchen": 0.1, "bathroom": 0.0,
         "dining_room": 0.4, "home_office": 0.4, "balcony": 0.0},
  "Loa": {"living_room": 1.0, "bedroom": 0.6, "kitchen": 0.4, "bathroom": 0.0,
          "dining_room": 0.6, "home_office": 0.8, "balcony": 0.4}
}
```

Thiếu file, thiếu phòng, hay điểm ngoài `[0, 1]` đều làm API **dừng khởi động** với
thông báo chỉ đúng nhóm/phòng sai — không có bảng điểm dự phòng trong code.

Sửa điểm: sửa CSV rồi `make scores`, hoặc sửa thẳng file JSON, rồi khởi động lại API.

## Luồng gợi ý

```
ảnh ──RT-DETR──> vật thể (Sofa, TV, Wall Art…)
                      │
                      ├─ P_cooc(c|D) = 1 − Π(1 − P̂(π(c)|d))   ← cooccurrence_matrix.json
                      ├─ W(c, r)                                ← category_room_scores.json
                      └─ final = P_cooc + W(c, r)  → sắp giảm dần → gắn sản phẩm
```

- Ứng viên: 62 nhóm có sản phẩm trong `mwg.product`.
- Loại nhóm đã có trong ảnh (có TV rồi thì không gợi ý TV).
- Chỉ 22 nhóm có detector proxy (`GROUP_TO_DETECTOR`); 40 nhóm còn lại
  `P_cooc = 0` và xếp hạng thuần theo `W(c, r)` — detector là COCO nội thất, không
  nhận ra tủ lạnh hay nồi cơm.
- Sản phẩm trả về: còn hàng, sắp theo giá tăng dần, lọc được theo `min_price`/
  `max_price` và `category_id`.

Chi tiết công thức + API: xem [`api/README.md`](api/README.md).

## Lệnh thường dùng

```bash
make up        # dựng DB + API, chờ /health/ready
make import    # xoá sạch DB, nạp lại từ data/dienmayxanh_products.csv
make verify    # kiểm tra toàn vẹn catalog (số dòng, FK, CHECK) + artifact
make psql      # psql vào mwgshop
make logs      # log API
make backup    # backups/<ts>.dump
make reset     # XOÁ volume, dựng lại DB từ db/init
```

Đổi cổng/credentials trong `.env` (copy từ `.env.example`). Cổng host 55432 tránh
đụng Postgres local ở 5432.

## Câu truy vấn mẫu

```sql
-- Số sản phẩm theo 5 mục lớn (parent category)
SELECT parent_category_code, count(*) AS products, count(DISTINCT p.category_id) AS groups
FROM mwg.product p
JOIN mwg.categories c ON c.category_id = p.category_id
GROUP BY 1 ORDER BY 2 DESC;

-- Top 5 TV đắt nhất còn hàng
SELECT p.product_name, b.brand_name, p.price
FROM mwg.product p
JOIN mwg.brand b ON b.brand_id = p.brand_id
WHERE p.category_id = 'tv' AND p.availability
ORDER BY p.price DESC LIMIT 5;

-- Nhóm có nhiều sản phẩm nhất
SELECT c.category_name, p.category_id, count(*) AS n
FROM mwg.product p
JOIN mwg.categories c ON c.category_id = p.category_id
GROUP BY 1, 2 ORDER BY 3 DESC LIMIT 10;
```

## Ghi chú nguồn

- Trường `brand` của bản ghi này **không sạch**: 348/1.413 giá trị là từ đầu tiên
  của tên sản phẩm (`"Quạt"`, `"Nồi"`, `"Loa"`, `"Camera"`, `"Cây"`…) chứ không
  phải tên hãng; một số khác là mã model (`"LTH-200W"`) hoặc chính tên nhóm
  (`"Dienmayxanh"`). Giữ nguyên vì đây là dữ liệu nguồn — muốn sửa thì làm ở CSV
  rồi `make import` lại.
- `availability` toàn bộ `TRUE`; `colors` toàn bộ một màu `Đen`; `style` toàn bộ
  `["modern"]`; `placement` toàn bộ `["floor"]`; `dimensions` toàn bộ `{}`.
  Đây là dữ liệu chưa được làm giàu, không phải lỗi import.
- `description` 1.413/1.413 giá trị trùng `product_name` — nguồn không có mô tả riêng.
