# AI Backend — RT-DETR room recommendation (Cell 1 → 10.14)

FastAPI port của notebook `yolo_train_predict_v5_spatial_relationships_rtdetr.ipynb`,
**chỉ tới Cell 10.14** (pipeline gợi ý từ ảnh). Cell 10b (đánh giá AP/AUC) **không
được port** — API chỉ phục vụ hai endpoint.

## Không có bước train

Cell 8 (`model.train(...)`) **không được port**. Model chỉ được **đọc từ file `.pt`**:

```python
# Cell 9 của notebook
best_model_path = rtdetr_checkpoints[0]
model = RTDETR(str(best_model_path))
```

API tái tạo đúng việc đó, tự nhận diện kiến trúc bên trong checkpoint:

| class trong checkpoint | Loader dùng |
| --- | --- |
| `RTDETRDetectionModel` | `ultralytics.RTDETR` |
| `DetectionModel` | `ultralytics.YOLO` |

Nên cùng một `MODEL_PATH` chạy được cả checkpoint RT-DETR lẫn YOLO cũ.

## Bản đồ cell → module

| Notebook | Nội dung | Module |
| --- | --- | --- |
| Cell 1–5 | path, data layout | `app/core/config.py` |
| Cell 6 | `CLASS_MERGE_MAP`, `DROP_CLASSES`, `build_class_map` | `app/services/cooccurrence.py` |
| Cell 7 | `data.yaml` | dùng để suy ra bộ class (`expected_detector_classes`) |
| **Cell 8** | **train** | **KHÔNG port** |
| Cell 9 | load `.pt` + kiểm tra class | `app/services/detector.py` |
| Cell 10.1 | hằng số cấu hình | `app/core/config.py` |
| Cell 10.2 | `GROUP_TO_DETECTOR`, `DETECTOR_TO_GROUP` | `app/services/recommender.py` |
| Cell 10.3 | `build_cooccurrence_matrix` | `app/services/cooccurrence.py` |
| Cell 10.4 | `P`, `P_smoothed`, `BASE_RATE`, `N` | `Recommender.p_smoothed` |
| Cell 10.5 | lưu `cooccurrence_matrix.json` | `scripts/build_artifacts.py` |
| Cell 10.6 | `W(c, r)` | `app/services/room_scores.py` (đọc artifact JSON) |
| Cell 10.7 | hàm `W` | `Recommender.w` |
| Cell 10.8 | `load_catalog` | `app/services/product_catalog.py` (PostgreSQL → CSV) |
| Cell 10.9 | Noisy-OR, `build_reason` | `Recommender.cooccurrence_score` |
| Cell 10.10 | `recommend()` | `Recommender.recommend` |
| Cell 10.11 / 10.15 | hiển thị | bỏ (API trả JSON) |
| Cell 10.12 | `load_image` | `app/services/images.py` |
| Cell 10.13 | `detect_objects` | `Detector.detect` |
| Cell 10.14 | pipeline | `POST /api/v1/recommendations/image` |
| Cell 10b.1–10b.3 | đánh giá AP/AUC | **không port** (số liệu vẫn ở mục dưới) |
| Cell 10c trở đi | spatial placement | **ngoài phạm vi** |

### Đường dẫn artifact

| Biến | Giá trị | Ghi chú |
| --- | --- | --- |
| `MODEL_PATH` | `artifacts/rtdetr_room_detection/weights/best.pt` | run `rtdetr_room_detection` |
| `COOCCURRENCE_MATRIX_PATH` | `artifacts/matrix/cooccurrence_matrix.json` | Cell 10.3 + 10.5 |
| `CATEGORY_ROOM_SCORES_PATH` | `artifacts/matrix/category_room_scores.json` | W(c,r), bắt buộc |
| `DB_HOST` / `DB_PORT` | `postgres` / `5432` | service trong compose gốc |
| `DB_SCHEMA` | `mwg` | schema chứa 3 bảng (`product`, `categories`, `brand`) |

`W(c, r)` đọc từ `artifacts/matrix/category_room_scores.json` (Cell 10.7) qua
`app/services/room_scores.py`. File khoá theo **tên nhóm hiển thị**, nên bootstrap
ghép với `mwg.categories` để ra bảng theo `group_code`. Sinh lại bằng
`python3 ../scripts/gen_group_scores.py`; đổi điểm thì sửa CSV rồi chạy lại.

## Dữ liệu: PostgreSQL là nguồn chính

Catalog 1.413 sản phẩm nằm trong schema `mwg` của PostgreSQL (đúng 3 bảng:
`product`, `categories`, `brand`). API join `product` + `brand` lúc khởi động.
**Không còn fallback CSV**: không kết nối được thì khởi động thất bại với
`DataSourceError` nêu rõ host/port/database và loại lỗi.

```bash
# Từ thư mục gốc repo
docker compose up -d --build     # postgres + api + adminer
curl -s localhost:8000/health/ready | python3 -m json.tool
```

`catalog_source` phải là `PostgreSQL mwg.product (1413 sản phẩm)`. Sai cổng/mật
khẩu/schema thì container dừng ngay và `docker compose logs api` ghi rõ lỗi.

| Lệnh (thư mục gốc) | Việc |
| --- | --- |
| `make up` | dựng DB + API |
| `make import` | xoá sạch DB, nạp lại từ `data/dienmayxanh_products.csv` |
| `make seed` | sinh lại `db/init/02_seed.sql` từ CSV |
| `make scores` | sinh lại `api/artifacts/matrix/category_room_scores.json` từ CSV |
| `make verify` | kiểm tra toàn vẹn catalog |

## Chạy

### Docker (một lệnh, từ thư mục gốc repo)

```bash
docker compose up -d --build     # postgres + api + adminer
```

hoặc `make up`. Compose gốc (`../docker-compose.yml`) dựng cả DB lẫn API trong cùng
network, API gọi DB qua hostname `postgres`. DB tự nạp 1.413 sản phẩm từ
`db/init/02_seed.sql` khi volume còn trống.

(`api/.env` chỉ dùng cho chạy local, không dùng trong container — biến của container
khai sẵn trong `docker-compose.yml`.)

Mở http://127.0.0.1:8000/docs. Container tự chứa checkpoint `.pt`, ma trận đồng
xuất hiện, annotation COCO; sản phẩm đọc từ PostgreSQL.

Lệnh hữu ích (chạy từ **thư mục gốc repo**):

| Lệnh | Việc |
| --- | --- |
| `make up` | build nếu cần + chạy DB & API + chờ ready |
| `make logs` | xem log API |
| `make import` | xoá sạch DB, nạp lại từ CSV |
| `make verify` | kiểm tra toàn vẹn catalog |
| `make down` | dừng, xoá container (giữ volume) |
| `make reset` | xoá volume rồi dựng lại DB |

Đổi cổng: `API_PORT=9000 make up` (hoặc sửa `API_PORT` trong `.env`).

### Local (không Docker)

```bash
cd api
cp .env.example .env          # DB_* trỏ về 127.0.0.1:55432 (DB trong docker)
# muốn chạy không cần DB: đặt DB_HOST= rỗng -> tự dùng CSV

# 1) Sinh ma trận đồng xuất hiện (Cell 10.3 + 10.5) từ COCO train
python scripts/build_artifacts.py

# 2) Trỏ MODEL_PATH tới checkpoint .pt có sẵn (Cell 9). API không train.
#    Mặc định đã trỏ vào run train sẵn có:
#    artifacts/rtdetr_room_detection/weights/best.pt
python scripts/build_artifacts.py --model /duong/dan/rtdetr_room_detection/weights/best.pt

# 3) Chạy
uvicorn app.main:get_app --factory --host 127.0.0.1 --port 8000
```

Muốn chạy hoàn toàn không cần PostgreSQL: để `DB_HOST=` rỗng, API tự đọc
`data/dienmayxanh_products.csv` và `/health/ready` sẽ báo `catalog_source` là CSV.


Swagger: http://127.0.0.1:8000/docs · ReDoc: http://127.0.0.1:8000/redoc ·
OpenAPI: http://127.0.0.1:8000/openapi.json

## Tài liệu API (Swagger)

- `/docs` — **Swagger UI**, có nút *Try it out* gọi thẳng API, ví dụ request điền sẵn.
- `/redoc` — **ReDoc**, đọc hợp đồng API dạng tài liệu.
- `/openapi.json` — schema OpenAPI 3.1 (dùng để code-gen client).

Tài liệu được sinh tự động từ Pydantic + docstring. Mô tả cố ý **ngắn** (một dòng
mỗi mục); công thức và hướng dẫn chi tiết nằm ở README này, không lặp lại trong
Swagger:

| Nội dung | Ở đâu |
| --- | --- |
| Mô tả tổng (một câu) | `API_DESCRIPTION` trong `app/api/openapi.py` |
| Mô tả từng nhóm endpoint | `TAGS_METADATA` |
| Mô tả route + tham số | `description` trong `app/api/routes/*.py` |
| Ví dụ request/response | `model_config` trong `app/schemas/recommendation.py` |
| Phản hồi lỗi dùng chung (`$ref` tới `ErrorResponse`) | `ERROR_CONTENT` |

`tests/test_docs.py` chặn mô tả phình to: `info.description` ≤ 300 ký tự,
`description` của operation ≤ 200 ký tự và chỉ một dòng, mô tả field ≤ 120 ký tự.

Tắt tài liệu khi production: `DOCS_ENABLED=false` → `/docs`, `/redoc`,
`/openapi.json` đều trả 404.

## Endpoint

Chỉ hai endpoint:

| Method | Path | Cell | Mô tả |
| --- | --- | --- | --- |
| POST | `/api/v1/recommendations/image` | 10.12 + 10.13 + 10.10 | Upload ảnh → detect → gợi ý |
| GET | `/health/ready` | — | Model + dữ liệu đã nạp xong (503 nếu chưa) |

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/recommendations/image?room_type=living_room&top_k=5" \
     -F "image=@phong-khach.jpg"
```

Không còn endpoint để liệt kê `room_type`, `category_id` hay class detector. Danh
sách `room_type` hợp lệ là `ROOM_CODES` trong `app/services/room_scores.py` (7 giá
trị: `balcony`, `bathroom`, `bedroom`, `dining_room`, `home_office`, `kitchen`,
`living_room`); `category_id` là 5 khoá của `CATEGORY_NAMES`. Truyền sai sẽ nhận
`404 not_found` kèm danh sách hợp lệ trong `message`.

Tham số `category_id` (tuỳ chọn) giới hạn kết quả trong một ngành hàng — hữu ích
khi muốn "chỉ đồ nhà bếp" thay vì trộn cả catalog.

## Công thức (giữ nguyên notebook)

\[
\hat{P}(e \mid d) = \frac{N(d \cap e) + \alpha \cdot P(e)}{N(d) + \alpha K}, \quad K=2,\ \alpha=1
\]

\[
P_{\text{cooc}}(c \mid D) = 1 - \prod_{d \in D}\left(1 - \hat{P}(\pi(c) \mid d)\right)
\]

\[
\text{final\_score}(c) = P_{\text{cooc}}(c \mid D) + W(c, r)
\]

Lưu ý: Cell 10.10 dùng **phép cộng** (`cooc + room_score`), không phải nhân.

## Kết quả Cell 10b.3 (tham chiếu — chạy trên split valid)

Số liệu dưới đây là kết quả đánh giá của notebook, **giữ lại để tham chiếu**; API
không còn endpoint đánh giá (Cell 10b không được port).

| Công thức | AP | AUC |
| --- | --- | --- |
| Trung bình thô (cũ) | 0.3276 | 0.9263 |
| Noisy-OR + α=1 (mới) | **0.3806** | **0.9397** |

Proxy hiếm (support < 50): AP **0.1980 → 0.3662**, đúng như số liệu ghi trong notebook.

## Dữ liệu & artifact

- `artifacts/matrix/cooccurrence_matrix.json` — sinh bởi `scripts/build_artifacts.py`
- `artifacts/rtdetr_room_detection/weights/best.pt` — checkpoint RT-DETR **copy sẵn**
  (không sinh ra từ API); `last.pt` cùng thư mục là bản của epoch cuối
- `data/coco/train/_annotations.coco.json` — bản sao annotation COCO (2.1 MB), đủ
  cho `expected_detector_classes`; không cần 82 MB ảnh.
- `data/dienmayxanh_products.csv` — 1.413 sản phẩm, 62 nhóm, 157 brand, 5 category
- `db/init/01_schema.sql` — schema `mwg` (3 bảng + index + CHECK)
- `db/init/02_seed.sql` — dữ liệu sinh từ CSV, chạy một lần khi volume trống
- `app/services/room_scores.py` — đọc `W(c, r)` từ artifact JSON, kiểm tra 7 phòng/nhóm
  ma trận từ ảnh (`scripts/build_artifacts.py`)
- `W(c, r)` — `artifacts/matrix/category_room_scores.json`

Trong container mọi đường dẫn tính từ `/app`, nên `.env` không dùng cho Docker —
biến môi trường được khai trong `docker-compose.yml`.

## Test

```bash
python -m pytest tests -q     # 65 test: công thức, xếp hạng, catalog DB/CSV, endpoint, OpenAPI
```

## Ghi chú

- `REQUIRE_CLASS_MATCH` — kiểm tra checkpoint có đúng 45 class như Cell 9, đối chiếu
  với `cooccurrence_matrix.json["classes"]` (không cần thư mục COCO). Bật `true`;
  lớp bảo vệ này báo lỗi nếu sau này thay nhầm checkpoint.
- Ma trận có **45 class** nhưng `Phone` và `Power Surge Protector` có
  `support = 0` (không có annotation trong train), nên `P̂` của chúng chỉ còn prior.
  Đây là hành vi đúng của Cell 10.3, không phải lỗi.
- **Không có fallback.** Thiếu checkpoint / ma trận / file điểm phòng / DB thì
  `build_state` raise `ArtifactError` hoặc `DataSourceError`, container dừng và
  `docker compose logs api` ghi đúng đường dẫn hoặc bảng thiếu. Đây là lựa chọn có
  chủ ý: chạy với nguồn khác chuẩn mà không ai biết còn tệ hơn là không chạy.
- `W(c, r)` là **giả định thiết kế**, không phải dữ liệu đo. File JSON khoá theo
  tên hiển thị; nếu đổi tên trong `mwg.categories` mà không sinh lại artifact thì
  bootstrap báo `thiếu điểm phòng cho N nhóm` kèm danh sách.
- Nhóm **không có detector proxy** (phần lớn catalog: tủ lạnh, nồi cơm, máy giặt…)
  có `P_cooc = 0` và chỉ xếp theo `W(c, r)`. Đây là giới hạn của detector COCO nội
  thất, không phải lỗi: danh sách proxy nằm ở `GROUP_TO_DETECTOR`.

### Docker

- `python:3.14-slim-trixie` + torch/torchvision **bản CPU** từ index `+cpu` của
  PyTorch (tránh kéo CUDA ~2.5 GB). Image ~2.5 GB.
- Base image trixie đổi tên `libglib2.0-0` → `libglib2.0-0t64`.
- Chạy bằng user `app` (uid 1000), không phải root.
- `HEALTHCHECK` gọi `/health/ready` để orchestration chỉ coi container là sẵn sàng
  khi model đã nạp + warmup xong.
- Lần build đầu tải torch 159 MB. Nếu mạng đứt, `PIP_RETRIES=10` giúp thử lại;
  build lại sẽ dùng cache các layer trước.

