# Gợi ý và Phối cảnh Đồ điện máy từ Ảnh phòng (Recommend by Image v5.2)

Hệ thống AI toàn diện: **Gợi ý sản phẩm điện máy (MWG / Điện Máy Xanh) theo không gian phòng** và **Tự động phối cảnh ghép sản phẩm 3D chân thực lên ảnh phòng** (Spatial Placement & Relighting Render).

Hệ sinh thái dịch vụ triển khai qua Docker:
- **FastAPI Backend (Python 3.14 / PyTorch)**: Tích hợp RF-DETR, SAM 2, Grounding DINO, Depth Anything V2 và thuật toán tối ưu Noisy-OR.
- **PostgreSQL 16**: Quản lý catalog 1.413 sản phẩm đồ điện máy thuộc schema `mwg` (categories, brand, product).
- **Adminer**: Giao diện trực quan quản trị cơ sở dữ liệu.

---

## 📌 Bảng thông tin Dịch vụ

| Dịch vụ | Địa chỉ truy cập | Thông tin đăng nhập / Ghi chú |
| :--- | :--- | :--- |
| **API Backend** | `http://localhost:8000` | Swagger UI: [`/docs`](http://localhost:8000/docs) · ReDoc: [`/redoc`](http://localhost:8000/redoc) · OpenAPI: [`/openapi.json`](http://localhost:8000/openapi.json) |
| **PostgreSQL 16** | `localhost:55432` | User: `mwg` / Pass: `mwg_secret` / DB: `mwgshop` (schema `mwg`) |
| **Adminer** | `http://localhost:58080` | System: `PostgreSQL` / Server: `postgres` / User: `mwg` / Pass: `mwg_secret` / DB: `mwgshop` |

---

## 🤖 Thông tin AI Models & Foundation Models

Hệ thống kết hợp 4 mô hình AI thị giác máy tính tiên tiến nhất để thực hiện pipeline từ nhận diện không gian đến định vị và render:

| Tên Model | Nhiệm vụ chính trong dự án | Kiến trúc / Model ID | Link Model | Giấy phép (License) |
| :--- | :--- | :--- | :--- | :--- |
| **RF-DETR Medium** | Phát hiện vật thể hiện hữu trong phòng (Sofa, TV, Giường, Tủ,...) để làm điều kiện tiên nghiệm cho ma trận đồng xuất hiện | `Roboflow/rf-detr-medium`<br>*(DINOv2 backbone)* | [Hugging Face](https://huggingface.co/Roboflow/rf-detr-medium) · [GitHub](https://github.com/roboflow/rf-detr) | **Apache-2.0** |
| **SAM 2 Small**<br>*(Segment Anything 2)* | Phân đoạn chính xác mặt phẳng (tường, sàn, trần, mặt bàn) và tạo mặt nạ vật cản chi tiết cho thuật toán placement | `facebook/sam2-hiera-small`<br>*(Hiera-Small)* | [Hugging Face](https://huggingface.co/facebook/sam2-hiera-small) · [GitHub](https://github.com/facebookresearch/sam2) | **Apache-2.0** |
| **Grounding DINO Base** | Zero-shot open-vocabulary detection để tìm các vùng bề mặt phòng (`wall`, `floor`, `ceiling`, `table`) làm prompt box cho SAM 2 | `IDEA-Research/grounding-dino-base` | [Hugging Face](https://huggingface.co/IDEA-Research/grounding-dino-base) · [GitHub](https://github.com/IDEA-Research/GroundingDINO) | **Apache-2.0** |
| **Depth Anything V2 Small** | Ước lượng bản đồ chiều sâu đơn mục (Monocular Depth Estimation), xác định độ sâu tương đối, tỷ lệ phối cảnh và thứ tự che khuất | `depth-anything/Depth-Anything-V2-Small-hf` | [Hugging Face](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf) · [GitHub](https://github.com/DepthAnything/Depth-Anything-V2) | **Apache-2.0** |
| **FLUX.2 Klein 4B**<br>*(Black Forest Labs)* | Hoàn thiện ảnh phối cảnh sinh thực tế (Generative Harmonization & Relighting): tái tạo ánh sáng, bóng đổ tiếp xúc và khử viền ghép | `black-forest-labs/FLUX.2-klein-4B`<br>*(Distilled 4-Step Flow)* | [Hugging Face](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) · [BFL](https://bfl.ai) | **Apache-2.0** |

> Ngoài ra, hệ thống tích hợp API tách nền chuẩn nội bộ (`REMOVE_BG_BASE_URL=https://ai.itmwg.vn/api/remove-background`) để trích xuất vật thể không nền từ catalog phục vụ ghép ảnh.

---

## 📜 Giấy phép bản quyền (Licenses)

- **Mã nguồn Dự án**: Toàn bộ mã nguồn dự án được phân phối dưới giấy phép [**MIT License**](LICENSE).
- **Mô hình AI bên thứ ba**:
  - `Roboflow/rf-detr-medium`: **Apache-2.0 License**
  - `facebook/sam2-hiera-small`: **Apache-2.0 License**
  - `IDEA-Research/grounding-dino-base`: **Apache-2.0 License**
  - `depth-anything/Depth-Anything-V2-Small-hf`: **Apache-2.0 License**
  - `black-forest-labs/FLUX.2-klein-4B`: **Apache-2.0 License**
- **Dữ liệu Catalog**: Dữ liệu sản phẩm mẫu (1.413 sản phẩm Điện Máy Xanh) phục vụ mục đích nghiên cứu, học thuật và phát triển tính năng.

---

## 🏗️ Kiến trúc & Luồng xử lý (Pipeline)

Quy trình xử lý ảnh phòng gồm 2 giai đoạn:

```
                  ┌────────────────────────┐
                  │ Ảnh chụp phòng tải lên │
                  └───────────┬────────────┘
                              │
             ┌────────────────┴────────────────┐
             ▼                                 ▼
   [ GIAI ĐOẠN 1: GỢI Ý ]            [ GIAI ĐOẠN 2: PHỐI CẢNH ]
   RF-DETR Medium                     Grounding DINO + SAM 2 + Depth Anything V2
         │                                     │
         ▼                                     ▼
Phát hiện đồ vật hiện hữu             Phân đoạn Mặt phẳng (Tường/Sàn/Trần/Bàn)
         │                            & Ước lượng Chiều sâu (Depth Map)
         ▼                                     │
Noisy-OR Co-occurrence                         ▼
  + Room Score W(c, r)                Tìm vị trí đặt tối ưu (Collision, Occlusion)
         │                                     │
         ▼                                     ▼
Truy vấn PostgreSQL Catalog           Gọi API Remove Background tách nền sản phẩm
         │                                     │
         ▼                                     ▼
Xếp hạng Top-K sản phẩm               Phối cảnh 3D, Cân bằng sáng (Relighting)
         │                            & Đổ bóng tiếp xúc (Contact Shadow)
         │                                     │
         ▼                                     ▼
 📤 JSON Danh sách gợi ý               🖼️ Ảnh PNG Render hoàn thiện
```

### 1. Giai đoạn Gợi ý (Recommendation Pipeline)
- **Công thức xác suất điều kiện Noisy-OR**:
  $$\hat{P}(e \mid d) = \frac{N(d \cap e) + \alpha \cdot P(e)}{N(d) + \alpha K}, \quad K=2,\ \alpha=1$$
  $$P_{\text{cooc}}(c \mid D) = 1 - \prod_{d \in D}\left(1 - \hat{P}(\pi(c) \mid d)\right)$$
- **Điểm tổng hợp**:
  $$\text{final\_score}(c) = P_{\text{cooc}}(c \mid D) + W(c, r)$$
  Trong đó $W(c, r)$ là độ tương thích giữa nhóm sản phẩm $c$ và loại phòng $r$ (`living_room`, `bedroom`, `kitchen`,...).

### 2. Giai đoạn Render Không gian (Spatial Placement & Photometric Render)
- **Xác định bề mặt hợp lệ**: Dựa trên `placement_rules.json` (ví dụ: TV treo tường, Loa đặt sàn/bàn, Máy lạnh gắn sát trần tường).
- **Tránh va chạm (Collision Avoidance)**: Loại trừ các bounding box đã phát hiện và vùng che chắn.
- **Tính toán tỷ lệ kích thước (Adaptive Sizing)**: Tự động scale kích thước theo depth tương đối (ví dụ: cặp tỉ lệ tương đối giữa TV và Điều hòa nhiệt độ).
- **Relighting & Shadowing**: Cân bằng phơi sáng tự động theo ánh sáng nền phòng và tạo bóng đổ tiếp xúc thực tế.

---

## ⚡ Khởi động nhanh (Quick Start)

### Yêu cầu hệ thống
- **Docker & Docker Compose** (khuyên dùng RAM $\ge$ 8GB cho container AI).
- Checkpoint và artifacts đã đặt đầy đủ trong thư mục `api/artifacts/`.

### Khởi chạy một lệnh qua Docker

```bash
# 1. Sao chép file môi trường
cp .env.example .env

# 2. Khởi động PostgreSQL, FastAPI Backend và Adminer
docker compose up -d --build

# 3. Kiểm tra trạng thái sẵn sàng (Healthcheck)
curl -s http://localhost:8000/health/ready | python3 -m json.tool
```

Khi sẵn sàng, endpoint `/health/ready` sẽ trả về trạng thái của cả 4 nguồn tài nguyên:
```json
{
  "ready": true,
  "model_source": "artifacts/rfdetr_medium (rfdetr)",
  "cooccurrence_source": "cooccurrence_matrix.json (45 class)",
  "room_scores_source": "artifact category_room_scores.json (62 nhóm × 7 phòng) + mwg.categories",
  "catalog_source": "PostgreSQL mwg.product (1413 sản phẩm)"
}
```

---

## 📡 API Endpoints

### 1. `POST /api/v1/recommendations/image`
Upload ảnh phòng và nhận danh sách các nhóm sản phẩm gợi ý kèm thông tin chi tiết từng sản phẩm.

```bash
curl -X POST "http://localhost:8000/api/v1/recommendations/image?room_type=living_room&top_k=5" \
     -F "image=@room_sample.jpg"
```

**Các tham số Query chính:**
- `room_type`: Loại phòng (`living_room`, `bedroom`, `kitchen`, `dining_room`, `bathroom`, `home_office`, `balcony`). Mặc định: `living_room`.
- `top_k`: Số lượng nhóm sản phẩm tối đa trả về (1–62). Mặc định: 10.
- `min_score`: Ngưỡng điểm tối thiểu để được gợi ý (mặc định: 0.05).
- `exclude_detected`: Loại bỏ các nhóm đồ đã xuất hiện trong ảnh (true/false, mặc định: true).
- `products_per_group`: Số lượng sản phẩm trả về trong mỗi nhóm (mặc định: 3).
- `min_price` / `max_price`: Lọc khoảng giá sản phẩm (VND).
- `category_id`: Lọc theo ngành hàng lớn (`electronics`, `appliances`,...).
- `confidence`: Ngưỡng tin cậy của object detector (mặc định: 0.5).

### 2. `POST /api/v1/recommendations/render`
Upload ảnh phòng, hệ thống tự động phát hiện, gợi ý, tách nền sản phẩm và render phối cảnh trực tiếp lên ảnh. Kết quả trả về là luồng nhị phân ảnh **PNG**.

```bash
curl -X POST "http://localhost:8000/api/v1/recommendations/render?room_type=living_room&top_k=3" \
     -F "image=@room_sample.jpg" \
     --output result_rendered.png
```

### 3. `GET /health/ready`
Kiểm tra tính sẵn sàng của hệ thống (Model, Ma trận xác suất, Điểm phòng, Kết nối PostgreSQL).

---

## 📁 Cấu trúc Thư mục Dự án

```
recommend-by-image-v5-2/
├── LICENSE                        # Giấy phép mã nguồn MIT
├── README.md                      # Tài liệu tổng quan dự án
├── Makefile                       # Phím tắt các lệnh quản trị tiện lợi
├── docker-compose.yml             # Cấu hình container: postgres, api, adminer
├── .env.example                   # Mẫu cấu hình môi trường
├── data/
│   └── dienmayxanh_products.csv   # Catalog 1.413 sản phẩm, 62 nhóm, 157 hãng
├── db/
│   └── init/
│       ├── 01_schema.sql          # Khởi tạo schema mwg (categories, brand, product)
│       └── 02_seed.sql            # Nạp dữ liệu 1.413 sản phẩm
├── scripts/
│   ├── import_catalog.py          # Script nạp CSV vào DB
│   ├── gen_seed_sql.py            # Sinh seed SQL từ CSV
│   ├── gen_group_scores.py        # Sinh ma trận category_room_scores.json
│   └── verify.sh                  # Kiểm tra tính toàn vẹn DB & Artifacts
└── api/
    ├── README.md                  # Hướng dẫn chi tiết kỹ thuật cho backend
    ├── Dockerfile                 # Đóng gói image FastAPI (Python 3.14-slim)
    ├── requirements.txt           # Danh sách thư viện Python
    ├── artifacts/                 # Weights các mô hình AI và ma trận tính toán
    │   ├── rfdetr_medium/         # RF-DETR weights (model.safetensors, config)
    │   ├── sam2_small/            # SAM 2 weights & processor
    │   ├── grounding_dino_base/   # Grounding DINO weights & tokenizer
    │   ├── depth_anything_v2_small/ # Depth Anything V2 weights
    │   ├── placement_rules.json   # Bộ luật quy định vị trí và kích thước gắn vật thể
    │   └── matrix/
    │       ├── cooccurrence_matrix.json  # Ma trận đồng xuất hiện COCO
    │       └── category_room_scores.json # Bảng điểm tương thích phòng W(c,r)
    ├── app/
    │   ├── main.py                # Điểm khởi tạo FastAPI app
    │   ├── core/                  # Cấu hình Settings và Exception Handlers
    │   ├── api/routes/            # Endpoints: recommendations, health
    │   └── services/              # Các service AI: detector, placement, recommender,...
    └── tests/                     # 65+ unit tests kiểm thử logic & API
```

---

## 🛠️ Makefile Cheatsheet

```bash
make up        # Dựng Docker container (DB + API + Adminer) và chờ sẵn sàng
make down      # Dừng và hạ containers (giữ nguyên volume dữ liệu)
make logs      # Theo dõi log thời gian thực của API
make import    # Xoá sạch DB và nạp lại toàn bộ từ CSV
make verify    # Kiểm tra toàn vẹn dữ liệu database và sự hiện diện của model artifacts
make test      # Chạy test suite (pytest) trong thư mục api/
make psql      # Mở terminal psql tương tác trực tiếp vào database mwgshop
make reset     # Xoá hoàn toàn volume cơ sở dữ liệu và khởi tạo lại từ đầu
```

---

## 📚 Thông tin Tham chiếu & Tài liệu Kỹ thuật

- Thông số đánh giá trên tập kiểm thử (Cell 10b Validation):
  - Phương pháp Noisy-OR smoothed ($\alpha=1$): **AP = 0.3806**, **AUC = 0.9397** (vượt trội so với trung bình thô AP = 0.3276).
  - Đối với các nhóm proxy ít xuất hiện (support < 50): AP tăng vọt từ **0.1980 $\rightarrow$ 0.3662**.
- Chi tiết kỹ thuật về các hàm giải thuật: Xem tiếp tại [**`api/README.md`**](api/README.md).
