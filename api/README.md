# AI Backend — Gợi ý & Phối cảnh Không gian Phòng (FastAPI)

Hệ thống AI Backend xây dựng trên **FastAPI (Python 3.14)** cung cấp 2 năng lực cốt lõi:
1. **Gợi ý sản phẩm thông minh**: Phân tích đồ nội thất trong ảnh phòng qua detector tiên tiến, áp dụng mô hình xác suất điều kiện Noisy-OR và độ phù hợp không gian phòng để xếp hạng danh mục sản phẩm điện máy từ catalog PostgreSQL MWG.
2. **Phối cảnh 3D & Relighting**: Phân đoạn bề mặt phẳng (tường, sàn, trần, bàn), ước lượng bản đồ chiều sâu (depth map), tự động gọi tách nền sản phẩm và ghép chân thực vào không gian phòng với bóng đổ tiếp xúc và cân bằng sáng.

---

## 🤖 Foundation Models, Checkpoints & Licenses

Hệ thống sử dụng các foundation models hàng đầu trong thị giác máy tính và phân tích không gian. Tất cả weights được lưu trữ tại `api/artifacts/`:

| Mô hình | Định dạng & Thư mục | Nguồn & Link Model | License | Chức năng trong Backend |
| :--- | :--- | :--- | :--- | :--- |
| **RF-DETR Medium** | `artifacts/rfdetr_medium/`<br>`model.safetensors` (~135MB) | [Roboflow/rf-detr-medium](https://huggingface.co/Roboflow/rf-detr-medium)<br>[GitHub](https://github.com/roboflow/rf-detr) | **Apache-2.0** | Object Detector phát hiện các đồ vật nội thất hiện hữu (Sofa, TV, Giường, Bàn, Tủ lạnh...) làm điều kiện tiên nghiệm $d \in D$. |
| **SAM 2 Small**<br>*(Segment Anything 2)* | `artifacts/sam2_small/`<br>`model.safetensors` (~184MB) | [facebook/sam2-hiera-small](https://huggingface.co/facebook/sam2-hiera-small)<br>[GitHub](https://github.com/facebookresearch/sam2) | **Apache-2.0** | Sinh mặt nạ phân đoạn (segmentation mask) độ chính xác cao cho từng bề mặt (sàn, tường, trần, bàn) và vật cản phòng. |
| **Grounding DINO Base** | `artifacts/grounding_dino_base/`<br>`model.safetensors` (~933MB) | [IDEA-Research/grounding-dino-base](https://huggingface.co/IDEA-Research/grounding-dino-base)<br>[GitHub](https://github.com/IDEA-Research/GroundingDINO) | **Apache-2.0** | Open-vocabulary text-prompted detection (`"wall. floor. ceiling. table."`) tạo prompt box chuẩn xác cung cấp cho SAM 2. |
| **Depth Anything V2 Small** | `artifacts/depth_anything_v2_small/`<br>`model.safetensors` (~99MB) | [depth-anything/Depth-Anything-V2-Small-hf](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf)<br>[GitHub](https://github.com/DepthAnything/Depth-Anything-V2) | **Apache-2.0** | Ước lượng bản đồ chiều sâu đơn mục (Monocular Depth), phân tầng z-depth, scale kích thước phối cảnh và xử lý che khuất (occlusion). |
| **FLUX.2 Klein 4B** | `artifacts/flux2_klein_4b/` *(hoặc HF hub)* | [black-forest-labs/FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)<br>[BFL](https://bfl.ai) | **Apache-2.0** | Tái tạo và hòa hợp ánh sáng không gian (Generative Harmonization & Relighting), làm mịn viền cắt dán và đổ bóng chân thực qua 4-step distilled diffusion. |

### Giấy phép bản quyền (Project License)
- Toàn bộ source code của API được phát hành theo giấy phép [**MIT License**](../LICENSE).
- Các mô hình AI nền tảng đi kèm đều sử dụng giấy phép nguồn mở tương thích cao [**Apache-2.0**](http://www.apache.org/licenses/LICENSE-2.0).

---

## 📂 Danh mục Artifacts bắt buộc (`api/artifacts/`)

Backend khởi động với cơ chế kiểm tra toàn vẹn tài nguyên nghiêm ngặt (Strict Resource Verification). Nếu thiếu bất kỳ tệp artifact nào, server sẽ lập tức dừng khởi động và thông báo rõ ràng trong log:

```
api/artifacts/
├── rfdetr_medium/
│   ├── config.json
│   ├── preprocessor_config.json
│   └── model.safetensors                 # Trọng số RF-DETR Medium
├── sam2_small/
│   ├── config.json
│   ├── preprocessor_config.json
│   ├── processor_config.json
│   └── model.safetensors                 # Trọng số SAM 2 Small
├── grounding_dino_base/
│   ├── config.json
│   ├── tokenizer.json, vocab.txt...
│   └── model.safetensors                 # Trọng số Grounding DINO
├── depth_anything_v2_small/
│   ├── config.json
│   ├── preprocessor_config.json
│   └── model.safetensors                 # Trọng số Depth Anything V2
├── placement_rules.json                  # Luật hình học đặt vật thể (mặt phẳng, tỷ lệ, ràng buộc)
└── matrix/
    ├── cooccurrence_matrix.json          # Ma trận đồng xuất hiện COCO (45 class)
    └── category_room_scores.json         # Điểm phù hợp W(c, r) cho 62 nhóm × 7 phòng
```

---

## 🧮 Giải thuật & Công thức Tính toán

### 1. Ma trận đồng xuất hiện Noisy-OR smoothed
Xác suất có mặt của nhóm mục tiêu $e$ khi xuất hiện vật thể nhận diện $d$:
$$
\hat{P}(e \mid d) = \frac{N(d \cap e) + \alpha \cdot P(e)}{N(d) + \alpha K}, \quad K=2,\ \alpha=1
$$

Tổng hợp bằng mô hình Noisy-OR trên tập vật thể phát hiện được trong phòng $D$:
$$
P_{\text{cooc}}(c \mid D) = 1 - \prod_{d \in D}\left(1 - \hat{P}(\pi(c) \mid d)\right)
$$

### 2. Điểm tổng hợp
$$
\text{final\_score}(c) = P_{\text{cooc}}(c \mid D) + W(c, r)
$$
- $W(c, r)$: Trọng số độ phù hợp của nhóm $c$ trong ngữ cảnh phòng $r$ (`living_room`, `bedroom`, `kitchen`, `dining_room`, `bathroom`, `home_office`, `balcony`).
- Nếu trong ảnh đã có sản phẩm thuộc nhóm $c$ (ví dụ: đã phát hiện TV), nhóm này sẽ được tự động loại bỏ (`exclude_detected=true`) để tránh gợi ý trùng lặp.

### 3. Kết quả đối sánh (Validation Reference)
| Phương pháp | AP | AUC |
| :--- | :--- | :--- |
| Trung bình cộng thông thường | 0.3276 | 0.9263 |
| **Noisy-OR + $\alpha=1$ (Hiện tại)** | **0.3806** | **0.9397** |
| Nhóm vật thể hiếm (support < 50) | 0.1980 | **0.3662** |

---

## 🚀 Endpoints & Hướng dẫn Tích hợp

### 1. `POST /api/v1/recommendations/image`
Upload ảnh phòng và nhận danh sách các nhóm sản phẩm được xếp hạng cùng các sản phẩm tương ứng trong catalog.

**Cú pháp cURL mẫu:**
```bash
curl -X POST "http://localhost:8000/api/v1/recommendations/image?room_type=living_room&top_k=5&products_per_group=3" \
     -H "Accept: application/json" \
     -F "image=@room_sample.jpg"
```

**Cấu trúc dữ liệu JSON trả về:**
```json
{
  "room_type": "living_room",
  "detected_objects": [
    {"class_name": "Sofa", "confidence": 0.92, "box": [120, 340, 890, 710]}
  ],
  "recommendations": [
    {
      "category_id": "tv",
      "category_name": "TV",
      "final_score": 1.745,
      "cooccurrence_score": 0.745,
      "room_score": 1.0,
      "reason": "Thường đi kèm với Sofa trong Phòng khách",
      "products": [
        {
          "product_id": "312456",
          "product_name": "Smart Tivi Samsung 4K 55 inch",
          "price": 12490000.0,
          "brand_name": "Samsung",
          "image_url": "https://cdn.tgdd.vn/products/..."
        }
      ]
    }
  ]
}
```

---

### 2. `POST /api/v1/recommendations/render`
Phối cảnh sản phẩm trực tiếp lên ảnh không gian phòng. Tự động tải ảnh sản phẩm từ catalog, gọi API tách nền, căn chỉnh phối cảnh 3D và cân bằng ánh sáng.

**Cú pháp cURL mẫu:**
```bash
curl -X POST "http://localhost:8000/api/v1/recommendations/render?room_type=living_room&top_k=3" \
     -F "image=@room_sample.jpg" \
     --output result_rendered.png
```

- **Đầu ra**: File ảnh định dạng `image/png`.

---

### 3. `GET /health/ready`
Kiểm tra sức khỏe và độ sẵn sàng của hệ thống AI Backend:

```bash
curl -s http://localhost:8000/health/ready
```

**Phản hồi:**
```json
{
  "ready": true,
  "model_source": "artifacts/rfdetr_medium (rfdetr)",
  "cooccurrence_source": "cooccurrence_matrix.json (45 class, 1717 ảnh train)",
  "room_scores_source": "artifact category_room_scores.json (62 nhóm × 7 phòng) + mwg.categories",
  "catalog_source": "PostgreSQL mwg.product (1413 sản phẩm)"
}
```

---

## 💻 Chạy Local (Không qua Docker)

```bash
# 1. Điều hướng vào thư mục api
cd api

# 2. Cài đặt các thư viện phụ thuộc
pip install -r requirements.txt

# 3. Thiết lập file cấu hình môi trường
cp .env.example .env

# 4. Chạy server qua uvicorn
uvicorn app.main:get_app --factory --host 0.0.0.0 --port 8000 --reload
```

---

## 🧪 Kiểm thử (Testing)

Suite kiểm thử bao quát toàn bộ logic xếp hạng, tính toàn vẹn artifact, schema validation và API endpoints:

```bash
# Chạy toàn bộ test
pytest tests -v

# Chạy kiểm tra tài liệu OpenAPI
pytest tests/test_docs.py -v
```
