"""Room spatial placement and photometric render service.
Ported from ROOM_PLACEMENT_V8_20_DYNAMIC_SCENE_BRIGHTNESS_MINUS_20.ipynb.
"""
from __future__ import annotations

import base64
import io
import json
import logging
import math
import mimetypes
import os
import random
import re
import sys
import time
import types
import unicodedata
import uuid
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from urllib.parse import urljoin, urlparse

import cv2
import numpy as np
import requests
import torch
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import (
    binary_closing,
    distance_transform_edt,
    label as connected_components,
    median_filter,
)
from transformers import (
    AutoImageProcessor,
    AutoModelForDepthEstimation,
    Mask2FormerForUniversalSegmentation,
)

logger = logging.getLogger("app.placement")

def display(*args, **kwargs):
    pass

class _DummyDF:
    def __init__(self, *args, **kwargs):
        pass
    def to_csv(self, *args, **kwargs):
        pass
    def __len__(self):
        return 0
    def __getitem__(self, item):
        return self
    def __setitem__(self, key, value):
        pass
    def map(self, *args, **kwargs):
        return self
    def sort_values(self, *args, **kwargs):
        return self
    def reset_index(self, *args, **kwargs):
        return self

class _DummyPD:
    DataFrame = _DummyDF
    Series = _DummyDF

pd = _DummyPD()

# Constants & Placement Models
ALLOWED_SURFACES = ["wall", "floor", "ceiling", "table"]
SEGMENTATION_MODEL_ID = "facebook/mask2former-swin-large-ade-semantic"
DEPTH_MODEL_ID = "Intel/dpt-large"
_SEG_CACHE = {}
_DEPTH_CACHE = {}

SEGMENTATION_SURFACE_LABELS = {
    "floor": "floor", "flooring": "floor",
    "wall": "wall", "ceiling": "ceiling",
    "table": "table", "desk": "table",
    "countertop": "table", "counter": "table",
}

SEGMENTATION_OBSTACLE_LABELS = {
    "window": "window", "windowpane": "window",
    "door": "door", "screen door": "door",
    "stairway": "stairs", "stairs": "stairs", "staircase": "stairs",
    "fireplace": "fireplace",
    "lamp": "light_fixture", "light": "light_fixture",
    "chandelier": "light_fixture",
}

API_CATEGORY_PRODUCTS = {}
PRODUCT_IMAGE_DOMAINS = ("tgdd.vn", "dienmayxanh.com", "mwgshop.vn")
MAX_PRODUCT_IMAGE_BYTES = 15 * 1024 * 1024
MAX_API_PNG_BYTES = 25 * 1024 * 1024
API_TIMEOUT = 45.0
REMOVE_BG_PRODUCT_TYPE_LOCATION = "form"
REMOVE_BG_BASE_URL = os.environ.get("REMOVE_BG_BASE_URL", "").strip()
REMOVE_BG_API_URL = os.environ.get("REMOVE_BG_API_URL", "").strip()
if not REMOVE_BG_API_URL and REMOVE_BG_BASE_URL:
    _b = REMOVE_BG_BASE_URL.rstrip("/")
    REMOVE_BG_API_URL = _b if "/api/v1/products/remove-background" in _b else f"{_b}/api/v1/products/remove-background"
OUTPUT_BASE = Path("/tmp/outputs")
IMAGE_INPUT = None
RECOMMEND_RANDOM_SEED = None

PLACEMENT_GROUP_ALIASES = {
    "ghe massage": "Ghế massage",
    "tivi": "TV", "tv": "TV", "tivi, tv": "TV",
    "may dieu hoa": "Máy điều hòa", "dieu hoa": "Máy điều hòa",
    "quat": "Quạt điện", "quat dien": "Quạt điện",
    "quat tran": "Quạt trần",
    "loa karaoke": "Loa karaoke",
    "may loc khong khi": "Máy lọc không khí",
    "robot hut bui": "Robot hút bụi",
    "tham trai san": "Thảm trải sàn",
    "tivi box & dau thu ky thuat so": "Tivi box & Đầu thu kỹ thuật số",
}

API_PRODUCT_TYPE_OVERRIDES = {}


# ===== CELL 7 =====
# Chỉ các cấu trúc phục vụ placement; không có catalog local hay recommendation algorithm.
import unicodedata
API_CATEGORY_PRODUCTS = {}   # chỉ được lấp từ response API live
GROUP_TO_DETECTOR = {}        # không có nhãn bbox tự nhận diện trong notebook

# ===== CELL 13 =====
# C.3 — Parse response shape được người dùng cung cấp.
# group_name và product_id lấy nguyên từ API. Placement group có alias riêng.
import random, unicodedata

def _fold_vi(value):
    s=unicodedata.normalize('NFD',str(value or '').casefold())
    return ''.join(c for c in s if unicodedata.category(c)!='Mn').replace('đ','d').strip()

PLACEMENT_GROUP_ALIASES={
    'ghe massage':'Ghế massage',
    'tivi':'TV','tv':'TV','tivi, tv':'TV',
    'may dieu hoa':'Máy điều hòa','dieu hoa':'Máy điều hòa',
    'quat':'Quạt điện','quat dien':'Quạt điện',
    'quat tran':'Quạt trần',
    'loa karaoke':'Loa karaoke',
    'may loc khong khi':'Máy lọc không khí',
    'robot hut bui':'Robot hút bụi',
    'tham trai san':'Thảm trải sàn',
    'tivi box & dau thu ky thuat so':'Tivi box & Đầu thu kỹ thuật số',
}

def canonical_placement_group(group_name, product=None):
    group=_fold_vi(group_name)
    name=_fold_vi((product or {}).get('product_name',''))
    if group in ('quat','quat dien') and ('quat tran' in name or 'ceiling fan' in name):
        return 'Quạt trần'
    return PLACEMENT_GROUP_ALIASES.get(group,group_name)

def _external_recommendation_root(payload):
    if not isinstance(payload,dict):
        raise ValueError('Recommend API phải trả JSON object')
    if 'recommendations' in payload: return payload
    for name in ('data','result','payload'):
        x=payload.get(name)
        if isinstance(x,dict) and 'recommendations' in x:
            return x
    raise ValueError('JSON API thiếu mảng recommendations')

def parse_and_random_select_api_response(payload,top_k=None,seed=None):
    root=_external_recommendation_root(payload)
    groups=root['recommendations']
    if not isinstance(groups,list):
        raise ValueError('recommendations phải là array')
    rnd=random.Random(seed)
    selected=[];seen=set()
    for group in groups:
        if not isinstance(group,dict): continue
        group_name=str(group.get('group_name') or '').strip()
        group_code=str(group.get('category_id') or group.get('group_code') or '').strip()
        key=group_code or _fold_vi(group_name)
        if not key or not group_name or key in seen:continue
        seen.add(key)
        products=group.get('products') or []
        if not isinstance(products,list): continue
        candidates=[p for p in products if isinstance(p,dict)]
        if not candidates: continue
        # Uniform random trong đúng mảng products của category; chỉ chọn một lần.
        prod=dict(rnd.choice(candidates))
        for k in ('product_id','product_code','product_name','image_url','product_url','category_id'):
            prod[k]=str(prod.get(k) or '').strip()
        prod['category_id']=prod['category_id'] or group_code
        try:prod['price']=float(prod.get('price') or 0)
        except (ValueError,TypeError):prod['price']=0.0
        prod['currency']=str(prod.get('currency') or 'VND')
        prod['subcategory']=str(prod.get('subcategory') or prod.get('product_type') or '').strip()
        canonical=canonical_placement_group(group_name,prod)
        item={k:v for k,v in group.items() if k!='products'}
        item.update(group_name=group_name,group_code=group_code,category_id=group_code,
                    placement_group=canonical,products=candidates,
                    selected_product=prod,selection_method='uniform_random_from_api_products',
                    selected_product_code=prod['product_code'],
                    sku_selection='random_from_external_api')
        selected.append(item)
        if top_k is not None and len(selected)>=int(top_k): break
    detected=root.get('detected_objects',[])
    if not isinstance(detected,list):detected=[]
    return {'room_type':root.get('room_type'),
            'detected_objects':detected,
            'recommendations':selected,
            'recommendation_source':'external_recommend_api',
            'selection_seed':seed,
            'raw_recommendation_response':payload}

# ===== CELL 15 =====
# C.4 — JSON mẫu KHÔNG có subcategory nên không được lấy product_type từ CSV cũ.
# Đây là các tên CHỜ XÁC NHẬN với API remove-background; nếu backend dùng enum
# khác thì cập nhật dictionary này. SKU không có mapping sẽ BỊ BỎ QUA khi render.
REMOVE_BG_PRODUCT_TYPE_BY_CATEGORY_ID={
    '2662':'massage_chair',  # Ghế Massage
    '1844':'pillow',         # Gối, nệm
    '941':'rug',             # Thảm trải sàn (chỉ hợp lý cho thảm thực sự)
    '59':'fan',              # Quạt; đổi theo subtype nếu backend yêu cầu
    '980':'tv_box',          # Tivi box & Đầu thu kỹ thuật số
}
REMOVE_BG_PRODUCT_TYPE_BY_GROUP={
    'TV':'tv','Máy điều hòa':'air_conditioner',
    'Ghế massage':'massage_chair','Quạt trần':'ceiling_fan',
    'Quạt điện':'fan','Loa':'speaker','Loa karaoke':'karaoke_speaker',
    'Máy lọc không khí':'air_purifier','Robot hút bụi':'robot_vacuum',
    'Quạt điều hòa':'air_cooler','Thảm trải sàn':'rug',
}

def resolve_remove_bg_product_type(reco,product):
    # Priority: explicitly provided by API -> category override -> known group mapping
    explicit=str(product.get('product_type') or product.get('subcategory') or '').strip()
    if explicit:return explicit
    code=str(reco.get('category_id') or product.get('category_id') or '')
    return str(REMOVE_BG_PRODUCT_TYPE_BY_CATEGORY_ID.get(code)
               or REMOVE_BG_PRODUCT_TYPE_BY_GROUP.get(reco.get('placement_group',''))
               or '').strip()

# ===== CELL 17 =====
# B.4.1 — Danh mục ID TÍCH HỢP từ categories(1).json (434 bản ghi).
# Có ID trùng giữa các catalog; KHÔNG bao giờ map chỉ bằng category_id.
# Bộ dữ liệu này đã được đóng gói vào notebook: KHÔNG cần tải thêm JSON lên Colab.
CATEGORY_REGISTRY_SOURCE = 'categories(1).json'
CATEGORY_REGISTRY_RAW = """1427|Bút cảm ứng
1421|Ống kính điện thoại
137|Thẻ nhớ
1437|Giá đỡ
1429|Vỏ bao, Ốp lưng máy tính bảng
2532|Bình thủy điện
2531|Bình đun siêu tốc
1442|Hub, cáp kết nối
2536|AirTag, Vỏ bảo vệ AirTag
127|Sạc dự phòng
2556|Gậy chụp ảnh, Gimbal
2671|Ghế gaming, ghế công thái học
1085|Thiết bị định vị GPS
2535|Sạc, cáp
2554|Adapter sạc
693|Khác
1422|Miếng dán màn hình
682|Vật tư y tế khác
2116|Máy tính bảng
913|Cân sức khỏe và phân tích cơ thể
1086|Đồng hồ thông minh & Vòng đeo tay sức khỏe
2643|Kim tiêm y tế
1018|Hỗ trợ làm đẹp
914|Bao tay và khẩu trang y tế
1541|Nhiệt kế
1019|Hỗ trợ kiểm soát cân nặng
1543|Máy đo lượng đường trong máu
2644|Máy xông khí dung
1544|Máy đo huyết áp
1574|Bao cao su, gel
2642|Dầu, tinh dầu
1669|Thuốc sát trùng & khử trùng y tế
1972|Que thử thai
1672|Băng gạc y tế
1975|Cốc nguyệt san
1974|Dung dịch vệ sinh phụ nữ
2027|Nước súc miệng
2028|Kem đánh răng
2029|Hỗ trợ trắng răng
1976|Băng vệ sinh có cánh & không cánh
2031|Tăm, chỉ nha khoa
2033|Bàn chải đánh răng
2032|Bàn chải điện & phụ kiện
300|Điện thoại
2035|Chống muỗi & xua đuổi côn trùng
2036|Chăm sóc tai
2558|Bồn ngâm chân
2034|Dụng cụ massage và trị liệu
941|Thảm trải sàn
1124|Móc treo
2647|Vệ sinh mắt, mũi, tai
1128|Hộp khăn giấy
1129|Hộp đựng, giỏ đựng đồ
1017|Hỗ trợ sức khỏe
2648|Vớ, đai y khoa
124|Thớt
1678|Hộp đựng thực phẩm
1430|Vỏ bao, Ốp lưng điện thoại
2544|Muỗng, nĩa
231|Nước xả
1699|Miếng bọt biển, miếng chà vệ sinh
227|Nước giặt
1709|Bộ & Cây lau nhà
1710|Bàn chải vệ sinh
1686|Dụng cụ đo lường
116|Nồi
1732|Máy khoan, máy vặn vít & phụ kiện
1735|Kìm
1740|Cưa, máy cắt & máy mài
1701|Khăn giấy, giấy ướt
117|Chảo
1687|Dao & kéo
2650|Tua vít
2663|Cọ quét sơn
1945|Máy khuếch tán, tạo ẩm & xông tinh dầu
2540|Ổn áp
1844|Gối, nệm
2541|Pin tiểu
2121|Khóa, ổ khóa
2113|Bình nước
1088|Xà phòng & sữa tắm
1091|Khử mùi cơ thể
1436|Bông tẩy trang
1093|Kem & sữa dưỡng thể
979|Tẩy trang
786|Máy tỉa lông mũi, lông tai
1692|Dụng cụ tẩy lông
1957|Sản phẩm dưỡng tóc
1712|Dụng cụ rửa mặt
359|Máy tạo kiểu tóc
1958|Dầu xả
2557|Tông đơ cắt tóc
1959|Dầu gội
2004|Tẩy tế bào chết
355|Máy cạo râu
75|Máy sấy tóc
2006|Nước cân bằng da
2007|Mặt nạ
1990|Sữa rửa mặt
2005|Hỗ trợ trị mụn, ngừa sẹo, mờ thâm
980|Tivi box & Đầu thu kỹ thuật số
981|Giá treo tivi
2009|Kem dưỡng ẩm
2539|Điều khiển tivi
2534|Bình, ly giữ nhiệt
67|Máy lọc không khí
81|Máy hút ẩm
357|Máy tăm nước
2652|Siêu sắc thuốc
2528|Bàn ủi
38|Tivi
42|Máy sấy quần áo
41|Máy giặt
1080|Thiết bị sấy khô nệm & Giày
44|Máy nước nóng
1083|Máy sưởi
1078|Máy hút bụi & Thiết bị làm sạch sàn
1498|Máy chiếu & Màn hình chiếu
2522|Máy rửa chén
1512|Chuông cửa
1082|Máy lạnh
1510|Ổ cắm điện & Dây nối
73|Lò nướng
58|Lò vi sóng
59|Quạt
76|Nồi áp suất
1799|Tủ đông, tủ mát
39|Tủ lạnh
312|Máy lọc nước & Phụ kiện lọc nước
1813|Máy pha cà phê & Phụ kiện
1810|Nồi chiên không dầu
1817|Máy chế biến thực phẩm & Xay thịt
1808|Nồi cơm điện
2525|Bếp hồng ngoại
2526|Máy làm sữa hạt
1816|Máy ép, xay sinh tố
2527|Bếp ga
2524|Bếp từ
2530|Lẩu điện
2529|Máy hút mùi
2656|Máy làm sữa chua
2657|Máy làm đá
279|Kem
713|Thực phẩm khác
165|Sữa tươi
1110|Phô mai & bột phô mai
1109|Sữa chua
2551|Mì - Nui - Miến khô
1470|Dầu ăn
1476|Mật ong và siro
1618|Nước chấm
2547|Gia vị rắc cơm
2543|Sữa cho người lớn
2549|Bột nêm
1753|Sữa thực vật
1756|Nước tinh khiết
1752|Thức uống Sô cô la
1901|Đồ ăn chế biến sẵn
2552|Nước trái cây
1911|Chà bông
2646|Nước yến, yến sào
1916|Các loại rong biển ăn liền
1915|Kẹo
1642|Scooters & Xe Đạp 1 Bánh
1902|Cơm và cháo ăn liền
1807|Nồi nấu chậm & Dụng cụ nấu chân không
2553|Xe lắc
2649|Xe đạp tập thể dục tại nhà
2662|Ghế Massage
2550|Snack - Bánh kẹo các loại
142|Ổ Cứng Di Động
1651|Xe Đạp
1071|USB & OTG
1357|Bàn Di Chuột
2669|Túi đựng phụ kiện
1481|Bộ Máy Tính Bàn
2066|Đèn & Đèn Pin
1486|Mực In & Khay Mực
1487|Máy In, Máy Scan & Máy Photo
1763|Màn hình máy tính
1940|Bảng Vẽ Điện Tử
2538|Balo, túi chống sốc
138|Chuột Máy Tính
1097|Bao tay trẻ em & Tất
2537|Bàn phím
1158|Bộ đồ liền thân
1135|Váy
931|Balo, túi đeo cho bé
951|Đồng hồ trẻ em
1282|Vali kéo
1772|Laptop
1102|Sữa công thức
1297|Mắt kính
1296|Mũ & mũ lưỡi trai
1157|Bộ quần áo
1607|Dép quai hậu
1099|Tã dùng một lần
1103|Đồ ăn nhẹ cho bé
1104|Cháo, Thực phẩm xay nhuyễn & Ngũ cốc
228|Nước hoa cho bé
2545|Quần áo sơ sinh
1824|Dụng cụ tắm & Phụ kiện
297|Xe tập đi
1840|Nôi & Cũi & Giường cho bé
1846|Chăn & Quấn em bé
2654|Máy tạo tiếng ồn trắng
1822|Khăn lau
1820|Sản phẩm tắm & gội cho bé
182|Địu em bé
187|Ghế ăn dặm
1715|Xe đẩy
195|Máy hâm sữa
1850|Ghế ngồi ô tô & xe máy
199|Miếng lót thấm sữa
213|Yếm
197|Máy tiệt trùng bình sữa
1854|Túi trữ sữa
1855|Máy hút sữa & phụ kiện
1852|Ti giả
1859|Đồ dùng ăn dặm
1863|Xe đạp, Xe chòi chân & Xe điện
1861|Dụng cụ vệ sinh bình sữa
166|Sữa bầu
1965|Kem dưỡng ẩm cho mẹ
1966|Vitamin & Thực phẩm bổ sung
1967|Chống nắng cho bé
1862|Bình sữa & Phụ kiện
1969|Chăm sóc mũi cho bé
1968|Chăm sóc răng miệng cho bé
2019|Sữa & Kem dưỡng da
2645|Miếng dán, dụng cụ giảm đau, hạ sốt
2020|Phấn cho bé
1623|Mũ bảo hiểm
1215|Quạt Cầm Tay
1502|Máy ảnh kỹ thuật số
1341|Camera hành trình & Camera lùi
330|Dây đồng hồ
1545|Micro thu âm
2555|Micro karaoke
2056|Camera giám sát kết nối internet
1870|Đồ chơi xếp hình
1933|Loa
1536|Tai nghe nhét tai & chụp tai
320|Chuông cửa camera
82|Miếng phủ bàn phím
323|Phụ kiện chơi game
140|Thiết bị định vị, phụ kiện
83|Ống kính tele
148|Ổ cắm, bóng đèn thông minh
279|Hộp đựng đồ đa năng
271|Máy pha cà phê
306|Máy ảnh
307|Phụ kiện máy ảnh
303|Tủ chống ẩm
304|Bồn nước
101|Máy xay thịt
60|Túi đựng phụ kiện
47|Dịch vụ vệ sinh, bảo dưỡng
126|Phụ kiện ốp lưng
99|Nồi áp suất
156|Khóa điện tử
115|Máy giặt
285|Dụng cụ mài dao
39|Máy rửa chén
283|Thớt
177|Máy rửa mặt
40|Tủ đông, tủ mát
166|Robot hút bụi
184|Bồn ngâm chân
183|Máy tăm nước
49|Đồng hồ thông minh
171|Thiết bị sưởi ấm khác
261|Máy tiệt trùng
313|Đồ dùng cho bé
238|Kìm
241|Dụng cụ cân, đo
168|Bếp hồng ngoại
167|Bếp từ
232|Ổ Khoá
297|Đèn thông minh
97|Máy xay sinh tố
215|Máy vắt cam
256|Khung treo Tivi
182|Bàn chải điện
317|Bóng đèn led
199|Quạt điều hoà
198|Quạt
69|Quạt mini
200|Quạt trần
28|Laptop
117|Tủ sấy quần áo
164|Phụ kiện lọc khí
98|Máy ép trái cây
162|Máy lọc không khí
187|Máy tạo ẩm
163|Máy hút ẩm
194|Ghế massage
275|Tô, chén, dĩa
149|Pin tiểu
120|Cáp sạc
157|Máy lọc nước
158|Lõi máy lọc nước
160|Cốc lọc nước đầu nguồn
161|Lọc nước không điện
321|Lọc tổng đầu nguồn
141|Giá đỡ điện thoại, máy tính bảng, laptop
319|Máy đọc sách
208|Bình, ly giữ nhiệt
203|Vá, sạn
68|Mắt kính
58|Phụ kiện tablet
79|Bộ lưu điện
178|Máy massage cá nhân
137|Micro thu âm điện thoại
64|Thẻ nhớ
135|Bàn phím
30|Máy tính bảng
318|Bảng vẽ tự xoá
151|Máy tạo tiếng ồn trắng
147|Máy chiếu
326|Cảm biến thông minh
65|USB
63|Ổ cứng di động
322|Bảng treo đồ Pegboard
123|Ốp lưng đ.thoại
96|Cây nước nóng lạnh
102|Lẩu điện
100|Nồi nấu chậm
274|Máy xay cà phê
276|Hộp đựng thực phẩm
282|Dao
284|Kéo
127|Camera giám sát
129|Camera hành trình / hành động
128|Gậy chụp ảnh, Gimbal
130|Flycam
331|Đầu ghi camera
212|Bình đựng nước
134|Chuột máy tính
136|Miếng lót chuột
142|Túi đựng AirPods
119|Adapter sạc
121|Hub, cáp chuyển đổi
329|Ghế chỉnh dáng ngồi
268|Máy làm sữa chua
267|Máy làm đá
333|Ghế gaming, ghế công thái học
44|Vali kéo
122|Ốp lưng tablet
330|Robot thông minh
312|Đồ dùng khác
180|Mặt nạ chăm sóc da
207|Bộ lau nhà
205|Nồi
206|Chảo
280|Thau, rổ, gáo nước
211|Mũ bảo hiểm
210|Muỗng, đũa
213|Khay đá - ly làm đá
216|Dụng cụ vắt cam
223|Bàn chải - Cọ toilet
222|Miếng rửa chén, cọ chùi
311|Thùng rác
293|Túi đựng rác
278|Hộp đựng giấy
186|Máy xông hơi mặt
193|Xe đạp tập thể dục tại nhà
179|Máy triệt lông
185|Cân sức khoẻ
226|Bộ dụng cụ đa năng
104|Bình đun siêu tốc
236|Mũi khoan
255|Giá đỡ máy giặt, máy lọc nước, chân loa
174|Máy cạo râu
257|Điều khiển Tivi
251|Vợt muỗi
253|Ổ cắm - Phích cắm
224|Máy khoan, vặn vít
250|Cọ quét sơn
244|Máy cưa, máy cắt
172|Máy sấy tóc
269|Máy làm sữa hạt
170|Quạt sưởi
169|Đèn sưởi
86|Nồi cơm điện
324|Phụ kiện hút bụi
165|Máy hút bụi
41|Máy nước nóng
72|Máy tính để bàn
57|Dây đồng hồ
80|Bảng vẽ điện tử
77|Giá treo màn hình
139|Micro karaoke
51|Sạc dự phòng
75|Máy in
221|Máy cắt lông xù quần áo
103|Lò vi sóng
125|Miếng dán camera
124|Miếng dán
116|Máy sấy quần áo
218|Khay - kệ đựng đồ
93|Bàn ủi
114|Siêu sắc thuốc
106|Máy hút mùi
59|Balo, túi chống sốc
105|Bình thủy điện
46|Xe đạp
132|Loa
309|Phụ kiện vệ sinh điện tử
91|Bếp ga
133|Tai nghe
220|Móc kẹp quần áo
201|Lò nướng
176|Máy tỉa lông mũi
181|Máy tạo kiểu tóc
175|Tông đơ cắt tóc
152|Đèn điện, đèn sạc
202|Máy nhồi bột, đánh trứng
316|Tủ trữ rượu vang
301|Đèn năng lượng mặt trời
38|Tủ lạnh
87|Nồi chiên không dầu
73|Màn hình máy tính
36|Máy lạnh
118|Máy sấy giày
37|Tivi
302|Thiết bị mã vạch
71|Máy chơi game
76|Mực in
70|Phụ kiện ô tô
66|Thiết bị mạng
74|Phần mềm
27|Điện thoại
50|Đồng hồ thời trang"""
CATEGORY_REGISTRY_V815 = [
    {'id': line.split('|', 1)[0], 'name': line.split('|', 1)[1]}
    for line in CATEGORY_REGISTRY_RAW.splitlines() if '|' in line
]
CATEGORY_NAMES_BY_ID_V815 = defaultdict(list)
for _entry in CATEGORY_REGISTRY_V815:
    CATEGORY_NAMES_BY_ID_V815[_entry['id']].append(_entry['name'])
CATEGORY_AMBIGUOUS_IDS_V815 = {
    k:v for k,v in CATEGORY_NAMES_BY_ID_V815.items() if len(set(v))>1
}
print('Category registry:', len(CATEGORY_REGISTRY_V815), 'entries;',
      len(CATEGORY_NAMES_BY_ID_V815), 'unique IDs;',
      len(CATEGORY_AMBIGUOUS_IDS_V815),'ambiguous IDs')

# ===== CELL 19 =====
# B.4.2 — Resolver ID + group_name + tên SKU => Placement Group + product_type.
# Phải có category_id VÀ tên nhóm đồng nhất; nếu không, FAIL CLOSED.
# Không dùng mapping kiểu {'38':'TV'} vì ID 38 còn là Tủ lạnh.
CATEGORY_NAME_ALIASES_V815 = {
    'tv':'tivi', 'television':'tivi',
    'air conditioner':'may lanh','may dieu hoa':'may lanh','dieu hoa':'may lanh',
    'air purifier':'may loc khong khi',
    'quat dien':'quat', 'fan':'quat',
    'massage chair':'ghe massage',
    'karaoke speaker':'loa', 'loa karaoke':'loa',
}
CATEGORY_TO_PLACEMENT_V815 = {
    'tivi':'TV', 'may lanh':'Máy điều hòa',
    'may loc khong khi':'Máy lọc không khí',
    'ghe massage':'Ghế massage',
    'loa':'Loa',
    'quat':'Quạt điện',
    'quat tran':'Quạt trần',
    'quat dieu hoa':'Quạt điều hòa',
    'quat mini':'Quạt điện',
    'robot hut bui':'Robot hút bụi',
    'may hut bui':'Máy hút bụi',
    'may hut bui & thiet bi lam sach san':'Máy hút bụi',
    'tham trai san':'Thảm trải sàn',
    'tivi box & dau thu ky thuat so':'Tivi box & Đầu thu kỹ thuật số',
    'may chieu & man hinh chieu':'Máy chiếu',
    'may chieu':'Máy chiếu',
    'may choi game':'Máy chơi game',
    'bon ngam chan':'Bồn ngâm chân',
}
# product_type là chuỗi phía API xóa nền; chỉnh tại đây nếu backend dùng enum khác.
# Đây KHÔNG phải field có sẵn trong categories.json (file đó chỉ có id/name).
REMOVE_BG_TYPE_BY_PLACEMENT_V815 = {
    'TV':'tv', 'Máy điều hòa':'air_conditioner',
    'Máy lọc không khí':'air_purifier',
    'Ghế massage':'massage_chair',
    'Loa':'speaker','Loa karaoke':'karaoke_speaker',
    'Quạt điện':'fan','Quạt trần':'ceiling_fan',
    'Quạt điều hòa':'air_cooler','Robot hút bụi':'robot_vacuum',
    'Máy hút bụi':'vacuum_cleaner',
    'Thảm trải sàn':'rug','Tivi box & Đầu thu kỹ thuật số':'tv_box',
    'Máy chiếu':'projector','Máy chơi game':'game_console',
    'Bồn ngâm chân':'foot_bath',
}

def v815_normalize_category_name(text):
    return CATEGORY_NAME_ALIASES_V815.get(_fold_vi(text),_fold_vi(text))

def v815_resolve_category(category_id,group_name,product=None):
    raw_id=str(category_id or '').strip()
    raw_name=str(group_name or '').strip()
    names=CATEGORY_NAMES_BY_ID_V815.get(raw_id,[])
    if not names:
        return {'status':'unknown_id','category_id':raw_id,'api_group_name':raw_name,
                'reason':'category_id không tồn tại trong categories(1).json',
                'renderable':False}
    unique={v815_normalize_category_name(x):x for x in names}
    target=v815_normalize_category_name(raw_name)
    if target in unique:
        official=unique[target]
        resolution='id_and_group_name'
    elif not raw_name and len(unique)==1:
        official=next(iter(unique.values()))
        resolution='unique_id_only'
    elif len(names) > 0:
        official=names[0]
        resolution='id_fallback'
    else:
        return {'status':'ambiguous_or_mismatched_name', 'category_id':raw_id,
                'api_group_name':raw_name,'candidate_names':list(names),
                'renderable':False,
                'reason':'Tên nhóm API không khớp danh mục cho ID này; không đoán từ ID.'}
    norm=_fold_vi(official)
    group=CATEGORY_TO_PLACEMENT_V815.get(norm)
    if not group:
        for rule_cat in CATEGORY_RULES:
            if _fold_vi(rule_cat) == norm or norm in _fold_vi(rule_cat) or _fold_vi(rule_cat) in norm:
                group = rule_cat
                break
        if not group:
            group = PLACEMENT_GROUP_ALIASES.get(norm)
    title=_fold_vi((product or {}).get('product_name',''))
    # Danh mục rộng: subtype SKU là bắt buộc để tránh placement sai.
    if norm=='loa' and ('karaoke' in title or 'keo' in title):
        group='Loa karaoke'
    if norm in ('quat','quat mini'):
        if 'quat tran' in title or 'ceiling fan' in title: group='Quạt trần'
        elif 'quat dieu hoa' in title or 'quat hoi nuoc' in title:group='Quạt điều hòa'
    if norm=='may hut bui & thiet bi lam sach san':
        if 'robot' in title:group='Robot hút bụi'
        else:group='Máy hút bụi'
    if norm=='may chieu & man hinh chieu' and 'may chieu' not in title:
        group=None  # màn chiếu không có surface như máy chiếu
    if norm=='quat mini':
        if not any(k in title for k in ('quat mini','quat ban','quat de ban')): group=None
    product_type=REMOVE_BG_TYPE_BY_PLACEMENT_V815.get(group, 'auto')
    status='matched' if group and product_type else 'unsupported_placement'
    return {'status':status,'category_id':raw_id,'api_group_name':raw_name,
        'official_category_name':official,'match_method':resolution,
        'id_has_collision':len(set(names))>1,
        'placement_group':group,'product_type':product_type,
        'renderable':status=='matched',
        'reason':'' if status=='matched' else 'Chưa có rule Placement an toàn cho category/SKU này.'}

# Wrapper: giữ ngẫu nhiên đúng 1 SKU mỗi nhóm từ API, bổ sung mapping và audit.
_v814_parse_random_api = parse_and_random_select_api_response
def parse_and_random_select_api_response(payload,top_k=None,seed=None):
    parsed=_v814_parse_random_api(payload,top_k=top_k,seed=seed)
    for item in parsed['recommendations']:
        product=item['selected_product']
        product_cid=str(product.get('category_id') or '').strip()
        group_cid=str(item.get('category_id') or '').strip()
        if product_cid and product_cid != group_cid:
            category={'status':'product_category_mismatch','renderable':False,
                      'category_id':group_cid,'api_group_name':item.get('group_name'),
                      'reason':'SKU có category_id khác category_id của nhóm API'}
        else:
            category=v815_resolve_category(group_cid,item['group_name'],product)
        item['category_mapping']=category
        if category.get('renderable'):
            item['placement_group']=category['placement_group']
        # Không suy đoán category mới từ tên; stage2_render sẽ bỏ qua nhóm chưa map.
    return parsed

# Wrapper: product_type ưu tiên từ response API nếu có, sau đó theo mapping đã xác minh.
_v814_resolve_remove_bg = resolve_remove_bg_product_type
def resolve_remove_bg_product_type(reco,product):
    mapping=reco.get('category_mapping') or {}
    if not mapping.get('renderable'):
        return ''
    explicit=str(product.get('product_type') or product.get('subcategory') or '').strip()
    inferred=explicit or str(REMOVE_BG_TYPE_BY_PLACEMENT_V815.get(
        mapping.get('placement_group'),'')).strip()
    overrides=globals().get('API_PRODUCT_TYPE_OVERRIDES',{})
    if not isinstance(overrides, dict):
        raise TypeError('API_PRODUCT_TYPE_OVERRIDES phải là dict')
    # ID riêng lẻ có thể trùng trong categories.json; ưu tiên ID + tên nhóm.
    keys=(f"{reco.get('category_id','')}:{reco.get('group_name','')}",
          str(reco.get('category_id','')), mapping.get('placement_group',''), inferred)
    for key in keys:
        if key in overrides:
            value=str(overrides[key]).strip()
            if not value:
                raise ValueError(f'product_type override rỗng cho {key!r}')
            return value
    return inferred

print('B.4.2: chỉ chọn category khớp ID+name và có rule đặt vật thể.')

# ===== CELL 20 =====
def load_image(source):
    """Nạp ảnh local/URL/PIL/numpy. Không phụ thuộc DATA_ROOT hay COCO."""
    if isinstance(source,Image.Image):
        return source.convert('RGB')
    if isinstance(source,np.ndarray):
        return Image.fromarray(source).convert('RGB')
    name=str(source).strip()
    if name.startswith(('https://','http://')):
        try:
            r=requests.get(name,timeout=(10,40),headers={'User-Agent':'Mozilla/5.0'})
            r.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError('Không tải được ảnh đầu vào: '+str(exc)) from exc
        return Image.open(BytesIO(r.content)).convert('RGB')
    path=Path(name).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f'Không tìm thấy ảnh: {path}. Sửa IMAGE_INPUT ở Cell A.2.')
    return Image.open(path).convert('RGB')

# ===== CELL 23 =====
# Dependencies đã được cài tại Cell A.1; không tự cài trong lúc thực thi pipeline.
import importlib.util
missing=[name for name in ('transformers','scipy','torch','cv2')
         if importlib.util.find_spec(name) is None]
if missing:
    print('Thiếu thư viện:',missing,'— chạy lại Cell A.1 để cài.')

# ===== CELL 25 =====
# ===== Cell 10c.2: 62-category catalog placement profiles + reference ratios =====
# Static 62 placement rules preserved from V8.12; catalog is supplied dynamically by external recommendation API.
# Surfaces and relative size factors below are DESIGN HEURISTICS, not observed SKU dimensions.
from copy import deepcopy
from pathlib import Path
import json
# import pandas as pd (replaced with dummy pd)

# ===== CELL 27 =====
# Groups appearing in the saved catalog run. The 62 rules are explicit to permit review.
SNAPSHOT_CATALOG_GROUPS = ['TV',
 'Loa karaoke',
 'Loa',
 'Quạt trần',
 'Máy điều hòa',
 'Máy lọc không khí',
 'Máy hút bụi',
 'Robot hút bụi',
 'Bóng đèn LED',
 'Ghế massage',
 'Microphone',
 'Máy chiếu',
 'Máy chơi game',
 'Camera trong nhà',
 'Quạt điều hòa',
 'Quạt điện',
 'Máy hút ẩm',
 'Bộ lau nhà',
 'Đèn điện/đèn sạc',
 'Ổ cắm điện',
 'Thùng rác',
 'Khay/kệ đựng đồ',
 'Màn hình máy tính',
 'Laptop',
 'Quạt sưởi',
 'Máy pha cà phê',
 'Cây nước nóng lạnh',
 'Hộp đựng đồ',
 'Thiết bị massage',
 'Máy tính để bàn',
 'Camera ngoài trời',
 'Tủ lạnh',
 'Tủ mát',
 'Ấm đun nước',
 'Máy lọc nước',
 'Đèn năng lượng mặt trời',
 'Bồn ngâm chân',
 'Tủ đông',
 'Máy giặt',
 'Máy sấy quần áo',
 'Cân sức khỏe',
 'Máy nước nóng',
 'Máy rửa chén',
 'Nồi cơm điện',
 'Bếp từ',
 'Bếp hồng ngoại',
 'Bếp gas',
 'Máy hút mùi',
 'Lò vi sóng',
 'Lò nướng',
 'Nồi chiên không dầu',
 'Máy xay sinh tố',
 'Máy ép trái cây',
 'Máy làm sữa hạt',
 'Bàn ủi',
 'Máy sấy tóc',
 'Máy cạo râu',
 'Máy tỉa lông',
 'Máy tạo kiểu tóc',
 'Máy rửa mặt',
 'Bàn chải đánh răng điện',
 'Máy tăm nước']

# ===== CELL 29 =====
CATEGORY_RULES = {'TV': {'surface_options': ['wall', 'table'],
        'relative_width': 1.5,
        'relative_height': 0.78,
        'preferred_x': 'center',
        'preferred_y': 'middle',
        'anchor_labels': ['Sofa']},
 'Loa karaoke': {'surface_options': ['floor', 'table'],
                 'relative_width': 0.85,
                 'relative_height': 1.28,
                 'preferred_x': 'edge',
                 'preferred_y': 'bottom',
                 'anchor_labels': ['TV']},
 'Loa': {'surface_options': ['table', 'floor'],
         'relative_width': 0.55,
         'relative_height': 0.77,
         'preferred_x': 'edge',
         'preferred_y': 'middle',
         'anchor_labels': ['TV']},
 'Quạt trần': {'surface_options': ['ceiling'],
               'relative_width': 1.35,
               'relative_height': 0.35,
               'preferred_x': 'center',
               'preferred_y': 'middle',
               'anchor_labels': []},
 'Máy điều hòa': {'surface_options': ['wall'],
                  'relative_width': 1.35,
                  'relative_height': 0.62,
                  'preferred_x': 'edge',
                  'preferred_y': 'top',
                  'anchor_labels': []},
 'Máy lọc không khí': {'surface_options': ['floor'],
                       'relative_width': 0.76,
                       'relative_height': 1.13,
                       'preferred_x': 'edge',
                       'preferred_y': 'bottom',
                       'anchor_labels': ['Sofa']}}

# ===== CELL 31 =====
CATEGORY_RULES.update({'Máy hút bụi': {'surface_options': ['floor'],
                 'relative_width': 0.57,
                 'relative_height': 1.1,
                 'preferred_x': 'edge',
                 'preferred_y': 'bottom',
                 'anchor_labels': []},
 'Robot hút bụi': {'surface_options': ['floor'],
                   'relative_width': 0.56,
                   'relative_height': 0.23,
                   'preferred_x': 'edge',
                   'preferred_y': 'bottom',
                   'anchor_labels': []},
 'Bóng đèn LED': {'surface_options': ['ceiling', 'wall'],
                  'relative_width': 0.22,
                  'relative_height': 0.22,
                  'preferred_x': 'center',
                  'preferred_y': 'top',
                  'anchor_labels': []},
 'Ghế massage': {'surface_options': ['floor'],
                 'relative_width': 1.48,
                 'relative_height': 1.56,
                 'preferred_x': 'edge',
                 'preferred_y': 'bottom',
                 'anchor_labels': ['Chair', 'Sofa']},
 'Microphone': {'surface_options': ['table'],
                'relative_width': 0.22,
                'relative_height': 0.44,
                'preferred_x': 'edge',
                'preferred_y': 'middle',
                'anchor_labels': []},
 'Máy chiếu': {'surface_options': ['ceiling', 'table'],
               'relative_width': 0.95,
               'relative_height': 0.52,
               'preferred_x': 'center',
               'preferred_y': 'top',
               'anchor_labels': ['TV']}})

# ===== CELL 33 =====
CATEGORY_RULES.update({'Máy chơi game': {'surface_options': ['table'],
                   'relative_width': 0.55,
                   'relative_height': 0.45,
                   'preferred_x': 'edge',
                   'preferred_y': 'middle',
                   'anchor_labels': ['TV']},
 'Camera trong nhà': {'surface_options': ['wall', 'table', 'ceiling'],
                      'relative_width': 0.3,
                      'relative_height': 0.3,
                      'preferred_x': 'edge',
                      'preferred_y': 'top',
                      'anchor_labels': []},
 'Quạt điều hòa': {'surface_options': ['floor'],
                   'relative_width': 0.76,
                   'relative_height': 1.3,
                   'preferred_x': 'edge',
                   'preferred_y': 'bottom',
                   'anchor_labels': []},
 'Quạt điện': {'surface_options': ['floor', 'table'],
               'relative_width': 0.62,
               'relative_height': 1.32,
               'preferred_x': 'edge',
               'preferred_y': 'bottom',
               'anchor_labels': ['Sofa', 'Chair']},
 'Máy hút ẩm': {'surface_options': ['floor'],
                'relative_width': 0.73,
                'relative_height': 0.95,
                'preferred_x': 'edge',
                'preferred_y': 'bottom',
                'anchor_labels': []},
 'Bộ lau nhà': {'surface_options': ['floor'],
                'relative_width': 0.43,
                'relative_height': 1.1,
                'preferred_x': 'edge',
                'preferred_y': 'bottom',
                'anchor_labels': []}})

# ===== CELL 35 =====
CATEGORY_RULES.update({'Đèn điện/đèn sạc': {'surface_options': ['table', 'floor'],
                      'relative_width': 0.47,
                      'relative_height': 0.75,
                      'preferred_x': 'edge',
                      'preferred_y': 'middle',
                      'anchor_labels': []},
 'Ổ cắm điện': {'surface_options': ['wall'],
                'relative_width': 0.27,
                'relative_height': 0.24,
                'preferred_x': 'edge',
                'preferred_y': 'middle',
                'anchor_labels': []},
 'Thùng rác': {'surface_options': ['floor'],
               'relative_width': 0.52,
               'relative_height': 0.63,
               'preferred_x': 'edge',
               'preferred_y': 'bottom',
               'anchor_labels': []},
 'Khay/kệ đựng đồ': {'surface_options': ['table', 'floor'],
                     'relative_width': 0.79,
                     'relative_height': 0.5,
                     'preferred_x': 'edge',
                     'preferred_y': 'middle',
                     'anchor_labels': []},
 'Màn hình máy tính': {'surface_options': ['table'],
                       'relative_width': 1.0,
                       'relative_height': 0.72,
                       'preferred_x': 'center',
                       'preferred_y': 'middle',
                       'anchor_labels': ['Table', 'Desk']},
 'Laptop': {'surface_options': ['table'],
            'relative_width': 0.92,
            'relative_height': 0.35,
            'preferred_x': 'center',
            'preferred_y': 'middle',
            'anchor_labels': ['Table', 'Desk']}})

# ===== CELL 37 =====
CATEGORY_RULES.update({'Quạt sưởi': {'surface_options': ['floor', 'table'],
               'relative_width': 0.65,
               'relative_height': 1.0,
               'preferred_x': 'edge',
               'preferred_y': 'bottom',
               'anchor_labels': []},
 'Máy pha cà phê': {'surface_options': ['table'],
                    'relative_width': 0.65,
                    'relative_height': 0.76,
                    'preferred_x': 'edge',
                    'preferred_y': 'middle',
                    'anchor_labels': []},
 'Cây nước nóng lạnh': {'surface_options': ['floor'],
                        'relative_width': 0.84,
                        'relative_height': 1.58,
                        'preferred_x': 'edge',
                        'preferred_y': 'bottom',
                        'anchor_labels': []},
 'Hộp đựng đồ': {'surface_options': ['floor', 'table'],
                 'relative_width': 0.59,
                 'relative_height': 0.43,
                 'preferred_x': 'edge',
                 'preferred_y': 'middle',
                 'anchor_labels': []},
 'Thiết bị massage': {'surface_options': ['table', 'floor'],
                      'relative_width': 0.51,
                      'relative_height': 0.47,
                      'preferred_x': 'edge',
                      'preferred_y': 'middle',
                      'anchor_labels': []},
 'Máy tính để bàn': {'surface_options': ['table', 'floor'],
                     'relative_width': 0.7,
                     'relative_height': 1.01,
                     'preferred_x': 'edge',
                     'preferred_y': 'middle',
                     'anchor_labels': ['Table', 'Desk']}})

# ===== CELL 39 =====
CATEGORY_RULES.update({'Camera ngoài trời': {'surface_options': ['wall'],
                       'relative_width': 0.36,
                       'relative_height': 0.28,
                       'preferred_x': 'edge',
                       'preferred_y': 'top',
                       'anchor_labels': []},
 'Tủ lạnh': {'surface_options': ['floor'],
             'relative_width': 1.08,
             'relative_height': 1.94,
             'preferred_x': 'edge',
             'preferred_y': 'bottom',
             'anchor_labels': []},
 'Tủ mát': {'surface_options': ['floor'],
            'relative_width': 1.08,
            'relative_height': 1.94,
            'preferred_x': 'edge',
            'preferred_y': 'bottom',
            'anchor_labels': []},
 'Ấm đun nước': {'surface_options': ['table'],
                 'relative_width': 0.48,
                 'relative_height': 0.6,
                 'preferred_x': 'edge',
                 'preferred_y': 'middle',
                 'anchor_labels': []},
 'Máy lọc nước': {'surface_options': ['floor', 'table'],
                  'relative_width': 0.72,
                  'relative_height': 1.16,
                  'preferred_x': 'edge',
                  'preferred_y': 'bottom',
                  'anchor_labels': []},
 'Đèn năng lượng mặt trời': {'surface_options': ['wall'],
                             'relative_width': 0.36,
                             'relative_height': 0.42,
                             'preferred_x': 'edge',
                             'preferred_y': 'top',
                             'anchor_labels': []}})

# ===== CELL 41 =====
CATEGORY_RULES.update({'Bồn ngâm chân': {'surface_options': ['floor'],
                   'relative_width': 0.7,
                   'relative_height': 0.43,
                   'preferred_x': 'edge',
                   'preferred_y': 'bottom',
                   'anchor_labels': []},
 'Tủ đông': {'surface_options': ['floor'],
             'relative_width': 1.18,
             'relative_height': 0.96,
             'preferred_x': 'edge',
             'preferred_y': 'bottom',
             'anchor_labels': []},
 'Máy giặt': {'surface_options': ['floor'],
              'relative_width': 1.16,
              'relative_height': 1.27,
              'preferred_x': 'edge',
              'preferred_y': 'bottom',
              'anchor_labels': []},
 'Máy sấy quần áo': {'surface_options': ['floor'],
                     'relative_width': 1.16,
                     'relative_height': 1.3,
                     'preferred_x': 'edge',
                     'preferred_y': 'bottom',
                     'anchor_labels': []},
 'Cân sức khỏe': {'surface_options': ['floor'],
                  'relative_width': 0.47,
                  'relative_height': 0.17,
                  'preferred_x': 'center',
                  'preferred_y': 'bottom',
                  'anchor_labels': []},
 'Máy nước nóng': {'surface_options': ['wall'],
                   'relative_width': 0.71,
                   'relative_height': 1.05,
                   'preferred_x': 'edge',
                   'preferred_y': 'top',
                   'anchor_labels': []}})

# ===== CELL 43 =====
CATEGORY_RULES.update({'Máy rửa chén': {'surface_options': ['floor', 'table'],
                  'relative_width': 1.09,
                  'relative_height': 1.23,
                  'preferred_x': 'edge',
                  'preferred_y': 'bottom',
                  'anchor_labels': []},
 'Nồi cơm điện': {'surface_options': ['table'],
                  'relative_width': 0.57,
                  'relative_height': 0.65,
                  'preferred_x': 'edge',
                  'preferred_y': 'middle',
                  'anchor_labels': []},
 'Bếp từ': {'surface_options': ['table'],
            'relative_width': 0.92,
            'relative_height': 0.3,
            'preferred_x': 'center',
            'preferred_y': 'middle',
            'anchor_labels': []},
 'Bếp hồng ngoại': {'surface_options': ['table'],
                    'relative_width': 0.92,
                    'relative_height': 0.3,
                    'preferred_x': 'center',
                    'preferred_y': 'middle',
                    'anchor_labels': []},
 'Bếp gas': {'surface_options': ['table'],
             'relative_width': 0.99,
             'relative_height': 0.4,
             'preferred_x': 'center',
             'preferred_y': 'middle',
             'anchor_labels': []},
 'Máy hút mùi': {'surface_options': ['wall'],
                 'relative_width': 1.23,
                 'relative_height': 0.57,
                 'preferred_x': 'center',
                 'preferred_y': 'top',
                 'anchor_labels': []}})

# ===== CELL 45 =====
CATEGORY_RULES.update({'Lò vi sóng': {'surface_options': ['table'],
                'relative_width': 0.83,
                'relative_height': 0.58,
                'preferred_x': 'edge',
                'preferred_y': 'middle',
                'anchor_labels': []},
 'Lò nướng': {'surface_options': ['table'],
              'relative_width': 0.95,
              'relative_height': 0.64,
              'preferred_x': 'edge',
              'preferred_y': 'middle',
              'anchor_labels': []},
 'Nồi chiên không dầu': {'surface_options': ['table'],
                         'relative_width': 0.6,
                         'relative_height': 0.73,
                         'preferred_x': 'edge',
                         'preferred_y': 'middle',
                         'anchor_labels': []},
 'Máy xay sinh tố': {'surface_options': ['table'],
                     'relative_width': 0.39,
                     'relative_height': 0.73,
                     'preferred_x': 'edge',
                     'preferred_y': 'middle',
                     'anchor_labels': []},
 'Máy ép trái cây': {'surface_options': ['table'],
                     'relative_width': 0.42,
                     'relative_height': 0.81,
                     'preferred_x': 'edge',
                     'preferred_y': 'middle',
                     'anchor_labels': []},
 'Máy làm sữa hạt': {'surface_options': ['table'],
                     'relative_width': 0.4,
                     'relative_height': 0.81,
                     'preferred_x': 'edge',
                     'preferred_y': 'middle',
                     'anchor_labels': []}})

# ===== CELL 47 =====
CATEGORY_RULES.update({'Bàn ủi': {'surface_options': ['table'],
            'relative_width': 0.64,
            'relative_height': 0.24,
            'preferred_x': 'edge',
            'preferred_y': 'middle',
            'anchor_labels': []},
 'Máy sấy tóc': {'surface_options': ['table'],
                 'relative_width': 0.42,
                 'relative_height': 0.3,
                 'preferred_x': 'edge',
                 'preferred_y': 'middle',
                 'anchor_labels': []},
 'Máy cạo râu': {'surface_options': ['table'],
                 'relative_width': 0.2,
                 'relative_height': 0.31,
                 'preferred_x': 'edge',
                 'preferred_y': 'middle',
                 'anchor_labels': []},
 'Máy tỉa lông': {'surface_options': ['table'],
                  'relative_width': 0.18,
                  'relative_height': 0.31,
                  'preferred_x': 'edge',
                  'preferred_y': 'middle',
                  'anchor_labels': []},
 'Máy tạo kiểu tóc': {'surface_options': ['table'],
                      'relative_width': 0.43,
                      'relative_height': 0.25,
                      'preferred_x': 'edge',
                      'preferred_y': 'middle',
                      'anchor_labels': []},
 'Máy rửa mặt': {'surface_options': ['table'],
                 'relative_width': 0.25,
                 'relative_height': 0.37,
                 'preferred_x': 'edge',
                 'preferred_y': 'middle',
                 'anchor_labels': []}})
CATEGORY_RULES.update({'Bàn chải đánh răng điện': {'surface_options': ['table'],
                             'relative_width': 0.19,
                             'relative_height': 0.39,
                             'preferred_x': 'edge',
                             'preferred_y': 'middle',
                             'anchor_labels': []},
 'Máy tăm nước': {'surface_options': ['table'],
                  'relative_width': 0.28,
                  'relative_height': 0.48,
                  'preferred_x': 'edge',
                  'preferred_y': 'middle',
                  'anchor_labels': []}})

SURFACE_NAMES_VI = {'floor':'sàn', 'wall':'tường', 'table':'bàn/kệ', 'ceiling':'trần'}
STATUS_NAMES_VI = {
    'suggested_unverified':'Có vùng đề xuất (chưa xác minh)',
    'unsupported_product':'Thiếu cấu hình sản phẩm',
    'surface_not_allowed':'Trần chỉ dành cho đèn chùm hoặc quạt trần',
    'segmentation_unavailable':'Không có mask phân đoạn',
    'no_surface_detected':'Không nhận diện được bề mặt',
    'missing_anchor':'Không thấy vật thể tham chiếu',
    'invalid_metadata':'Metadata không hợp lệ',
    'no_safe_region':'Không tìm được vùng hợp lệ',
    'no_relation_satisfying_region':'Không có vùng thỏa quan hệ vật thể',
    'no_ceiling_detected':'Không nhận diện được mép trần',
    'no_ceiling_satisfying_region':'Không có vùng tường nằm sát dưới trần',
}
ALLOWED_SURFACES = {'floor','wall','table','ceiling'}
PRODUCT_PLACEMENT_OVERRIDES = {}  # Optional per-SKU metadata overrides

# ===== CELL 49 =====
# These are priors in IMAGE coordinates, used only as gentle regularization or
# when no YOLO detections exist. They do NOT represent real-world dimensions.
DEFAULT_REGION_RATIOS = {'floor':(0.10,0.18), 'wall':(0.14,0.12),
                         'table':(0.10,0.13), 'ceiling':(0.12,0.11)}


def _initial_profile(rule, source='catalog_group_rule'):
    out = deepcopy(rule)
    surfaces = out.get('surface_options', [])
    out['surface'] = surfaces[0] if surfaces else None
    out['clearance_ratio'] = 0.02
    out['require_anchor'] = False
    out['profile_source'] = source
    out['needs_review'] = source != 'catalog_group_rule'
    return out

PLACEMENT_PROFILES = {g: _initial_profile(spec) for g, spec in CATEGORY_RULES.items()}


def _guess_unknown_group_profile(group):
    # Never silently classify a newly encountered group as verified.
    s = str(group).casefold()
    if any(t in s for t in ('trần', 'ceiling')): opts=['ceiling']
    elif any(t in s for t in ('treo', 'điều hòa', 'camera', 'hút mùi')): opts=['wall']
    elif any(t in s for t in ('tủ', 'ghế', 'máy giặt', 'máy sấy quần áo')): opts=['floor']
    else: opts=['table']
    return _initial_profile({'surface_options':opts,'relative_width':0.65,
                             'relative_height':0.65,'preferred_x':'edge',
                             'preferred_y':'middle', 'anchor_labels':[]},
                            source='auto_guessed_requires_review')


def sync_placement_profiles_with_catalog(catalog=None, display_audit=True):
    """Audit placement profiles for categories returned by the API (not local CSV)."""
    catalog = API_CATEGORY_PRODUCTS if catalog is None else catalog
    live_groups = sorted(set(catalog.keys()))
    missing = []
    for group in live_groups:
        if group not in PLACEMENT_PROFILES:
            PLACEMENT_PROFILES[group] = _guess_unknown_group_profile(group)
            missing.append(group)
    rows=[]
    for group in live_groups:
        p=PLACEMENT_PROFILES[group]
        rows.append({'category':group,'surface_options':','.join(p['surface_options']),
                     'relative_width':p['relative_width'],
                     'relative_height':p['relative_height'],
                     'profile_source':p['profile_source'],
                     'needs_review':p['needs_review'],
                     'n_products':len(catalog[group])})
    audit=pd.DataFrame(rows,columns=['category','surface_options','relative_width','relative_height','profile_source','needs_review','n_products'])
    print(f'Placement profiles: {len(live_groups)} categories from external API; '
          f'{len(missing)} new/unknown categories require review.')
    if missing: print('Cần duyệt metadata các nhóm mới:',missing)
    if display_audit and len(audit):
        display(audit)
    return audit

# Initialized empty until the remote recommendation API returns its groups.
PLACEMENT_PROFILE_AUDIT = sync_placement_profiles_with_catalog(display_audit=False)

# ===== CELL 51 =====
# Optional audit: placement profile summary for external API groups.
try:
    path=Path(OUT_DIR)/'placement_profiles_from_catalog.csv'
    path.parent.mkdir(parents=True,exist_ok=True)
    PLACEMENT_PROFILE_AUDIT.to_csv(path,index=False,encoding='utf-8-sig')
    print('Đã lưu bảng profile kiểm tra:',path)
except NameError:
    pass  # Lightweight unit-test context without OUT_DIR

# ===== CELL 53 =====
def resolve_product_profile(group_name, product=None):
    if group_name not in PLACEMENT_PROFILES:
        PLACEMENT_PROFILES[group_name] = _guess_unknown_group_profile(group_name)
    spec=deepcopy(PLACEMENT_PROFILES[group_name])
    pid=(product or {}).get('product_id')
    if pid in PRODUCT_PLACEMENT_OVERRIDES:
        spec.update(PRODUCT_PLACEMENT_OVERRIDES[pid])
        spec['profile_source']='sku_override'
    opts = spec.get('surface_options') or ([spec['surface']] if spec.get('surface') else [])
    if spec.get('surface') not in opts and spec.get('surface') in ALLOWED_SURFACES:
        opts = [spec['surface']] + [x for x in opts if x != spec['surface']]
    opts = [x for x in opts if x in ALLOWED_SURFACES]
    if not opts: return None
    spec['surface_options'] = opts
    spec['surface'] = opts[0]
    try:
        w=float(spec.get('relative_width',0.65))
        h=float(spec.get('relative_height',0.65))
        clear=float(spec.get('clearance_ratio',0.02))
    except (ValueError,TypeError): return None
    if not (0.10<=w<=3.0 and 0.10<=h<=3.0 and 0<clear<0.30):return None
    spec['relative_width']=w
    spec['relative_height']=h
    spec['clearance_ratio']=clear
    return spec

# Check the actual CSV, including changes since the previous notebook run.
# Updated geometry audit is displayed after all v4 rules are applied.


# ===== 10c.2 v4: PHYSICAL GEOMETRY + MOUNT TYPE =====
# Values are class-level rough priors in centimeters, NOT measured SKU dimensions.
# For reliable sizing, set PRODUCT_PLACEMENT_OVERRIDES[product_id]['size_cm']
# to the actual width/height/depth from the manufacturer.
CATEGORY_SIZE_PRIORS_CM = {'TV': (115, 68, 9),
 'Loa karaoke': (36, 73, 38),
 'Loa': (22, 36, 23),
 'Quạt trần': (120, 22, 120),
 'Máy điều hòa': (95, 31, 23),
 'Máy lọc không khí': (30, 65, 30),
 'Máy hút bụi': (27, 110, 25),
 'Robot hút bụi': (35, 10, 35),
 'Bóng đèn LED': (12, 12, 12),
 'Ghế massage': (85, 115, 125),
 'Microphone': (6, 23, 6),
 'Máy chiếu': (32, 13, 28),
 'Máy chơi game': (31, 12, 25),
 'Camera trong nhà': (11, 15, 11),
 'Quạt điều hòa': (39, 92, 36),
 'Quạt điện': (45, 115, 45)}

# ===== CELL 55 =====
CATEGORY_SIZE_PRIORS_CM.update({'Máy hút ẩm': (32, 56, 25),
 'Bộ lau nhà': (30, 125, 30),
 'Đèn điện/đèn sạc': (22, 42, 22),
 'Ổ cắm điện': (12, 10, 5),
 'Thùng rác': (30, 42, 30),
 'Khay/kệ đựng đồ': (38, 22, 28),
 'Màn hình máy tính': (62, 45, 17),
 'Laptop': (35, 23, 25),
 'Quạt sưởi': (30, 65, 27),
 'Máy pha cà phê': (25, 38, 35),
 'Cây nước nóng lạnh': (36, 105, 38),
 'Hộp đựng đồ': (35, 24, 30),
 'Thiết bị massage': (25, 24, 22),
 'Máy tính để bàn': (22, 47, 45),
 'Camera ngoài trời': (18, 14, 18),
 'Tủ lạnh': (70, 170, 70)})
CATEGORY_SIZE_PRIORS_CM.update({'Tủ mát': (70, 175, 70),
 'Ấm đun nước': (23, 25, 18),
 'Máy lọc nước': (38, 85, 35),
 'Đèn năng lượng mặt trời': (25, 34, 15),
 'Bồn ngâm chân': (40, 24, 35),
 'Tủ đông': (100, 88, 60),
 'Máy giặt': (60, 85, 64),
 'Máy sấy quần áo': (60, 85, 64),
 'Cân sức khỏe': (32, 5, 32),
 'Máy nước nóng': (42, 65, 27),
 'Máy rửa chén': (60, 85, 60),
 'Nồi cơm điện': (28, 27, 32),
 'Bếp từ': (55, 9, 37),
 'Bếp hồng ngoại': (55, 9, 37),
 'Bếp gas': (67, 17, 42),
 'Máy hút mùi': (70, 35, 45)})
CATEGORY_SIZE_PRIORS_CM.update({'Lò vi sóng': (48, 29, 39),
 'Lò nướng': (55, 35, 42),
 'Nồi chiên không dầu': (30, 34, 36),
 'Máy xay sinh tố': (20, 42, 20),
 'Máy ép trái cây': (23, 43, 27),
 'Máy làm sữa hạt': (22, 43, 22),
 'Bàn ủi': (30, 15, 13),
 'Máy sấy tóc': (26, 22, 10),
 'Máy cạo râu': (7, 17, 6),
 'Máy tỉa lông': (6, 18, 5),
 'Máy tạo kiểu tóc': (32, 8, 7),
 'Máy rửa mặt': (8, 16, 8),
 'Bàn chải đánh răng điện': (4, 25, 4),
 'Máy tăm nước': (15, 24, 13)})

# ===== CELL 57 =====
CATEGORY_MOUNT_TYPE = {'TV': 'wall_mount',
 'Máy hút mùi': 'wall_mount',
 'Đèn năng lượng mặt trời': 'wall_mount',
 'Camera ngoài trời': 'wall_mount',
 'Máy nước nóng': 'wall_mount',
 'Ổ cắm điện': 'wall_mount',
 'Máy điều hòa': 'wall_mount',
 'Quạt trần': 'ceiling_hanging',
 'Bóng đèn LED': 'ceiling_hanging',
 'Máy chiếu': 'table_top',
 'Robot hút bụi': 'floor_low',
 'Bồn ngâm chân': 'floor_low',
 'Cân sức khỏe': 'floor_low',
 'Máy hút bụi': 'floor_standing',
 'Máy giặt': 'floor_standing',
 'Ghế massage': 'floor_standing',
 'Cây nước nóng lạnh': 'floor_standing',
 'Tủ lạnh': 'floor_standing'}
CATEGORY_MOUNT_TYPE.update({'Máy lọc nước': 'floor_standing',
 'Bộ lau nhà': 'floor_standing',
 'Máy sấy quần áo': 'floor_standing',
 'Máy rửa chén': 'floor_standing',
 'Quạt sưởi': 'floor_standing',
 'Tủ đông': 'floor_standing',
 'Tủ mát': 'floor_standing',
 'Quạt điều hòa': 'floor_standing',
 'Quạt điện': 'floor_standing',
 'Thùng rác': 'floor_standing',
 'Máy lọc không khí': 'floor_standing',
 'Loa karaoke': 'floor_standing',
 'Máy hút ẩm': 'floor_standing',
 'Máy rửa mặt': 'table_top',
 'Máy tăm nước': 'table_top',
 'Loa': 'table_top',
 'Ấm đun nước': 'table_top',
 'Laptop': 'table_top'})
CATEGORY_MOUNT_TYPE.update({'Đèn điện/đèn sạc': 'table_top',
 'Bếp gas': 'table_top',
 'Nồi chiên không dầu': 'table_top',
 'Hộp đựng đồ': 'table_top',
 'Thiết bị massage': 'table_top',
 'Máy tạo kiểu tóc': 'table_top',
 'Nồi cơm điện': 'table_top',
 'Bếp hồng ngoại': 'table_top',
 'Lò nướng': 'table_top',
 'Máy xay sinh tố': 'table_top',
 'Máy ép trái cây': 'table_top',
 'Máy làm sữa hạt': 'table_top',
 'Bàn ủi': 'table_top',
 'Màn hình máy tính': 'table_top',
 'Máy tính để bàn': 'table_top',
 'Camera trong nhà': 'table_top',
 'Lò vi sóng': 'table_top',
 'Máy sấy tóc': 'table_top'})
CATEGORY_MOUNT_TYPE.update({'Máy pha cà phê': 'table_top',
 'Máy cạo râu': 'table_top',
 'Máy tỉa lông': 'table_top',
 'Khay/kệ đựng đồ': 'table_top',
 'Bếp từ': 'table_top',
 'Bàn chải đánh răng điện': 'table_top',
 'Máy chơi game': 'table_top',
 'Microphone': 'table_top'})

# ===== CELL 59 =====
MOUNT_TO_SURFACE = {
    'floor_standing':'floor', 'floor_low':'floor', 'table_top':'table',
    'wall_mount':'wall', 'ceiling_hanging':'ceiling'
}
CATEGORY_SPATIAL_RULES = {
    'Quạt điện':'near_seating_avoid_walkway',
    'Máy lọc không khí':'edge_with_clearance',
    'Quạt điều hòa':'near_seating_avoid_walkway',
    'Loa':'near_tv', 'Loa karaoke':'near_tv',
    'TV':'face_seating', 'Quạt trần':'ceiling_clear_of_fixtures',
    'Máy điều hòa':'upper_wall_clear', 'Máy hút mùi':'above_cooking',
    'Robot hút bụi':'floor_edge_docking',
    'Khay/kệ đựng đồ':'table_support', 'Đèn điện/đèn sạc':'table_support',
}
for _g,_dims in CATEGORY_SIZE_PRIORS_CM.items():
    _rule = CATEGORY_RULES[_g]
    _mt = CATEGORY_MOUNT_TYPE[_g]
    _surf = MOUNT_TO_SURFACE[_mt]
    _rule.update({
        'size_cm':dict(zip(('width','height','depth'),_dims)),
        'size_source':'category_prior_estimate',
        'mount_type':_mt,
        'spatial_rule':CATEGORY_SPATIAL_RULES.get(_g, 'clear_surface'),
        # Ambiguous type? Do not silently switch from shelf to floor: require SKU override.
        'surface_options':[_surf],
        'min_visible_fraction':0.95,
    })
# Update the already-instantiated dictionary (no need to rerun other cells).
PLACEMENT_PROFILES = {g:_initial_profile(rule) for g,rule in CATEGORY_RULES.items()}
for _g,_p in PLACEMENT_PROFILES.items():
    _p['needs_review'] = True  # Approximate class size, not verified product dimensions

def set_sku_geometry(product_id, width_cm, height_cm, depth_cm, mount_type=None):
    """Add verified SKU dimensions and optional variant mounting type."""
    size = {'width':float(width_cm),'height':float(height_cm),'depth':float(depth_cm)}
    if not all(0 < v < 500 for v in size.values()):
        raise ValueError('Kích thước SKU phải là cm, nằm trong khoảng (0, 500)')
    override=PRODUCT_PLACEMENT_OVERRIDES.setdefault(str(product_id),{})
    override['size_cm']=size
    override['size_source']='sku_verified_input'  # Verify against manufacturer yourself
    if mount_type is not None:
        if mount_type not in MOUNT_TO_SURFACE:raise ValueError('mount_type không hỗ trợ')
        override['mount_type']=mount_type
        override['surface']=MOUNT_TO_SURFACE[mount_type]
        override['surface_options']=[MOUNT_TO_SURFACE[mount_type]]

_resolve_v3=resolve_product_profile

# ===== CELL 61 =====
def resolve_product_profile(group_name, product=None):
    spec=_resolve_v3(group_name,product)
    if spec is None:return None
    mt=spec.get('mount_type')
    if mt not in MOUNT_TO_SURFACE:
        spec['needs_review']=True
        # Unknown group defaults to its original surface, but isn't verified.
    else:
        expected=MOUNT_TO_SURFACE[mt]
        spec['surface_options']=[expected]
        spec['surface']=expected
    dims=spec.get('size_cm')
    if not isinstance(dims,dict) or not all(
            isinstance(dims.get(k),(float,int)) and 0<dims[k]<500
            for k in ('width','height','depth')):
        spec['size_cm']=None
        spec['needs_review']=True
    else:
        spec['needs_review'] = spec.get('size_source')!='sku_verified_input'
    spec['group_name']=group_name
    return spec

# (Notebook audit code omitted)

# ===== CELL 63 =====
# ==== HARD RULE: ceiling -> chandelier or ceiling fan ONLY ====
# Enforced at both product-profile resolution and candidate search; SKU overrides
# and unknown categories cannot silently bypass this condition.
import unicodedata

CEILING_ALLOWED_KINDS = ('chandelier', 'ceiling_fan')
CEILING_REJECT_REASON = ('Bề mặt trần (ceiling) chỉ nhận Đèn chùm (Chandelier) '
                         'hoặc Quạt trần; không tạo bbox cho các nhóm khác.')


def _normalized_product_label(value):
    value = unicodedata.normalize('NFKD', str(value or '').casefold())
    value = value.replace('đ', 'd')
    return ''.join(ch for ch in value if not unicodedata.combining(ch))


def _ceiling_kind_for(group_name, product_name=None):
    # Use explicit product/category identity, not mount_type or a guessed surface.
    for original in (group_name, product_name):
        label = _normalized_product_label(original)
        if 'quat tran' in label or 'ceiling fan' in label:
            return 'ceiling_fan'
        if 'den chum' in label or 'chandelier' in label or 'chanderli' in label:
            return 'chandelier'
    return None


def _ceiling_profile_is_allowed(profile):
    return _ceiling_kind_for(profile.get('group_name'),
                              profile.get('product_name')) in CEILING_ALLOWED_KINDS


_resolve_product_profile_before_ceiling_rule = resolve_product_profile


def resolve_product_profile(group_name, product=None):
    spec = _resolve_product_profile_before_ceiling_rule(group_name, product)
    if spec is None:
        return None
    spec = deepcopy(spec)
    name = (product or {}).get('product_name', '')
    spec['group_name'] = group_name
    spec['product_name'] = name
    kind = _ceiling_kind_for(group_name, name)
    spec['ceiling_kind'] = kind

    # Chandelier SKUs may live under a generic lighting category. This is only
    # a SKU-level exception; the remaining lamps keep their ordinary surfaces.
    if kind == 'chandelier':
        spec['surface'] = 'ceiling'
        spec['surface_options'] = ['ceiling']
        spec['mount_type'] = 'ceiling_hanging'
        spec['spatial_rule'] = 'ceiling_clear_of_fixtures'
        spec['preferred_x'], spec['preferred_y'] = 'center', 'top'
        spec['anchor_labels'] = []
        if spec.get('size_source') != 'sku_verified_input':
            spec['size_cm'] = {'width': 60.0, 'height': 45.0, 'depth': 60.0}
            spec['size_source'] = 'chandelier_type_prior_unverified'
            spec['needs_review'] = True

    options = spec.get('surface_options') or ([spec['surface']] if spec.get('surface') else [])
    if kind not in CEILING_ALLOWED_KINDS:
        filtered = [s for s in options if s != 'ceiling']
        if 'ceiling' in options and not filtered:
            # No arbitrary fallback: lamp bulbs / other ceiling-only profiles
            # must be explicitly reviewed and given a supported mounting type.
            spec['blocked_surface'] = 'ceiling'
            spec['placement_blocked_status'] = 'surface_not_allowed'
            spec['placement_blocked_reason'] = CEILING_REJECT_REASON
            spec['needs_review'] = True
            spec['surface_options'] = []
            spec['surface'] = 'ceiling'
            return spec
        options = filtered
    spec['surface_options'] = options
    if options:
        spec['surface'] = options[0]
    return spec

# Make profile audit reflect the effective rule, not only the old static rules.
PLACEMENT_PROFILE_AUDIT['ceiling_allowed'] = PLACEMENT_PROFILE_AUDIT['category'].map(
    lambda g: _ceiling_kind_for(g) in CEILING_ALLOWED_KINDS)
PLACEMENT_PROFILE_AUDIT['effective_surfaces'] = PLACEMENT_PROFILE_AUDIT['category'].map(
    lambda g: ','.join(resolve_product_profile(g).get('surface_options', []))
              if resolve_product_profile(g) is not None else '')
print('Ceiling placement policy: chỉ Đèn chùm hoặc Quạt trần; SKU ngoại lệ được kiểm tra riêng.')

# ===== CELL 65 =====
# ===== Cell 10c.2.20: air conditioner must mount on WALL, BELOW detected ceiling =====
# These are image-space heuristics, NOT installation clearance specifications.
AIRCON_CEILING_RULE = {
    'min_gap_image_ratio': 0.012,       # >= 1.2% image height beneath ceiling edge
    'max_gap_image_ratio': 0.095,       # <= 14.5% image height beneath ceiling edge
    'ideal_gap_image_ratio': 0.035,     # soft preference within hard range
    'min_ceiling_column_coverage': 0.60,
    'max_ceiling_edge_spread_ratio': 0.075,
}

def _is_air_conditioner_profile(profile):
    if not profile: return False
    label = _normalized_product_label(profile.get('group_name', '')).strip()
    return label in {'may dieu hoa', 'may lanh', 'air conditioner', 'air conditioning'}

_previous_resolve_product_profile_aircon = resolve_product_profile

def resolve_product_profile(group_name, product=None):
    profile = _previous_resolve_product_profile_aircon(group_name, product)
    if profile is None: return None
    if _is_air_conditioner_profile(profile):
        profile = deepcopy(profile)
        # Enforced regardless of a mistaken SKU surface/mount_type override.
        profile['surface'] = 'wall'
        profile['surface_options'] = ['wall']
        profile['mount_type'] = 'wall_mount'
        profile['spatial_rule'] = 'upper_wall_below_ceiling'
        profile['ceiling_relation_rule'] = dict(AIRCON_CEILING_RULE)
        profile['ceiling_relation_required'] = True
    return profile

print('Air conditioner: tường, ngay dưới trần; không lấn trần hoặc vật thể khác.')

# ===== CELL 67 =====
# ===== 10c.2.21: V7 CATEGORY RULEBOOK =====
# IMPORTANT: The thresholds are image-space heuristics (not installation specs).
from copy import deepcopy

NINE_CATEGORY_RULEBOOK_V7 = {
    'Máy điều hòa': dict(surface='wall', mount_type='wall_mount',
        spatial_rule='upper_wall_below_ceiling', clearance_ratio=0.040,
        aspect_ratio_bounds=(1.8, 4.5),
        hard=['tường', 'nằm dưới ranh trần-tường', 'không chồng trần/cửa/TV/vật thể'],
        soft=['ưu tiên gần phần tường cao', 'ở trên khu vực Sofa/Chair nếu phù hợp']),
    'TV': dict(surface='wall', mount_type='wall_mount',
        spatial_rule='face_seating', clearance_ratio=0.030,
        aspect_ratio_bounds=(1.5, 2.3),
        hard=['treo trên tường hoặc SKU có chân đặt trên mặt kệ',
              'không nằm trên cửa sổ/cửa ra vào', 'giữ khung ngang',
              'TV treo tường phải đạt kích thước bbox theo scale policy từ SKU'],
        soft=['căn giữa kệ TV nếu có', 'ưu tiên vùng nhìn về khu vực ghế ngồi',
              'ưu tiên TV ở thấp hơn, dựa trên mép trần-tường và mép tường-sàn tại vùng TV']),
    'Máy lọc không khí': dict(surface='floor', mount_type='floor_standing',
        spatial_rule='edge_with_clearance', clearance_ratio=0.040,
        aspect_ratio_bounds=(0.32, 1.30), min_floor_support_coverage=0.72,
        hard=['chân đặt trên sàn', 'không chắn cửa/lối đi', 'có khoảng trống quanh thân'],
        soft=['gần khu vực sinh hoạt', 'ưu tiên sát tường nhưng không kín cửa hút/xả']),
    'Ghế massage': dict(surface='floor', mount_type='floor_standing',
        spatial_rule='recline_clear_zone', clearance_ratio=0.075,
        aspect_ratio_bounds=(0.40, 1.65), min_floor_support_coverage=0.76,
        hard=['chân ghế đặt trên sàn', 'dành thêm không gian ngả/gác chân',
              'không chặn lối đi, cửa hoặc cầu thang'],
        soft=['gần tường hoặc góc phòng đủ rộng', 'gần khu vực nghỉ ngơi']),
    'Loa': dict(surface='table', mount_type='table_top',
        spatial_rule='near_tv', clearance_ratio=0.025,
        aspect_ratio_bounds=(0.40, 1.7),
        hard=['phân loại loa trước khi đặt', 'loa kệ phải có mặt đỡ'],
        soft=['loa kệ trên kệ và dưới TV khi có anchor', 'soundbar dưới TV']),
    'Loa karaoke': dict(surface='floor', mount_type='floor_standing',
        spatial_rule='near_tv', clearance_ratio=0.050,
        aspect_ratio_bounds=(0.30, 1.30), min_floor_support_coverage=0.73,
        hard=['loa lớn đứng trên sàn; loa mini cần cấu hình SKU',
              'không chắn lối đi/cửa'],
        soft=['gần TV', 'ưu tiên hai bên khu giải trí']),
    'Quạt trần': dict(surface='ceiling', mount_type='ceiling_hanging',
        spatial_rule='ceiling_clear_of_fixtures', clearance_ratio=0.060,
        aspect_ratio_bounds=(1.40, 7.0), min_ceiling_coverage=0.80,
        hard=['chỉ trần', 'ít nhất 80% khung hình chiếu ở trần',
              'không chồng đèn/quạt/cửa sổ'],
        soft=['tâm vùng trần khả dụng', 'xa đèn treo']),
    'Quạt điều hòa': dict(surface='floor', mount_type='floor_standing',
        spatial_rule='near_seating_avoid_walkway', clearance_ratio=0.050,
        aspect_ratio_bounds=(0.30, 1.10), min_floor_support_coverage=0.72,
        hard=['đặt trên sàn, không treo tường',
              'không chắn lối đi, có vùng thông gió'],
        soft=['gần Sofa/Chair', 'cạnh tường nếu không chắn luồng gió']),
    'Robot hút bụi': dict(surface='floor', mount_type='floor_low',
        spatial_rule='floor_edge_docking', clearance_ratio=0.040,
        aspect_ratio_bounds=(1.35, 7.0), min_floor_support_coverage=0.85,
        max_height_image_ratio=0.12, min_floor_component_fraction=0.02,
        hard=['đặt thấp trên sàn liên thông', 'không đè chân ghế/bàn/giường',
              'không ở cạnh mép ảnh'],
        soft=['gần tường có thể bố trí dock', 'có khoảng trống phía trước dock']),
}

# Subtypes have different supports and silhouette dimensions. Values are PRIOR estimates.
V7_SPEAKER_TYPES = {
    'bookshelf': {'mount_type':'table_top', 'surface':'table',
                  'size_cm':{'width':22.,'height':35.,'depth':24.},
                  'aspect_ratio_bounds':(0.40,1.70)},
    'soundbar': {'mount_type':'table_top', 'surface':'table',
                 'size_cm':{'width':90.,'height':10.,'depth':14.},
                 'aspect_ratio_bounds':(3.0,12.0)},
    'floorstanding': {'mount_type':'floor_standing', 'surface':'floor',
                      'size_cm':{'width':32.,'height':110.,'depth':36.},
                      'aspect_ratio_bounds':(0.25,1.0)},
}

def v7_classify_speaker(product_name, karaoke=False):
    name=_normalized_product_label(product_name)
    if any(k in name for k in ('soundbar','loa thanh','loa dang thanh')):
        return 'soundbar'
    if any(k in name for k in ('loa dung','floorstanding','tower speaker','loa cot')):
        return 'floorstanding'
    if karaoke:
        if any(k in name for k in ('mini','de ban','desktop')):
            return 'bookshelf'
        return 'floorstanding'
    return 'bookshelf'

_resolve_product_profile_v6 = resolve_product_profile
def resolve_product_profile(group_name, product=None):
    """V7 overlays 9 explicit category constraints without changing other groups."""
    profile=_resolve_product_profile_v6(group_name,product)
    if profile is None or group_name not in NINE_CATEGORY_RULEBOOK_V7:
        return profile
    profile=deepcopy(profile)
    rule=NINE_CATEGORY_RULEBOOK_V7[group_name]
    profile.update({k:deepcopy(v) for k,v in rule.items()
                    if k not in ('hard','soft')})
    profile['group_name']=group_name
    profile['product_name']=(product or {}).get('product_name','')
    profile['hard_rules_descriptions']=list(rule['hard'])
    profile['soft_rules_descriptions']=list(rule['soft'])
    profile['rulebook_version']='V7.1'
    # If a SKU's dimensions are manufacturer-supplied, never overwrite them.
    verified=profile.get('size_source')=='sku_verified_input'
    if group_name in ('Loa','Loa karaoke'):
        subtype=v7_classify_speaker(profile['product_name'],karaoke=group_name=='Loa karaoke')
        profile['speaker_type']=subtype
        var=V7_SPEAKER_TYPES[subtype]
        for key in ('mount_type','surface','aspect_ratio_bounds'):
            profile[key]=deepcopy(var[key])
        # Respect a verified SKU mounting choice (e.g. wall-mounted soundbar).
        pid=str((product or {}).get('product_id',''))
        sku_mount=((product or {}).get('mount_type') or
                   PRODUCT_PLACEMENT_OVERRIDES.get(pid,{}).get('mount_type'))
        if sku_mount in MOUNT_TO_SURFACE:
            profile['mount_type']=sku_mount
            profile['surface']=MOUNT_TO_SURFACE[sku_mount]
        if not verified:
            profile['size_cm']=deepcopy(var['size_cm'])
            profile['size_source']='v7_speaker_type_prior_unverified'
            profile['needs_review']=True
    if group_name=='TV':
        # SKU name may include a screen diagonal or model number.
        # The actual scaling logic is in D.3; no external object px/cm fallback.
        profile['tv_scaling']='sku_diagonal_plus_2d_wall_prior'
        label=_normalized_product_label(profile['product_name'])
        explicit_mount=(product or {}).get('mount_type') or (
            PRODUCT_PLACEMENT_OVERRIDES.get(str((product or {}).get('product_id','')),{}).get('mount_type'))
        if explicit_mount=='table_top' or any(k in label for k in ('de ban','dat ban','chan de')):
            profile.update(mount_type='table_top',surface='table',
                           spatial_rule='table_support')
    profile['surface_options']=[profile['surface']]
    # AC and ceiling rules from V6 still apply through the fixed surface.
    return profile

print('V7 Rulebook: 9 nhóm, Loa tách bookshelf/soundbar/floorstanding.')

# ===== CELL 69 =====
# D.2.22 — Placement policies phục vụ các category có trong API JSON mẫu.
# Không sử dụng dữ liệu này để tạo danh sách recommendation.
EXTERNAL_API_CATEGORY_POLICY_V813 = {
    'Thảm trải sàn': {
        'surface_options':['floor'], 'mount_type':'floor_low',
        'relative_width':0.90, 'relative_height':0.20,
        'preferred_x':'center', 'preferred_y':'bottom',
        'anchor_labels':[], 'spatial_rule':'floor_clear',
        'size_cm':{'width':120.,'height':2.,'depth':180.},
        'size_source':'category_prior_estimate',
    },
    'Tivi box & Đầu thu kỹ thuật số': {
        'surface_options':['table'], 'mount_type':'table_top',
        'relative_width':0.23, 'relative_height':0.18,
        'preferred_x':'center', 'preferred_y':'bottom',
        'anchor_labels':['TV'], 'spatial_rule':'near_tv',
        'size_cm':{'width':12.,'height':4.,'depth':12.},
        'size_source':'category_prior_estimate',
    },
}
for group,spec in EXTERNAL_API_CATEGORY_POLICY_V813.items():
    PLACEMENT_PROFILES[group]=_initial_profile(spec,source='external_api_explicit_placement_v813')
    PLACEMENT_PROFILES[group]['needs_review']=True

# Các category này có thể được recommend, nhưng không thể đặt đúng nếu chỉ có
# mask floor/wall/table/ceiling. Không tạo vị trí giả để làm đẹp ảnh.
API_UNSUPPORTED_SURFACE_GROUPS={
    'Gối, nệm':'Cần sofa/bed support surface; không thể đặt gối/nệm lên floor/table tùy ý.'
}

def api_product_placement_restriction(group,product):
    if group in API_UNSUPPORTED_SURFACE_GROUPS:
        return API_UNSUPPORTED_SURFACE_GROUPS[group]
    title=_fold_vi(product.get('product_name',''))
    if group=='Thảm trải sàn' and any(w in title for w in (
            'tam lot so sinh','tam lot chong tham','lot mama','top kids')):
        return 'SKU là tấm lót trẻ em, không phải thảm trải sàn để đặt trực tiếp lên floor.'
    return None

def apply_api_sku_surface_metadata(group,product):
    # Chỉ quy tắc placement; không chọn lại SKU đã random.
    if group!='Quạt điện':return
    name=_fold_vi(product.get('product_name',''))
    if 'quat ban' in name:
        kind,sz='table_top',{'width':34.,'height':44.,'depth':26.}
    elif 'quat hop' in name:
        kind,sz='floor_standing',{'width':34.,'height':42.,'depth':28.}
    else:
        return
    PRODUCT_PLACEMENT_OVERRIDES[str(product.get('product_id',''))]={
        'mount_type':kind,'surface':MOUNT_TO_SURFACE[kind],
        'surface_options':[MOUNT_TO_SURFACE[kind]],
        'size_cm':sz,'size_source':'sku_subtype_prior_estimate',
    }

# ===== CELL 71 =====
# D.2.23 — Product subtype và vùng đặt an toàn từ mapping V8.15.
# Các category không xác định không được tự tạo profile bằng guess_unknown_group.
_v814_apply_api_sku_surface_metadata = apply_api_sku_surface_metadata

def apply_api_sku_surface_metadata(group,product):
    _v814_apply_api_sku_surface_metadata(group,product)
    name=_fold_vi(product.get('product_name',''))
    pid=str(product.get('product_id') or '')
    if group=='Quạt điện':
        if 'quat mini' in name or 'quat ban' in name:
            kind,sz='table_top',{'width':30.,'height':35.,'depth':25.}
        elif 'quat treo tuong' in name:
            kind,sz='wall_mount',{'width':42.,'height':42.,'depth':25.}
        elif any(k in name for k in ('quat dung','quat cay','quat hop')):
            kind,sz='floor_standing',{'width':43.,'height':105.,'depth':42.}
        else:
            return
        PRODUCT_PLACEMENT_OVERRIDES[pid]={
            'mount_type':kind,'surface':MOUNT_TO_SURFACE[kind],
            'surface_options':[MOUNT_TO_SURFACE[kind]],
            'size_cm':sz,'size_source':'sku_subtype_prior_estimate',
        }
    elif group=='Robot hút bụi':
        PRODUCT_PLACEMENT_OVERRIDES[pid]={
            'mount_type':'floor_low','surface':'floor','surface_options':['floor'],
            'size_cm':{'width':35.,'height':10.,'depth':35.},
            'size_source':'sku_subtype_prior_estimate',
        }

_v814_product_placement_restriction = api_product_placement_restriction
def api_product_placement_restriction(group,product):
    reason=_v814_product_placement_restriction(group,product)
    if reason:return reason
    if group=='Quạt điện':
        name=_fold_vi(product.get('product_name',''))
        if not any(k in name for k in (
            'quat ban','quat mini','quat dung','quat cay','quat hop','quat treo tuong')):
            return 'Chưa nhận ra subtype quạt; không xác định được sàn/bàn/tường.'
    if group=='Máy chiếu':
        title=_fold_vi(product.get('product_name',''))
        if 'may chieu' not in title:
            return 'SKU không phải thân máy chiếu; cần placement riêng.'
    return None

# ===== CELL 73 =====
# ===== Cell 10c.3: MASK2FORMER =====
import numpy as np
import torch
from PIL import Image

SEGMENTATION_MODEL_ID = 'facebook/mask2former-swin-small-ade-semantic'
_SEG_CACHE = {}
SEGMENTATION_SURFACE_LABELS = {
    'floor': 'floor', 'flooring': 'floor',
    'wall': 'wall', 'ceiling': 'ceiling',
    'table': 'table', 'desk': 'table',
    'countertop': 'table', 'counter': 'table',
}


# These are NOT valid placement supports. They provide semantic exclusion masks
# for windows, doors, stairs, fireplaces, and hanging/standing lamps.
SEGMENTATION_OBSTACLE_LABELS = {
    'window': 'window', 'windowpane': 'window',
    'door': 'door', 'screen door': 'door',
    'stairway': 'stairs', 'stairs': 'stairs', 'staircase': 'stairs',
    'fireplace': 'fireplace',
    'lamp': 'light_fixture', 'light': 'light_fixture',
    'chandelier': 'light_fixture',
}


def extract_api_spatial_detections(objects, image_size):
    """Chỉ nhận bbox thực sự xuất hiện trong JSON API. Không bịa tọa độ từ label/confidence.

    Hỗ trợ bbox=[x1,y1,x2,y2] (pixel). Nếu API chỉ trả label/confidence,
    sẽ trả [] và placement vẫn chạy với mask Mask2Former.
    """
    W,H=image_size
    found=[]
    for obj in objects or []:
        if not isinstance(obj,dict): continue
        raw=obj.get('bbox_xyxy',obj.get('bbox'))
        if isinstance(raw,dict): raw=[raw.get(k) for k in ('x1','y1','x2','y2')]
        if not isinstance(raw,(list,tuple)) or len(raw)!=4:continue
        try:
            x1,y1,x2,y2=map(float,raw)
        except (TypeError,ValueError):continue
        if not all(np.isfinite([x1,y1,x2,y2])):continue
        if 0<=x1<x2<=W and 0<=y1<y2<=H and x2-x1>=2 and y2-y1>=2:
            found.append({'label':str(obj.get('label','Object')),
                          'confidence':float(obj.get('confidence',0.0)),
                          'bbox':[x1,y1,x2,y2]})
    return found

# ===== CELL 75 =====
def segment_room_surfaces(img, model_id=SEGMENTATION_MODEL_ID, device_name=None):
    """Run Mask2Former ADE20K, return ({surface: bool_mask(H,W)}, metadata).

    No SegFormer fallback or fabricated surface masks when inference fails.
    """
    try:
        from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
        if not isinstance(img, Image.Image):
            img = Image.fromarray(np.asarray(img)).convert('RGB')
        else:
            img = img.convert('RGB')
        device_name = device_name or ('cuda' if torch.cuda.is_available() else 'cpu')
        key = (model_id, device_name)
        if key not in _SEG_CACHE:
            processor = AutoImageProcessor.from_pretrained(model_id)
            seg_model = Mask2FormerForUniversalSegmentation.from_pretrained(model_id)
            seg_model = seg_model.to(device_name).eval()
            _SEG_CACHE[key] = (processor, seg_model)
        processor, seg_model = _SEG_CACHE[key]
        inputs = processor(images=img, return_tensors='pt')
        inputs = {k: v.to(device_name) for k, v in inputs.items()}
        with torch.inference_mode():
            outputs = seg_model(**inputs)
            semantic_map = processor.post_process_semantic_segmentation(
                outputs, target_sizes=[(img.height, img.width)])[0]
        predicted = semantic_map.cpu().numpy()
        masks = {s: np.zeros((img.height, img.width), dtype=bool)
                 for s in ALLOWED_SURFACES}
        # Some id2label values contain multiple ADE20K synonyms separated by commas.
        for category_id, raw_label in seg_model.config.id2label.items():
            names = [s.strip().casefold() for s in str(raw_label).split(',')]
            for name in names:
                surface = SEGMENTATION_SURFACE_LABELS.get(name)
                if surface:
                    masks[surface] |= predicted == int(category_id)
        # Semantic obstacles: crucial when detector has no explicit Window/Door classes.
        # All masks align to the ORIGINAL image pixels.
        obstacles={s:np.zeros((img.height,img.width),dtype=bool)
                   for s in sorted(set(SEGMENTATION_OBSTACLE_LABELS.values()))}
        for category_id, raw_label in seg_model.config.id2label.items():
            names = [s.strip().casefold() for s in str(raw_label).split(',')]
            for name in names:
                obstacle=SEGMENTATION_OBSTACLE_LABELS.get(name)
                if obstacle is not None:
                    obstacles[obstacle] |= predicted == int(category_id)
        masks.update(obstacles)
        counts = {s: int(m.sum()) for s, m in masks.items()}
        return masks, {
            'status': 'ok', 'backend': 'mask2former_ade20k',
            'model_id': model_id, 'device': device_name,
            'pixels_per_surface': counts,
        }
    except Exception as exc:
        return {}, {'status': 'unavailable', 'backend': 'mask2former_ade20k',
                    'model_id': model_id,
                    'reason': f'{type(exc).__name__}: {exc}'}


# ===== 10c.3 v4: OPTIONAL RELATIVE DEPTH (NOT METRIC) =====
DEPTH_MODEL_ID='depth-anything/Depth-Anything-V2-Small-hf'
_DEPTH_CACHE={}

# ===== CELL 77 =====
# ===== D.3.1: Tu mask WALL chung -> cac MAT TUONG rieng =====
# Khong suy dien la mot mat tuong lien tuc chi vi Mask2Former tra label 'wall'.
# Xac dinh duong giao bang hinh hoc duong tran-tuong va tuong-san.
from scipy.ndimage import median_filter

WALL_PLANE_POLICY_V83 = {
    'corner_band_image_fraction': 0.023,        # vung cam quanh giao tuong-tuong
    'corner_band_min_px': 9,
    'uncertain_edge_image_fraction': 0.12,   # neu khong nhin ro giao: tranh mep anh
    'bbox_same_plane_min_fraction': 0.995,    # gan nhu toan bo bbox cung mot mat
    'allow_side_wall_mount': False,            # chua warp phoi canh -> mac dinh chi tuong chinh
    'min_side_change_wall_height_fraction': 0.035,
    'min_corner_fit_improvement': 0.34,
    'min_plane_area_image_fraction': 0.025,
    'debug_wall_planes': True,
}


def _wall_column_edges(mask):
    arr=np.asarray(mask,dtype=bool)
    H,W=arr.shape
    count=arr.sum(axis=0)
    valid=count>=max(12,int(round(H*0.12)))
    top=np.where(valid,np.argmax(arr,axis=0),np.nan).astype(float)
    bottom=np.where(valid,H-1-np.argmax(arr[::-1,:],axis=0),np.nan).astype(float)
    def smooth(v):
        known=np.isfinite(v)
        if known.sum()<max(12,int(.32*W)):
            return None
        interp=np.interp(np.arange(W),np.where(known)[0],v[known])
        span=max(3,int(round(.025*W)) | 1)
        return median_filter(interp,size=span,mode='nearest')
    return smooth(top),smooth(bottom)


def _wall_edge_reference(edge):
    W=len(edge); xs=np.arange(W,dtype=float)
    a=max(0,int(.36*W));b=min(W,int(.64*W))
    if b-a<8:return np.full(W,float(np.median(edge)))
    c=np.polyfit(xs[a:b],edge[a:b],1)
    return np.polyval(c,xs)


def _fit_wall_corner(edge,side):
    """Detect piecewise-linear perspective boundary, not raw RGB color alone.

    A corner is accepted only when a sloping side-wall boundary explains
    substantially more of the ceiling/floor silhouette than one front plane.
    """
    W=len(edge); xx=np.arange(W,dtype=float)
    baseline=_wall_edge_reference(edge)
    delta=(edge-baseline)
    if side=='left':
        roi=xx<=.57*W
        starts,stops=.075,.43
    else:
        roi=xx>=.43*W
        starts,stops=.57,.925
    xr=xx[roi]; dy=delta[roi]
    old=float(np.sum(dy*dy))
    if old<1e-6:return None
    best=None
    for k in np.linspace(starts*W,stops*W,74):
        hinge=np.maximum(0,k-xr) if side=='left' else np.maximum(0,xr-k)
        denom=float(np.sum(hinge*hinge))
        if denom<1e-6:continue
        slope=float(np.sum(dy*hinge)/denom)
        err=float(np.sum((dy-slope*hinge)**2))
        if best is None or err<best['error']:
            best={'x':float(k),'error':err,'slope':slope,
                  'improvement':float(1-err/old),
                  'edge_amplitude':abs(slope)*(k if side=='left' else W-k)}
    return best


def infer_wall_plane_masks(image,masks,policy=None):
    """Create mutually exclusive wall-plane masks with perspective corner gaps.

    Semantic segmentation alone cannot label which wall is which. For images
    where ceiling/floor geometry cannot resolve corners, confidence is marked
    low and the algorithm must not claim a verified 3D wall plane.
    """
    cfg=WALL_PLANE_POLICY_V83 if policy is None else policy
    wall=np.asarray(masks.get('wall',[]),bool)
    if wall.ndim!=2 or not wall.any():
        return {}, {'status':'unavailable','reason':'Mask tường trống'}
    H,W=wall.shape
    top,bottom=_wall_column_edges(wall)
    if top is None or bottom is None:
        return {},{'status':'unavailable','reason':'Không đủ pixel tường xác định ranh giới'}
    candidates={}; detected={}
    for side in ('left','right'):
        fit_top=_fit_wall_corner(top,side)
        fit_bottom=_fit_wall_corner(bottom,side)
        accepted=[]
        for source,fit in (('ceiling_wall',fit_top),('wall_floor',fit_bottom)):
            if not fit:continue
            if (fit['improvement']>=cfg['min_corner_fit_improvement'] and
                fit['edge_amplitude']>=cfg['min_side_change_wall_height_fraction']*H):
                accepted.append((source,fit))
        if not accepted:continue
        # A single boundary can suffice when the other is occluded/poorly segmented.
        ktop=fit_top['x'] if fit_top and ('ceiling_wall',fit_top) in accepted else None
        kbottom=fit_bottom['x'] if fit_bottom and ('wall_floor',fit_bottom) in accepted else None
        if ktop is None:ktop=kbottom
        if kbottom is None:kbottom=ktop
        if ktop is None or kbottom is None:continue
        # Large disagreement is a telltale segmentation artefact, not a safe corner.
        if abs(ktop-kbottom)>.16*W:continue
        detected[side]={'top_x':float(ktop),'bottom_x':float(kbottom),
            'evidence':[s for s,_ in accepted],
            'confidence':'geometry_supported' if len(accepted)==2 else 'partial_geometry',
            'top_improvement':round(fit_top['improvement'],3) if fit_top else None,
            'bottom_improvement':round(fit_bottom['improvement'],3) if fit_bottom else None}
    yy,xx=np.indices((H,W),dtype=float)
    margin=max(cfg['corner_band_min_px'],int(round(cfg['corner_band_image_fraction']*W)))
    front=wall.copy();corner_forbidden=np.zeros((H,W),dtype=bool)
    for side,geom in detected.items():
        tx,bx=geom['top_x'],geom['bottom_x']
        # Interpolate along each visible front-wall span; continuous with y.
        # Top/bottom line intersections may differ (oblique photo perspective).
        mid=int(round((tx+bx)*.5))
        y_top=float(np.clip(top[np.clip(mid,0,W-1)],0,H-1))
        y_bottom=float(np.clip(bottom[np.clip(mid,0,W-1)],y_top+1,H-1))
        alpha=np.clip((yy-y_top)/max(1.,y_bottom-y_top),0.,1.)
        line=tx+(bx-tx)*alpha
        corner_forbidden |= wall & (np.abs(xx-line)<=margin)
        if side=='left':
            front &= xx>line+margin
            candidates['left']=wall & (xx<line-margin)
        else:
            front &= xx<line-margin
            candidates['right']=wall & (xx>line+margin)
    # Không có bằng chứng về giao ở một phía KHÔNG có nghĩa là phía đó
    # chắc chắn là cùng mặt tường. Chỉ chấp nhận vùng nội thất an toàn hơn.
    uncertainty_band=int(round(cfg['uncertain_edge_image_fraction']*W))
    if 'left' not in detected:front &= xx>=uncertainty_band
    if 'right' not in detected:front &= xx<W-uncertainty_band
    planes={'wall_plane_front':front}
    for side,arr in candidates.items():planes['wall_plane_'+side]=arr
    planes['wall_junction_forbidden']=corner_forbidden
    usable={k:int(v.sum()) for k,v in planes.items()
            if k.startswith('wall_plane_')}
    viable=[k for k,n in usable.items() if n>=cfg['min_plane_area_image_fraction']*W*H]
    if 'wall_plane_front' not in viable:
        return {},{'status':'unresolved','reason':'Không xác định được mặt tường chính đủ lớn',
                   'corners':detected}
    method=('perspective_ceiling_floor_boundary' if detected else
            'single_visible_wall_no_corner_evidence')
    return planes,{'status':'ok','method':method,'corners':detected,
                   'corner_margin_px':margin,'plane_pixels':usable,
                   'plane_ids':viable,'certainty':('estimated' if not detected else 'geometry_inferred'),
                   'note':'Tách mặt tường là ước lượng 2D, không phải phép đo 3D'}


_segment_room_surfaces_pre_wall_planes = segment_room_surfaces

def segment_room_surfaces(img, model_id=SEGMENTATION_MODEL_ID, device_name=None):
    masks,meta=_segment_room_surfaces_pre_wall_planes(img,model_id,device_name)
    if meta.get('status')!='ok':return masks,meta
    plane_masks,plane_meta=infer_wall_plane_masks(img,masks)
    masks.update(plane_masks)
    meta['wall_planes']=plane_meta
    return masks,meta


def display_wall_planes_debug(image,masks,output_path=None):
    """Show front/left/right wall regions + forbidden corners; save for QA."""
    from PIL import Image
    src=np.array(image.convert('RGB') if isinstance(image,Image.Image)
                 else Image.fromarray(np.asarray(image)).convert('RGB'),dtype=np.uint8)
    out=src.astype(np.float32).copy()
    tones={'wall_plane_front':(48,180,115),'wall_plane_left':(64,140,245),
           'wall_plane_right':(230,159,62),'wall_junction_forbidden':(232,52,56)}
    for name,rgb in tones.items():
        m=np.asarray(masks.get(name,[]),dtype=bool)
        if m.shape!=src.shape[:2]:continue
        alpha=.16 if name!='wall_junction_forbidden' else .70
        out[m]=(1-alpha)*out[m]+alpha*np.asarray(rgb,dtype=float)
    img=Image.fromarray(np.clip(out,0,255).astype('uint8'))
    if output_path is not None:img.save(output_path)
    return img

# ===== CELL 79 =====
# D.3.2 — V8.7: Vertical wall-junction evidence from RGB + Mask2Former.
# Runs AFTER semantic wall segmentation; does not change RT-DETR, recommendation,
# depth scaling, no-ceiling floor reference, or STRICT no-fallback background API.
import cv2
from PIL import Image, ImageDraw, ImageFont

WALL_CORNER_VISUAL_POLICY_V87 = {
    'left_roi_fraction': (0.16, 0.55),
    'right_roi_fraction': (0.45, 0.86),
    'scan_y_fraction': (0.04, 0.75),
    'min_wall_pair_row_fraction': 0.22,
    'min_wall_strip_coverage': 0.62,
    'min_lab_appearance_step': 18.0,
    'min_strong_contrast_row_fraction': 0.45,
    'strong_contrast_lab_step': 22.0,
    'minimum_corner_score': 21.0,
    'min_vertical_line_fraction': 0.15,
    'max_vertical_line_slant_fraction': 0.13,
    'min_side_wall_continuity': 0.46,
    'corner_band_image_fraction': 0.034,
    'corner_band_min_px': 12,
    'geometry_disagreement_fraction': 0.07,
    'use_rgb_vertical_seams': True,
    'debug': True,
}

_infer_wall_plane_masks_before_v87 = infer_wall_plane_masks


def _v87_detect_visual_corner(image, masks, side, cfg=None):
    """Return visual wall-to-wall seam only with vertical-edge AND wall evidence.

    Door/window edges are discouraged by requiring Mask2Former wall pixels on
    both sides of the line and continuity on the candidate front-wall side.
    If ambiguous/weak, return None. Never guess a corner from color alone.
    """
    cfg = WALL_CORNER_VISUAL_POLICY_V87 if cfg is None else cfg
    wall = np.asarray(masks.get('wall', []), dtype=bool)
    if wall.ndim != 2 or not wall.any():
        return None
    H, W = wall.shape
    if min(H, W) < 80:
        return None
    rgb = np.asarray(image.convert('RGB') if isinstance(image, Image.Image)
                     else Image.fromarray(np.asarray(image)).convert('RGB'), dtype=np.uint8)
    if rgb.shape[:2] != (H, W):
        rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_LINEAR)
    lab = cv2.GaussianBlur(cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32),
                           (5, 5), 1.2)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 1.2), 35, 100)
    min_length = max(30, int(cfg['min_vertical_line_fraction'] * H))
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                            threshold=max(20, int(0.08 * H)),
                            minLineLength=min_length,
                            maxLineGap=max(10, int(0.065 * H)))
    frac = cfg['left_roi_fraction'] if side == 'left' else cfg['right_roi_fraction']
    xlo, xhi = int(frac[0] * W), int(frac[1] * W)
    verticals = []
    if lines is not None:
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            dy, dx = abs(int(y2) - int(y1)), abs(int(x2) - int(x1))
            if dy < min_length or dx > max(4, cfg['max_vertical_line_slant_fraction'] * dy):
                continue
            xmid = 0.5 * (int(x1) + int(x2))
            if xlo <= xmid <= xhi:
                verticals.append((xmid, dy))
    if not verticals:
        return None
    ylo = max(0, int(cfg['scan_y_fraction'][0] * H))
    yhi = min(H, int(cfg['scan_y_fraction'][1] * H))
    pad = max(3, int(0.008 * W))
    strip = max(4, int(0.013 * W))
    if yhi - ylo < 25:
        return None
    # Explicitly exclude non-wall semantic masks, even if wall labels leak.
    support = wall.copy()
    for name in ('window', 'door', 'stairs', 'fireplace'):
        m = np.asarray(masks.get(name, []), dtype=bool)
        if m.shape == support.shape:
            support &= ~m
    tested = []
    search_lo = max(xlo, pad + strip)
    search_hi = min(xhi, W - pad - strip - 1)
    for x in range(search_lo, search_hi):
        l1, l2 = x - pad - strip, x - pad
        r1, r2 = x + pad, x + pad + strip
        left_support = support[ylo:yhi, l1:l2].mean(axis=1)
        right_support = support[ylo:yhi, r1:r2].mean(axis=1)
        paired = ((left_support >= cfg['min_wall_strip_coverage']) &
                  (right_support >= cfg['min_wall_strip_coverage']))
        paired_count = int(paired.sum())
        if paired_count < max(22, cfg['min_wall_pair_row_fraction'] * (yhi - ylo)):
            continue
        near_lines = [length for xm, length in verticals
                      if abs(xm - x) <= max(5, int(0.025 * W))]
        if not near_lines:
            continue
        # Longitudinal appearance change, rather than a single bright pixel.
        left_lab = lab[ylo:yhi, l1:l2].mean(axis=1)
        right_lab = lab[ylo:yhi, r1:r2].mean(axis=1)
        differences = np.linalg.norm(left_lab - right_lab, axis=1)[paired]
        color_step = float(np.percentile(differences, 55))
        coherence = float((differences >= cfg['strong_contrast_lab_step']).mean())
        if (color_step < cfg['min_lab_appearance_step'] or
                coherence < cfg['min_strong_contrast_row_fraction']):
            continue
        # Side-specific continuity: the interior (the main/front wall) should
        # remain a large uninterrupted wall region, unlike the inside of a window.
        reach = max(12, int(0.14 * W))
        if side == 'left':
            a, b = min(W, x + pad + strip), min(W, x + reach)
        else:
            a, b = max(0, x - reach), max(0, x - pad - strip)
        if b <= a:
            continue
        continuity = float(support[ylo:yhi, a:b].mean())
        if continuity < cfg['min_side_wall_continuity']:
            continue
        line_support = min(1.0, max(near_lines) / max(1.0, 0.42 * H))
        coverage_support = min(1.0, paired_count / max(1.0, 0.45 * (yhi - ylo)))
        score = (color_step * (0.25 + 0.75 * coherence) *
                 (0.60 + 0.40 * line_support) * coverage_support *
                 (0.65 + 0.35 * continuity))
        if score >= cfg['minimum_corner_score']:
            tested.append((score, x, color_step, coherence,
                           continuity, paired_count, line_support))
    if not tested:
        return None
    # The score may peak a few px inside the brighter wall due to patch width.
    # The forbidden margin protects this uncertainty as well as fixture width.
    score, x, step, coh, contin, paired_count, line_support = max(tested)
    return {'x': int(x), 'score': round(float(score), 2),
            'lab_step': round(float(step), 2),
            'coherent_rows_fraction': round(float(coh), 3),
            'interior_wall_continuity': round(float(contin), 3),
            'paired_wall_rows': int(paired_count),
            'vertical_edge_support': round(float(line_support), 3),
            'source': 'rgb_vertical_line+lab_change+two_wall_sides',
            'confidence': 'image_evidence_estimated'}


def infer_wall_plane_masks(image, masks, policy=None):
    """V8.7: refine V8.3 perspective planes using no-ceiling RGB wall seams."""
    planes, meta = _infer_wall_plane_masks_before_v87(image, masks, policy=policy)
    if not WALL_CORNER_VISUAL_POLICY_V87['use_rgb_vertical_seams']:
        return planes, meta
    wall = np.asarray(masks.get('wall', []), dtype=bool)
    if wall.ndim != 2 or not wall.any():
        return planes, meta
    H, W = wall.shape
    image_corners = {side: _v87_detect_visual_corner(image, masks, side)
                     for side in ('left', 'right')}
    image_corners = {side: d for side, d in image_corners.items() if d is not None}
    if not image_corners:
        if isinstance(meta, dict):
            meta['rgb_corner_evidence'] = 'not_detected'
            meta['rgb_corner_policy'] = 'no_unverified_corner_invented'
        return planes, meta
    # Apply visual evidence additionally to existing perspective evidence.
    # For disagreement, take the *more conservative* front-wall extent.
    base = np.asarray(planes.get('wall_plane_front', []), dtype=bool)
    if base.shape != (H, W):
        base = wall.copy()
    else:
        base = base.copy()
    forbidden = np.asarray(planes.get('wall_junction_forbidden', []), dtype=bool)
    if forbidden.shape != (H, W):
        forbidden = np.zeros((H, W), dtype=bool)
    else:
        forbidden = forbidden.copy()
    xx = np.arange(W, dtype=int)[None, :]
    band = max(WALL_CORNER_VISUAL_POLICY_V87['corner_band_min_px'],
               int(np.ceil(WALL_CORNER_VISUAL_POLICY_V87['corner_band_image_fraction'] * W)))
    diagnostics = {}
    corners = dict(meta.get('corners', {})) if isinstance(meta, dict) else {}
    for side, evidence in image_corners.items():
        x = evidence['x']
        geom = corners.get(side)
        conflicting = False
        if geom is not None:
            prior_x = float((geom.get('top_x', x) + geom.get('bottom_x', x)) / 2)
            conflicting = abs(prior_x - x) > WALL_CORNER_VISUAL_POLICY_V87['geometry_disagreement_fraction'] * W
        forbidden |= wall & (np.abs(xx - x) <= band)
        if side == 'left':
            base &= xx > x + band
            side_mask = wall & (xx < x - band)
        else:
            base &= xx < x - band
            side_mask = wall & (xx > x + band)
        name = 'wall_plane_' + side
        old = np.asarray(planes.get(name, []), dtype=bool)
        planes[name] = (old & side_mask) if old.shape == (H, W) else side_mask
        diagnostics[side] = {**evidence, 'corner_band_px': band,
                             'geometry_conflict': bool(conflicting)}
        if geom is None:
            corners[side] = {'top_x': float(x), 'bottom_x': float(x),
                             'evidence': ['rgb_vertical_wall_seam'],
                             'confidence': 'image_evidence_estimated'}
        else:
            corners[side] = {**geom, 'visual_x': float(x),
                             'visual_confidence': 'image_evidence_estimated',
                             'geometry_conflict': bool(conflicting)}
    planes['wall_plane_front'] = base
    planes['wall_junction_forbidden'] = forbidden
    if not base.any():
        return {}, {'status': 'unresolved', 'reason': 'front_wall_missing_after_visual_corner_filter',
                    'visual_corners': diagnostics, 'corners': corners}
    new_meta = dict(meta) if isinstance(meta, dict) else {}
    new_meta.update(status='ok', method='perspective+rgb_vertical_wall_seams_v87',
                    corners=corners, visual_corners=diagnostics,
                    rgb_corner_evidence='detected',
                    corner_margin_px=max(int(new_meta.get('corner_margin_px', 0)), band),
                    plane_pixels={k: int(v.sum()) for k, v in planes.items()
                                  if k.startswith('wall_plane_')},
                    certainty='image_evidence_estimated_not_metric')
    return planes, new_meta


# V8.3 segment_room_surfaces calls global infer_wall_plane_masks dynamically, so
# this updated function is automatically used for every new room image.

def display_wall_corner_debug_v87(image, masks, wall_plane_meta, output_path=None):
    """Render explicit vertical corner lines and keep the original photo intact."""
    src = image.convert('RGB').copy() if isinstance(image, Image.Image) else Image.fromarray(np.asarray(image)).convert('RGB')
    W, H = src.size
    drawer = ImageDraw.Draw(src)
    corners = wall_plane_meta.get('visual_corners', {}) if isinstance(wall_plane_meta, dict) else {}
    for side, corner in corners.items():
        x = int(round(corner['x']))
        band = int(corner['corner_band_px'])
        drawer.line([(x, 0), (x, H - 1)], fill=(239, 40, 41), width=max(2, W // 280))
        drawer.line([(max(0, x - band), 0), (max(0, x - band), H - 1)],
                    fill=(255, 156, 13), width=2)
        drawer.line([(min(W - 1, x + band), 0), (min(W - 1, x + band), H - 1)],
                    fill=(255, 156, 13), width=2)
        drawer.text((min(W - 130, x + 4), max(3, int(0.05 * H))),
                    f'{side}: x={x}', fill=(255, 40, 40), stroke_width=2,
                    stroke_fill=(255, 255, 255))
    if output_path is not None:
        src.save(output_path)
    return src

# ===== CELL 82 =====
def estimate_relative_depth(img, model_id=DEPTH_MODEL_ID, device_name=None):
    """Return raw relative depth prediction and metadata. Never invent depth on error.

    Depth Anything V2 pretrained relative depth is NOT meters. Image-space
    object size still requires a reference object or manually supplied camera
    calibration. Download only at inference time, never during unit tests.
    """
    try:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        if not isinstance(img,Image.Image):img=Image.fromarray(np.asarray(img)).convert('RGB')
        device_name=device_name or ('cuda' if torch.cuda.is_available() else 'cpu')
        key=(model_id,device_name)
        if key not in _DEPTH_CACHE:
            processor=AutoImageProcessor.from_pretrained(model_id)
            model=AutoModelForDepthEstimation.from_pretrained(model_id).to(device_name).eval()
            _DEPTH_CACHE[key]=(processor,model)
        processor,model=_DEPTH_CACHE[key]
        inputs=processor(images=img,return_tensors='pt')
        inputs={k:v.to(device_name) for k,v in inputs.items()}
        with torch.inference_mode():
            prediction=model(**inputs)
            depth=processor.post_process_depth_estimation(
                prediction,target_sizes=[(img.height,img.width)])[0]['predicted_depth']
        d=depth.detach().cpu().numpy().astype('float32')
        if d.shape!=(img.height,img.width) or not np.isfinite(d).all():
            raise ValueError('Invalid depth tensor')
        return d,{'status':'ok','model_id':model_id,'type':'relative_not_metric',
                  'device':device_name}
    except Exception as exc:
        return None,{'status':'unavailable','model_id':model_id,
                     'reason':f'{type(exc).__name__}: {exc}'}

# ===== CELL 84 =====
# ===== Spatial relationships: explicit rules only, no training =====
# IMPORTANT: Explicit human-defined constraints are NOT model learning.
import json
from collections import defaultdict

# Relations are expressed in image coordinates. They are not guaranteed 3D relations.
# require_if_visible means reject any candidate violating that relation WHEN its reference
# object can be detected, rather than returning an apparently confident wrong position.
SPATIAL_RELATION_RULES = {
    'Loa': [
        {'kind':'on_top_of','targets':['Shelf','Cabinet','Table','Side Table'],
         'weight':1.1,'require_if_visible':True,'name':'Trên kệ/tủ/bàn'},
        {'kind':'below','targets':['TV'],'weight':1.2,
         'require_if_visible':True,'name':'Dưới TV'},
    ],
    'Loa karaoke': [
        {'kind':'below','targets':['TV'],'weight':0.9,
         'require_if_visible':False,'name':'Bên dưới khu vực TV'},
        {'kind':'near','targets':['TV'],'weight':0.5,
         'require_if_visible':False,'name':'Gần TV'},
    ],
    'Máy điều hòa': [
        {'kind':'above','targets':['Sofa','Chair'],'weight':1.2,
         'require_if_visible':True,'name':'Ở trên khu vực Sofa/ghế'},
    ],
    'Đèn điện/đèn sạc': [
        {'kind':'on_top_of','targets':['Table','Side Table','Shelf','Cabinet'],
         'weight':0.8,'require_if_visible':True,'name':'Ở trên bàn/kệ'},
    ],
    'Khay/kệ đựng đồ': [
        {'kind':'on_top_of','targets':['Table','Side Table','Shelf','Cabinet'],
         'weight':0.8,'require_if_visible':True,'name':'Trên bề mặt bàn/kệ'},
    ],
    'Máy hút mùi': [
        {'kind':'above','targets':['Stove','Oven','Cooktop'],'weight':1.1,
         'require_if_visible':True,'name':'Bên trên bếp'},
    ],
    'Quạt điện': [{'kind':'near','targets':['Sofa','Chair'],'weight':0.7,
                  'require_if_visible':False,'name':'Gần ghế ngồi'}],
    'Quạt điều hòa': [{'kind':'near','targets':['Sofa','Chair'],'weight':0.6,
                      'require_if_visible':False,'name':'Gần khu vực ngồi'}],
    'TV': [{'kind':'above','targets':['Cabinet','Shelf'],'weight':0.7,
            'require_if_visible':False,'name':'Phía trên kệ/tủ TV'}],
    'Màn hình máy tính': [{'kind':'on_top_of','targets':['Table','Desk'],
                           'weight':0.8,'require_if_visible':True,'name':'Đặt trên bàn'}],
    'Laptop': [{'kind':'on_top_of','targets':['Table','Desk'],
                'weight':0.8,'require_if_visible':True,'name':'Đặt trên bàn'}],
}
GENERIC_SUPPORT_TARGETS=['Table','Side Table','Shelf','Cabinet','Desk']


def rules_for_product(profile):
    """Explicit overrides first; others receive a SOFT plausible relationship."""
    group=profile.get('group_name')
    if group in SPATIAL_RELATION_RULES:return SPATIAL_RELATION_RULES[group]
    if profile.get('mount_type') == 'table_top':
        return [{'kind':'on_top_of','targets':GENERIC_SUPPORT_TARGETS,
                 'weight':0.3,'require_if_visible':False,'name':'Trên bàn/kệ (gợi ý)'}]
    return []

# Không tải COCO/không huấn luyện spatial priors. Chỉ dùng quy tắc có sẵn.
SPATIAL_PRIORS = {}
SPATIAL_TRAINING_INFO = {'status':'disabled','reason':'API-only, no training'}
MIN_SPATIAL_PAIR_SAMPLES = 8

# ===== CELL 86 =====
# ===== 10c.3b.01: explicit relationship priorities =====
# rules_for_product is looked up at execution time inside spatial_relation_candidates().
_rules_for_product_v6 = rules_for_product

def rules_for_product(profile):
    group=profile.get('group_name')
    if group=='Loa':
        subtype=profile.get('speaker_type','bookshelf')
        if subtype=='floorstanding':
            return [{'kind':'near','targets':['TV'],'weight':0.75,
                     'require_if_visible':False,'name':'Loa đứng gần TV'}]
        if profile.get('surface')=='wall':
            return [{'kind':'below','targets':['TV'],'weight':1.4,
                     'require_if_visible':True,'name':'Loa treo tường ở dưới TV'}]
        if subtype=='soundbar':
            return [{'kind':'below','targets':['TV'],'weight':1.6,
                     'require_if_visible':True,'name':'Soundbar dưới TV'},
                    {'kind':'on_top_of','targets':['Shelf','Cabinet','Table','Side Table','Desk'],
                     'weight':1.1,'require_if_visible':True,'name':'Soundbar trên kệ nếu có'}]
        return [{'kind':'on_top_of','targets':['Shelf','Cabinet','Table','Side Table','Desk'],
                 'weight':1.4,'require_if_visible':True,'name':'Loa đặt trên kệ/bàn'},
                {'kind':'below','targets':['TV'],'weight':1.0,
                 'require_if_visible':True,'name':'Loa kệ dưới TV'}]
    if group=='Máy điều hòa':
        return [{'kind':'above','targets':['Sofa','Chair'],'weight':0.55,
                 'require_if_visible':False,'name':'Ưu tiên máy lạnh trên khu Sofa/Chair'}]
    if group=='TV':
        if profile.get('surface')=='table':
            return [{'kind':'on_top_of','targets':['Cabinet','Shelf','Table','Side Table','Desk'],
                     'weight':1.25,'require_if_visible':True,
                     'name':'Chân đế TV nằm trên mặt kệ'}]
        return [{'kind':'above','targets':['Cabinet','Shelf'],'weight':0.85,
                 'require_if_visible':False,'name':'Ưu tiên TV treo phía trên kệ'}]
    if group=='Máy lọc không khí':
        return [{'kind':'near','targets':['Sofa','Chair'],'weight':0.45,
                 'require_if_visible':False,'name':'Máy lọc gần khu sinh hoạt'}]
    if group=='Quạt điều hòa':
        return [{'kind':'near','targets':['Sofa','Chair'],'weight':0.7,
                 'require_if_visible':False,'name':'Quạt điều hòa gần chỗ ngồi'}]
    if group=='Loa karaoke':
        return [{'kind':'near','targets':['TV'],'weight':0.8,
                 'require_if_visible':False,'name':'Loa karaoke gần khu vực TV'}]
    return _rules_for_product_v6(profile)

# ===== CELL 89 =====
def spatial_relation_candidates(profile, detections, image_size,
                                x1,y1,x2,y2, with_learning=False):
    """Vectorized rule-based scores and hard requirements for candidate rectangles.

    bbox arrays must be in pixel coordinates of image_size.
    Rules require actual visible YOLO anchors; missing anchors are not fabricated.
    """
    W,H=map(float,image_size)
    x1,y1,x2,y2=[np.asarray(v,dtype=float) for v in (x1,y1,x2,y2)]
    n=len(x1)
    scores=np.zeros(n,dtype='float64')
    hard=np.ones(n,dtype=bool)
    matched=[]; missing=[]
    if n==0:return scores,hard,matched,missing
    cx=(x1+x2)/2;cy=(y1+y2)/2
    for rule in rules_for_product(profile):
        anchors=[d for d in detections if d.get('label') in rule['targets'] and d.get('bbox')]
        if not anchors:
            missing.append(rule['name'] + ' (thiếu YOLO anchor: '+', '.join(rule['targets'])+')')
            continue
        best_q=np.zeros(n)
        best_valid=np.zeros(n,dtype=bool)
        best_learn=np.zeros(n)
        for anchor in anchors:
            ax1,ay1,ax2,ay2=map(float,anchor['bbox'])
            acx=(ax1+ax2)/2;acy=(ay1+ay2)/2
            aw=max(1,ax2-ax1)
            gap_x=np.abs(cx-acx)
            horizontal=np.clip(1.0-gap_x/np.maximum(0.8*aw+0.5*(x2-x1)+0.06*W,1),0,1)
            overlap=np.maximum(0,np.minimum(x2,ax2)-np.maximum(x1,ax1))
            kind=rule['kind']
            if kind=='above':
                dy=(ay1-y2)/H
                good=(dy>=-0.025)&(overlap>=0.18*np.minimum(aw,x2-x1))
                q=np.exp(-np.abs(dy-0.07)/0.22)*horizontal
            elif kind=='below':
                dy=(y1-ay2)/H
                good=(dy>=-0.03)&(overlap>=0.15*np.minimum(aw,x2-x1))
                q=np.exp(-np.abs(dy-0.07)/0.2)*horizontal
            elif kind=='on_top_of':
                # Bottom of proposed product should meet *top* of supporting object.
                delta=(y2-ay1)/H
                ov=overlap/np.maximum(1,(x2-x1))
                good=(ov>=0.28)&(delta>=-0.105)&(delta<=0.09)
                q=np.exp(-np.abs(delta)/0.085)*np.clip(ov,0,1)
            elif kind=='near':
                dist=np.hypot((cx-acx)/W,(cy-acy)/H)
                good=dist<=0.48
                q=np.exp(-dist/0.24)
            else:
                continue
            learned=np.zeros(n)
            if with_learning:
                # Only if this PRODUCT class has actual YOLO annotation evidence.
                proxy=globals().get('GROUP_TO_DETECTOR',{}).get(profile.get('group_name'))
                prior=SPATIAL_PRIORS.get((proxy,anchor['label']))
                if prior and prior['n']>=MIN_SPATIAL_PAIR_SAMPLES:
                    dx=(cx-acx)/W
                    dy=(cy-acy)/H
                    learned=np.exp(-0.5*(
                        ((dx-prior['dx'])/max(prior['sigma_x'],.05))**2 +
                        ((dy-prior['dy'])/max(prior['sigma_y'],.05))**2))
            better=q>best_q
            best_q=np.maximum(best_q,q)
            best_learn=np.where(better,learned,best_learn)
            best_valid |= good
        if rule.get('require_if_visible',False):hard &= best_valid
        scores += float(rule['weight'])*(0.90*best_q + 0.10*best_learn)
        matched.append({'name':rule['name'],'kind':rule['kind'],
                        'targets':[a['label'] for a in anchors],
                        'require_if_visible':rule.get('require_if_visible',False),
                        'satisfied':bool(best_valid[0]) if n==1 else None,
                        'relation_score':round(float(best_q[0]),3) if n==1 else None,
                        'learned_pairs_used':[
                            {'target':globals().get('GROUP_TO_DETECTOR',{}).get(profile.get('group_name')),
                             'anchor':a['label'],
                             'n':SPATIAL_PRIORS[(globals().get('GROUP_TO_DETECTOR',{}).get(profile.get('group_name')),a['label'])]['n']}
                            for a in anchors if (globals().get('GROUP_TO_DETECTOR',{}).get(profile.get('group_name')),a['label']) in SPATIAL_PRIORS
                        ]})
    return scores,hard,matched,missing

# ===== CELL 91 =====
def attach_group_name_to_profiles():
    for g, profile in PLACEMENT_PROFILES.items():
        profile['group_name']=g
attach_group_name_to_profiles()
# Không initialize_spatial_learning hoặc đọc/train COCO.

# ===== CELL 93 =====
# ===== Cell 10c.4: SURFACE-CONTACT REGION SEARCH =====
from scipy.ndimage import distance_transform_edt, label as connected_components

DEFAULT_REGION_RATIOS = {
    'floor': (0.10, 0.18),
    'wall': (0.14, 0.12),
    'table': (0.10, 0.13),
    'ceiling': (0.12, 0.11),
}


def _unavailable_placement(status, reason, surface=None):
    return {
        'status': status, 'bbox_px': None, 'bbox_norm': None,
        'point_px': None, 'point_norm': None,
        'support_point_px': None, 'surface': surface,
        'anchor': None, 'placement_score': None, 'verified': False,
        'reason': reason,
        'relative_to_surface': 'Chưa xác định', 'surface_component_bbox_px': None,
        'surface_position': None, 'surface_relation': None,
        'surface_mask_pixels': None, 'assumed_profile': False,
    }


def _resize_bool_mask(mask, size):
    return np.asarray(Image.fromarray(np.asarray(mask, dtype=np.uint8))
                      .resize(size, resample=Image.Resampling.NEAREST)) > 0


def _bbox_overlaps_yolo(bbox, detections, margin=0):
    """bbox=(x1,y1,x2,y2), with exclusive right/bottom bounds."""
    x1, y1, x2, y2 = bbox
    for obj in detections:
        bx1, by1, bx2, by2 = map(float, obj['bbox'])
        if x1 < bx2 + margin and x2 > bx1 - margin and \
                y1 < by2 + margin and y2 > by1 - margin:
            return True
    return False


SUPPORT_DETECTOR_LABELS={'Shelf','Cabinet','Table','Side Table','Desk'}

def _inferred_support_from_yolo(detections, image_size):
    """Unverified top-of-shelf/cabinet bands from YOLO bbox, not a real tabletop mask."""
    W,H=map(int,image_size)
    mask=np.zeros((H,W),dtype=bool)
    for obj in detections:
        if obj.get('label') not in SUPPORT_DETECTOR_LABELS:continue
        x1,y1,x2,y2=map(int,obj['bbox'])
        x1=max(0,x1);x2=min(W,x2);y1=max(0,y1)
        if x1>=x2 or y1>=H:continue
        height=max(1,y2-y1)
        band=max(7,min(16,round(0.10*height)))
        mask[y1:min(H,y1+band),x1:x2]=True
    return mask

# ===== CELL 95 =====
def _collision_with_yolo_excluding_support(bbox,detections,surface,margin=0):
    """Never let a proposed bbox intersect ANY detected/reserved object.

    A tabletop speaker may touch the top edge of its supporting table but must
    not enter the detected table bounding box. Clearance margin applies to
    non-supporting objects; contact with a table/shelf has zero overlap tolerance.
    """
    x1,y1,x2,y2=map(float,bbox)
    for obj in detections:
        if not obj.get('bbox') or len(obj['bbox'])!=4:continue
        a,b,c,d=map(float,obj['bbox'])
        support=(surface=='table' and obj.get('label') in SUPPORT_DETECTOR_LABELS)
        effective_margin=0.0 if support else float(max(0,margin))
        if (x1<c+effective_margin and x2>a-effective_margin and
            y1<d+effective_margin and y2>b-effective_margin):
            return True
    return False


def _table_upper_support(mask):
    """Use top part of each sufficiently large table/desk segment as support candidates.
    ADE20K 'table' labels the entire object, NOT an explicitly segmented tabletop.
    """
    H, W = mask.shape
    comps, count = connected_components(mask)
    valid = np.zeros_like(mask)
    for i in range(1, count + 1):
        yy, xx = np.where(comps == i)
        if yy.size < max(12, int(0.0005 * H * W)):
            continue
        top, bottom = int(yy.min()), int(yy.max())
        upper = top + max(3, int(0.22 * (bottom - top + 1)))
        take = yy <= upper
        valid[yy[take], xx[take]] = True
    return valid


def _rect_counts(integral, x1, y1, x2, y2):
    return (integral[y2, x2] - integral[y1, x2] -
            integral[y2, x1] + integral[y1, x1])


def _integral(mask):
    return np.pad(mask.astype(np.int64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)


def _valid_support(mask, surface, bbox, support_band):
    """Strict original-resolution checks after low-resolution candidate search."""
    x1, y1, x2, y2 = bbox
    H, W = mask.shape
    if min(x1, y1) < 0 or x2 > W or y2 > H or x1 >= x2 or y1 >= y2:
        return False
    if surface == 'wall':
        return float(mask[y1:y2, x1:x2].mean()) >= 0.98
    xa = x1 + max(1, (x2-x1)//6)
    xb = x2 - max(1, (x2-x1)//6)
    if xa >= xb:
        return False
    if surface in ('floor', 'table'):
        ya, yb = y2, min(H, y2 + support_band)
    else:  # ceiling: contact above proposed hanging object
        ya, yb = max(0, y1 - support_band), y1
    if ya >= yb:
        return False
    return float(mask[ya:yb, xa:xb].mean()) >= 0.88

# ===== CELL 97 =====
def _surface_relation_at(mask, surface, point):
    """Spatial 2D relation to a connected mask COMPONENT; never infers depth.

    Relations are measured using the support point for suggested objects,
    never the bbox center (which may lie above a floor surface).
    """
    mask = np.asarray(mask, dtype=bool)
    H, W = mask.shape
    x = int(np.clip(round(point[0]), 0, W-1))
    y = int(np.clip(round(point[1]), 0, H-1))
    if not mask.any():
        return None
    if not mask[y, x]:
        # Find a supporting pixel nearby. Do not attach to a distant surface.
        radius = max(6, min(32, round(0.035 * min(H, W))))
        xa, xb = max(0, x-radius), min(W, x+radius+1)
        ya, yb = max(0, y-radius), min(H, y+radius+1)
        yy, xx = np.where(mask[ya:yb, xa:xb])
        if xx.size == 0:
            return None
        ind = np.argmin((xx+xa-x)**2 + (yy+ya-y)**2)
        x, y = int(xx[ind]+xa), int(yy[ind]+ya)
    labels, _ = connected_components(mask)
    component = labels[y, x]
    if component == 0:
        return None
    yy, xx = np.where(labels == component)
    x1, y1, x2, y2 = int(xx.min()), int(yy.min()), int(xx.max()+1), int(yy.max()+1)
    rx = float(np.clip((x-x1)/max(1, x2-x1), 0, 1))
    ry = float(np.clip((y-y1)/max(1, y2-y1), 0, 1))
    horizontal = 'bên trái' if rx < 1/3 else ('ở giữa' if rx < 2/3 else 'bên phải')
    vertical = 'phía trên' if ry < 1/3 else ('ở giữa' if ry < 2/3 else 'phía dưới')
    mode = {'floor':'chân đế tiếp xúc vùng sàn',
            'table':'đáy vật thể tiếp xúc vùng bàn/kệ (mặt trên ước lượng)',
            'wall':'vùng đề xuất nằm trên mask tường',
            'ceiling':'đỉnh vật thể tiếp xúc vùng trần'}[surface]
    return {
        'reference_point_px': {'x':x, 'y':y},
        'surface_component_bbox_px': {'x1':x1,'y1':y1,'x2':x2,'y2':y2},
        'surface_position': {'horizontal':horizontal,'vertical':vertical,
                             'relative_x':round(rx,4),'relative_y':round(ry,4)},
        'surface_relation': mode,
        'relative_to_surface': (
            f'{mode}; {horizontal}, {vertical} của vùng '
            f'{SURFACE_NAMES_VI.get(surface, surface)} (tọa độ ảnh 2D)'),
    }

# ===== CELL 99 =====
def _nearby_surface_context(bbox, masks, image_size, distance_maps=None):
    """Explain which detected surface is near an observed YOLO instance.

    This is only image-plane adjacency, NOT a proof that the object rests on
    or is mounted to that physical surface.
    """
    W,H=image_size
    x1,y1,x2,y2=map(float,bbox)
    x=int(np.clip((x1+x2)/2,0,W-1))
    center_y=int(np.clip((y1+y2)/2,0,H-1))
    foot_y=int(np.clip(y2,0,H-1))
    top_y=int(np.clip(y1,0,H-1))
    best=None
    for surface, raw in masks.items():
        if surface not in ALLOWED_SURFACES:
            continue
        m=np.asarray(raw,dtype=bool)
        if m.shape!=(H,W) or not m.any():
            continue
        py=top_y if surface=='ceiling' else (center_y if surface=='wall' else foot_y)
        # Distance is measured in the 2D image plane only.
        dist=(distance_maps.get(surface) if distance_maps is not None else None)
        if dist is None:
            dist=distance_transform_edt(~m)
        distance_px=float(dist[py,x])
        score=distance_px/max(1,min(H,W))
        if best is None or score<best['distance_ratio']:
            pos=_surface_relation_at(m,surface,(x,py))
            best={'surface':surface, 'distance_px':round(distance_px,2),
                  'distance_ratio':score,
                  'association':'giao với mask' if distance_px==0 else 'gần mask trong ảnh 2D',
                  'relative_to_surface': (
                      (f"Điểm tham chiếu {pos['surface_position']['horizontal']}, "
                       f"{pos['surface_position']['vertical']} vùng "
                       f"{SURFACE_NAMES_VI[surface]} (chỉ 2D, không suy ra chỗ đặt)")
                      if pos else 'Chưa xác định trên surface'),
                  'surface_component_bbox_px': (pos['surface_component_bbox_px'] if pos else None)}
    if best is None:
        return {'surface':None,'association':'không có mask bề mặt',
                'relative_to_surface':'Chưa xác định','distance_px':None}
    # Prevent overclaiming when only a distant surface exists.
    if best['distance_ratio']>0.08:
        best['association']='không đủ gần để xác nhận quan hệ 2D'
        best['relative_to_surface']='Không xác nhận được quan hệ với surface'
    best.pop('distance_ratio',None)
    return best

# ===== CELL 101 =====
def label_yolo_object_surface_relations(detections, masks, image_size):
    """Add surface-neighborhood labels to YOLO-detected objects."""
    out=[]
    W,H=image_size
    distance_maps={
        surface:distance_transform_edt(~np.asarray(raw,dtype=bool))
        for surface,raw in masks.items()
        if surface in ALLOWED_SURFACES and np.asarray(raw).shape==(H,W)
        and np.asarray(raw).any()
    }
    for obj in detections:
        row=deepcopy(obj)
        row['surface_context']=_nearby_surface_context(
            row['bbox'],masks,image_size,distance_maps=distance_maps)
        out.append(row)
    return out

# ===== CELL 103 =====
# ===== Cell 10c.4.05b: hard geometry of an AC bbox relative to ceiling mask =====
def _aircon_ceiling_geometry(ceiling_mask, x1, y1, x2, y2, profile):
    """Vectorized tests: AC must sit below nearby ceiling edge, on the wall.

    Returns (valid, gap_fraction_of_image_height, ceiling_column_coverage).
    Use *local* ceiling boundary above the SAME horizontal span, not image top.
    If segmentation is missing/ambiguous, reject rather than fabricate a position.
    """
    ceiling=np.asarray(ceiling_mask, dtype=bool)
    if ceiling.ndim != 2:raise ValueError('Ceiling mask must be a 2D array')
    H,W=ceiling.shape
    xx1,yy1,xx2,yy2=np.broadcast_arrays(*[
        np.asarray(v,dtype=float) for v in (x1,y1,x2,y2)])
    valid=np.zeros(xx1.shape,dtype=bool)
    gap=np.full(xx1.shape, np.nan,dtype=float)
    coverage=np.zeros(xx1.shape,dtype=float)
    if H==0 or W==0 or not ceiling.any():
        return valid,gap,coverage
    rule=profile.get('ceiling_relation_rule') or AIRCON_CEILING_RULE
    # The ceiling in an indoor photo should occupy the upper portion.
    search_h=max(1,round(H*0.62))
    ceiling_upper=ceiling[:search_h]
    yy=np.arange(search_h,dtype=int)[:,None]
    bottom=np.where(ceiling_upper,yy,-1).max(axis=0)
    enough=ceiling_upper.sum(axis=0)>=max(2,round(0.006*H))
    bottom=np.where(enough,bottom,-1)
    # Compare five columns spanning width of candidate AC.
    fracs=(0.12,0.30,0.50,0.70,0.88)
    positions=np.stack([np.clip(np.rint(xx1+(xx2-xx1)*frac).astype(int),0,W-1)
                        for frac in fracs],axis=0)
    seam=bottom[positions]
    present=(seam>=0)
    coverage=present.mean(axis=0)
    # Masked sum / count, then median-like robust mean ignoring absent columns.
    sorted_seam=np.sort(np.where(present,seam,np.inf),axis=0)
    middle_index=np.maximum(0,present.sum(axis=0)-1)//2
    selected=np.take_along_axis(sorted_seam,middle_index[None,...],axis=0)[0]
    selected=np.where(present.any(axis=0),selected,np.nan)
    spread=np.where(present,np.abs(seam-selected),0).max(axis=0)/max(1,H)
    gap=(yy1-selected)/max(1,H)
    min_gap=float(rule.get('min_gap_image_ratio',0.012))
    max_gap=float(rule.get('max_gap_image_ratio',0.145))
    required_coverage=float(rule.get('min_ceiling_column_coverage',0.60))
    spread_limit=float(rule.get('max_ceiling_edge_spread_ratio',0.13))
    valid=(coverage>=required_coverage)&(spread<=spread_limit)&(
        gap>=min_gap)&(gap<=max_gap)
    # Even if the wall mask was misclassified, do not put AC OVER ceiling pixels.
    # Integral image test is vectorized and exact at current resolution.
    xa=np.clip(np.floor(xx1).astype(int),0,W)
    xb=np.clip(np.ceil(xx2).astype(int),0,W)
    ya=np.clip(np.floor(yy1).astype(int),0,H)
    yb=np.clip(np.ceil(yy2).astype(int),0,H)
    valid &= (xb>xa)&(yb>ya)
    integral=_integral(ceiling)
    valid &= (_rect_counts(integral,xa,ya,xb,yb)==0)
    return valid,gap,coverage


def _pixel_bbox_intersects(a,b):
    """Strict 2D bbox overlap: touching edges is allowed; shared area is not."""
    if isinstance(a,dict):a=[a[k] for k in ('x1','y1','x2','y2')]
    if isinstance(b,dict):b=[b[k] for k in ('x1','y1','x2','y2')]
    return float(a[0])<float(b[2]) and float(a[2])>float(b[0]) and \
           float(a[1])<float(b[3]) and float(a[3])>float(b[1])

# ===== CELL 105 =====
# ===== 10c.4.05c — STRICT spatial policies (V6) =====
# Important: these are image-space constraints, NOT guaranteed real-world safety.
# If semantic segmentation is uncertain, prefer an unavailable placement over an invented one.
STRICT_PLACEMENT_POLICY = {
    'edge_x_ratio': 0.045,             # keep entire proposal away from left/right image edges
    'edge_y_ratio': 0.028,             # keep bbox away from top/bottom edges
    'forbidden_overlap_ratio': 0.012, # max semantic obstacle footprint in candidate box
    'floor_window_overlap_ratio': 0.015,
    'ceiling_fan_min_mask_coverage': 0.70,
    'ceiling_fan_min_width_height_ratio': 1.35,
    'wall_tv_min_width_height_ratio': 1.18,
    'aircon_min_width_height_ratio': 1.25,
    'robot_max_height_image_ratio': 0.16,
    'robot_min_support_y_ratio': 0.58,
}


def _semantic_obstacle_masks(masks, image_size):
    W,H=map(int,image_size)
    keys=('window','door','stairs','fireplace','light_fixture')
    return {k:np.asarray(masks.get(k,np.zeros((H,W),dtype=bool)),bool)
            for k in keys}


def _strict_candidate_validity(profile, masks, image_size, x1,y1,x2,y2):
    """Vectorized hard checks; coordinates must align to masks at image_size.

    Avoid uncertain physically implausible regions: doors/windows, stairs,
    ceiling-fan boxes falling on the wall, TV/AC wrong silhouette proportions,
    and floor vacuum robots vertically oversized.
    """
    W,H=map(int,image_size)
    x1,y1,x2,y2=np.broadcast_arrays(*[np.asarray(v,dtype=float)
                                       for v in (x1,y1,x2,y2)])
    valid=np.ones(x1.shape,dtype=bool)
    if min(W,H)<=0:return valid & False
    minx=int(round(W*STRICT_PLACEMENT_POLICY['edge_x_ratio']))
    miny=int(round(H*STRICT_PLACEMENT_POLICY['edge_y_ratio']))
    valid &= (x1>=minx)&(x2<=W-minx)&(y1>=miny)&(y2<=H-miny)
    xa=np.clip(np.floor(x1).astype(int),0,W)
    xb=np.clip(np.ceil(x2).astype(int),0,W)
    ya=np.clip(np.floor(y1).astype(int),0,H)
    yb=np.clip(np.ceil(y2).astype(int),0,H)
    valid &= (xb>xa)&(yb>ya)
    area=np.maximum(1,(xb-xa)*(yb-ya))
    surface=profile.get('surface','floor')
    label=_normalized_product_label(profile.get('group_name',''))
    obs=_semantic_obstacle_masks(masks,(W,H))
    # Window/door/stairs are never considered a safe installation region.
    avoid=['window','door','stairs','fireplace']
    if surface=='ceiling': avoid+=['light_fixture']
    for name in avoid:
        raw=obs[name]
        if raw.shape!=(H,W) or not raw.any():continue
        amount=_rect_counts(_integral(raw),xa,ya,xb,yb)/area
        limit=(STRICT_PLACEMENT_POLICY['floor_window_overlap_ratio']
               if surface=='floor' and name=='window'
               else STRICT_PLACEMENT_POLICY['forbidden_overlap_ratio'])
        valid &= amount<=limit
    bw=np.maximum(1,x2-x1)
    bh=np.maximum(1,y2-y1)
    if surface=='ceiling' and _ceiling_kind_for(profile.get('group_name'),
                                                profile.get('product_name'))=='ceiling_fan':
        ceiling=np.asarray(masks.get('ceiling',np.zeros((H,W),dtype=bool)),bool)
        if ceiling.shape!=(H,W) or not ceiling.any(): return valid & False
        ceiling_cover=_rect_counts(_integral(ceiling),xa,ya,xb,yb)/area
        valid &= ceiling_cover>=STRICT_PLACEMENT_POLICY['ceiling_fan_min_mask_coverage']
        valid &= bw/bh>=STRICT_PLACEMENT_POLICY['ceiling_fan_min_width_height_ratio']
    if surface=='wall' and label=='tv':
        valid &= bw/bh>=STRICT_PLACEMENT_POLICY['wall_tv_min_width_height_ratio']
        valid &= (y1>=0.12*H)&(y2<=0.78*H)
    if _is_air_conditioner_profile(profile):
        valid &= surface=='wall'
        valid &= bw/bh>=STRICT_PLACEMENT_POLICY['aircon_min_width_height_ratio']
        valid &= (y1<=0.48*H)
    if 'robot hut bui' in label:
        valid &= (bh/H<=STRICT_PLACEMENT_POLICY['robot_max_height_image_ratio'])
        valid &= (y2/H>=STRICT_PLACEMENT_POLICY['robot_min_support_y_ratio'])
    if surface=='floor':
        floor=np.asarray(masks.get('floor',np.zeros((H,W),dtype=bool)),bool)
        if floor.shape==(H,W) and floor.any():
            # A physically floor-standing product needs a real floor contact band,
            # and not merely any green pixel somewhere elsewhere in the image.
            base_top=np.maximum(ya,yb-np.maximum(2,np.rint(0.07*(yb-ya)).astype(int)))
            nfloor=_rect_counts(_integral(floor),xa,base_top,xb,yb)
            base_area=np.maximum(1,(xb-xa)*(yb-base_top))
            valid &= nfloor/base_area>=0.50
    return valid

# ===== CELL 107 =====
# ===== 10c.4.05d: V7 vectorized hard constraints =====
# Extends V6, never replaces its ceiling/wall/obstacle protection.
_strict_candidate_validity_v6 = _strict_candidate_validity
STRICT_PLACEMENT_POLICY['ceiling_fan_min_mask_coverage'] = 0.80

def _strict_candidate_validity(profile, masks, image_size, x1,y1,x2,y2):
    valid=_strict_candidate_validity_v6(profile,masks,image_size,x1,y1,x2,y2)
    group=profile.get('group_name','')
    if group not in NINE_CATEGORY_RULEBOOK_V7:
        return valid
    W,H=map(int,image_size)
    if W<1 or H<1:
        return valid & False
    x1,y1,x2,y2=np.broadcast_arrays(*[np.asarray(v,dtype=float)
                                      for v in (x1,y1,x2,y2)])
    bw=np.maximum(1.0,x2-x1)
    bh=np.maximum(1.0,y2-y1)
    surface=profile.get('surface')
    required=profile.get('mount_type')
    if required in MOUNT_TO_SURFACE and surface!=MOUNT_TO_SURFACE[required]:
        return valid & False
    # Object silhouette priors are 2D checks, not measurements in cm.
    lo,hi=profile.get('aspect_ratio_bounds',(0.12,12.0))
    valid &= ((bw/bh)>=float(lo)) & ((bw/bh)<=float(hi))
    if group=='Quạt trần':
        ceiling=np.asarray(masks.get('ceiling',[]),dtype=bool)
        if surface!='ceiling' or ceiling.shape!=(H,W) or not ceiling.any():
            return valid & False
        xa=np.clip(np.floor(x1).astype(int),0,W)
        xb=np.clip(np.ceil(x2).astype(int),0,W)
        ya=np.clip(np.floor(y1).astype(int),0,H)
        yb=np.clip(np.ceil(y2).astype(int),0,H)
        fraction=_rect_counts(_integral(ceiling),xa,ya,xb,yb) / np.maximum(1,(xb-xa)*(yb-ya))
        valid &= fraction >= profile.get('min_ceiling_coverage',0.80)
    if group in ('Máy lọc không khí','Ghế massage','Loa karaoke',
                 'Quạt điều hòa','Robot hút bụi') or (group=='Loa' and surface=='floor'):
        if surface=='floor':
            floor=np.asarray(masks.get('floor',[]),dtype=bool)
            if floor.shape!=(H,W) or not floor.any():return valid & False
            xa=np.clip(np.floor(x1+0.12*bw).astype(int),0,W)
            xb=np.clip(np.ceil(x2-0.12*bw).astype(int),0,W)
            ya=np.clip(np.ceil(y2).astype(int),0,H)
            yb=np.clip(ya+max(2,int(round(H*0.008))),0,H)
            supp=_rect_counts(_integral(floor),xa,ya,xb,yb)
            fraction=supp/np.maximum(1,(xb-xa)*(yb-ya))
            valid &= (yb>ya) & (fraction>=float(profile.get('min_floor_support_coverage',0.72)))
            if group=='Robot hút bụi':
                valid &= bh/H <= float(profile.get('max_height_image_ratio',0.12))
                # Reject disconnected floor islands; require a sufficiently large floor component.
                cc,count=connected_components(floor)
                sizes=np.bincount(cc.ravel())
                cx=np.clip(np.floor((x1+x2)/2).astype(int),0,W-1)
                yy=np.clip(np.ceil(y2+1).astype(int),0,H-1)
                ids=cc[yy,cx]
                valid &= (ids>0) & (sizes[ids]>=float(profile.get('min_floor_component_fraction',0.02))*W*H)
    if group=='TV' and surface not in ('wall','table'):
        return valid & False
    if group=='TV':
        # Enforce screen-size rule even when candidate search is called directly.
        # Rounding tolerance is necessary when masks are resized for search.
        policy=globals().get('TV_SCALE_POLICY')
        if policy:
            width_fraction=bw/max(1,W)
            valid &= width_fraction >= policy['min_width_image_ratio']-0.012
            valid &= width_fraction <= policy['max_width_image_ratio']+0.012
    if group=='Máy điều hòa' and surface!='wall':
        return valid & False
    return valid

# ===== CELL 109 =====
def _choose_placement_region_fixed(profile, masks, detections, image_size,
                            max_processing_side=640):
    """Find a proposed 2D rectangle supported by a surface, avoiding YOLO boxes.

    Floor/table: rectangle extends ABOVE its supporting contact point.
    Ceiling: rectangle extends BELOW the ceiling contact point.
    Wall: rectangle resides within predicted wall.
    This is a heuristic candidate, not verified physical placement.
    """
    W, H = map(int, image_size)
    if W <= 0 or H <= 0:
        raise ValueError('image_size khong hop le')
    surface = profile.get('surface')
    is_aircon=_is_air_conditioner_profile(profile)
    if is_aircon and surface!='wall':
        return _unavailable_placement('surface_not_allowed',
                'Máy lạnh chỉ được gắn trên tường, không đặt trên trần.',surface)
    if is_aircon:
        cm=np.asarray(masks.get('ceiling',[]),dtype=bool)
        if cm.shape!=(H,W) or not cm.any():
            return _unavailable_placement('no_ceiling_detected',
                'Không xác định được mép trần để kiểm tra vị trí máy lạnh.', 'wall')
    if surface == 'ceiling' and not _ceiling_profile_is_allowed(profile):
        return _unavailable_placement('surface_not_allowed',
                                      CEILING_REJECT_REASON, 'ceiling')
    if surface not in ALLOWED_SURFACES:
        return _unavailable_placement('unsupported_product', 'Thieu metadata surface')
    if surface not in masks and surface != 'table':
        return _unavailable_placement('segmentation_unavailable',
                                      'Khong co mask tu Mask2Former', surface)
    raw = np.asarray(masks.get(surface,np.zeros((H,W),dtype=bool)), dtype=bool)
    inferred_support=np.zeros((H,W),dtype=bool)
    if surface=='table':
        inferred_support=_inferred_support_from_yolo(detections,(W,H))
        raw=(raw | inferred_support)  # Inferred top is NOT a certified table plane.
    if raw.shape != (H, W):
        raise ValueError(f'Mask {surface}: {raw.shape} != {(H, W)}')
    if not raw.any():
        return _unavailable_placement('no_surface_detected',
                                      f'Khong thay mat {surface}', surface)
    try:
        clear = float(profile.get('clearance_ratio', 0.035))
        rw0, rh0 = DEFAULT_REGION_RATIOS[surface]
        rw = float(profile.get('region_width_ratio', rw0))
        rh = float(profile.get('region_height_ratio', rh0))
    except (ValueError, TypeError):
        return _unavailable_placement('invalid_metadata', 'Kich thuoc/clearance khong hop le', surface)
    if not (0 < clear < 0.30 and 0 < rw <= 0.70 and 0 < rh <= 0.70):
        return _unavailable_placement('invalid_metadata', 'Kich thuoc/clearance ngoai gioi han', surface)

    anchor_names = set(profile.get('anchor_labels', []))
    anchors = [o for o in detections if o['label'] in anchor_names]
    if profile.get('require_anchor', False) and not anchors:
        return _unavailable_placement('missing_anchor',
                f'Thieu vat the tham chieu: {sorted(anchor_names)}', surface)
    anchor = max(anchors, key=lambda o: o.get('confidence', 0)) if anchors else None

    scale = min(1, max_processing_side / max(W, H))
    w, h = max(1, round(W * scale)), max(1, round(H * scale))
    support = _resize_bool_mask(raw, (w, h))
    if surface == 'table':
        base=np.asarray(masks.get('table',np.zeros_like(raw)),dtype=bool)
        support=(_table_upper_support(_resize_bool_mask(base,(w,h))) |
                 _resize_bool_mask(inferred_support,(w,h)))
    occupied = np.zeros((h, w), dtype=bool)
    margin = max(1, round(min(h, w) * clear))  # SKU-specific 2D clearance
    for o in detections:
        bx1, by1, bx2, by2 = o['bbox']
        a = max(0, int(np.floor(bx1 * w / W)) - margin)
        b = min(w, int(np.ceil(bx2 * w / W)) + margin)
        if surface=='table' and o.get('label') in SUPPORT_DETECTOR_LABELS:
            by1 = min(by2, by1 + max(8, 0.18*(by2-by1)))
            c = max(0, int(np.floor(by1 * h / H)))
        else:
            c = max(0, int(np.floor(by1 * h / H)) - margin)
        d = min(h, int(np.ceil(by2 * h / H)) + margin)
        if a < b and c < d:
            occupied[c:d, a:b] = True

    bw = max(4, int(round(rw * w)))
    bh = max(4, int(round(rh * h)))
    band = max(2, int(round(h * 0.012)))
    if bw >= w or bh >= h:
        return _unavailable_placement('no_safe_region', 'Bbox lon hon anh', surface)
    stride = max(2, round(min(w, h) * 0.012))
    xs_grid = np.arange(bw // 2, w - (bw - bw // 2), stride, dtype=np.int32)
    if surface == 'ceiling':
        ystart, ystop = band, h - bh - band
    elif surface == 'wall':
        ystart, ystop = bh // 2 + band, h - bh // 2 - band
    else:
        ystart, ystop = bh + band, h - band
    ys_grid = np.arange(ystart, ystop, stride, dtype=np.int32)
    if xs_grid.size == 0 or ys_grid.size == 0:
        return _unavailable_placement('no_safe_region', 'Khong du khong gian', surface)
    xx, yy = np.meshgrid(xs_grid, ys_grid)
    xs, ys = xx.ravel(), yy.ravel()
    x1 = xs - bw // 2
    x2 = x1 + bw
    if surface == 'wall':
        y1 = ys - bh // 2
        y2 = y1 + bh
    elif surface in ('floor', 'table'):
        y2 = ys
        y1 = y2 - bh
    else:
        y1 = ys
        y2 = y1 + bh
    edge_pad = max(1, round(0.25 * clear * min(w, h)))
    in_bounds = ((x1 >= edge_pad) & (y1 >= edge_pad) &
                 (x2 <= w-edge_pad) & (y2 <= h-edge_pad))
    # Restrict candidate supporting point to the depth band used to size it.
    yrange=profile.get('candidate_y_range')
    if yrange is not None:
        lo,hi=yrange
        in_bounds &= (ys >= lo*h) & (ys < hi*h)
    xs, ys, x1, y1, x2, y2 = (arr[in_bounds] for arr in (xs, ys, x1, y1, x2, y2))
    if not xs.size:
        return _unavailable_placement('no_safe_region', 'Khong du khong gian', surface)

    # Every candidate must NOT overlap any YOLO detected object (2D projection).
    occ_sum = _integral(occupied)
    no_collision = _rect_counts(occ_sum, x1, y1, x2, y2) == 0
    surface_sum = _integral(support)
    if surface == 'wall':
        coverage = _rect_counts(surface_sum, x1, y1, x2, y2) / float(bw * bh)
        surface_ok = coverage >= 0.98
    else:
        xa = x1 + max(1, bw // 6)
        xb = x2 - max(1, bw // 6)
        if surface in ('floor', 'table'):
            ya, yb = y2, np.minimum(h, y2 + band)
        else:
            ya, yb = np.maximum(0, y1 - band), y1
        coverage = _rect_counts(surface_sum, xa, ya, xb, yb) / np.maximum(1, (xb - xa) * (yb - ya))
        surface_ok = coverage >= 0.90
    # V5: hard constraints for observed object-object relationships.
    rel_bonus, rel_hard, rel_matches, rel_missing = spatial_relation_candidates(
        profile,detections,(W,H),x1*W/w,y1*H/h,x2*W/w,y2*H/h)
    # Block candidate boxes crossing windows/doors/stairs/fixtures and invalid sizes.
    strict_masks={k:_resize_bool_mask(v,(w,h)) for k,v in masks.items()
                  if isinstance(v,np.ndarray) and v.ndim==2 and v.shape==(H,W)}
    strict_valid=_strict_candidate_validity(profile,strict_masks,(w,h),x1,y1,x2,y2)
    pre_strict_valid=no_collision & surface_ok
    strict_blocked=bool(np.any(pre_strict_valid) and
                        not np.any(pre_strict_valid & strict_valid))
    basic_valid=pre_strict_valid & strict_valid
    valid = basic_valid & rel_hard
    blocked_by_relations=bool(np.any(basic_valid) and not np.any(valid))
    ceiling_gap=np.full(valid.shape,np.nan)
    ceiling_coverage=np.zeros(valid.shape)
    if is_aircon:
        ceiling_scaled=_resize_bool_mask(masks['ceiling'],(w,h))
        ceil_ok,ceiling_gap,ceiling_coverage=_aircon_ceiling_geometry(
            ceiling_scaled,x1,y1,x2,y2,profile)
        valid &= ceil_ok
    blocked_by_ceiling=(is_aircon and bool(np.any(basic_valid & rel_hard))
                        and not bool(np.any(valid)))
    xs, ys, x1, y1, x2, y2, coverage, rel_bonus, ceiling_gap, ceiling_coverage = (
        arr[valid] for arr in (xs, ys, x1, y1, x2, y2, coverage, rel_bonus,
                               ceiling_gap,ceiling_coverage))
    if not xs.size:
        status=('no_ceiling_satisfying_region' if blocked_by_ceiling else
                'no_relation_satisfying_region' if blocked_by_relations else 'no_safe_region')
        why=('Có vùng tường nhưng máy lạnh không nằm sát dưới trần theo khoảng hở yêu cầu'
             if blocked_by_ceiling else
             'Có vùng trống nhưng không thỏa quan hệ với vật thể YOLO'
             if blocked_by_relations else
             'Không có vùng vừa khít bề mặt và không chồng lên vật thể khác')
        return _unavailable_placement(status,why,surface)

    # Candidate ranking: prioritize surface coverage, room centrality and anchor.
    distance = distance_transform_edt(~occupied)
    free_dist = distance[np.clip(ys, 0, h - 1), np.clip(xs, 0, w - 1)]
    free_score = np.minimum(1.0, free_dist / max(1, 0.14 * min(w, h)))
    # Prefer plausible vertical zones; still allow other zones if needed.
    prefer_y = {'floor': 0.78, 'table': 0.52, 'wall': 0.42, 'ceiling': 0.18}[surface] * h
    vertical = np.exp(-np.abs(ys - prefer_y) / max(1., 0.35 * h))
    central = 1.0 - np.abs(xs - 0.5 * w) / max(0.5 * w, 1.0)
    # Main room center at near-floor pixels is often the circulation aisle;
    # penalize it for floor-standing appliances (heuristic, not certified path).
    rule=profile.get('spatial_rule','clear_surface')
    aisle_risk=(surface=='floor') * (1-np.clip(np.abs(xs/w-0.5)/0.30,0,1)) * np.clip((ys/h-0.60)/0.35,0,1)
    scores = 0.48 * coverage + 0.31 * free_score + 0.13 * vertical + 0.08 * central
    if rule in ('near_seating_avoid_walkway','edge_with_clearance','floor_edge_docking','recline_clear_zone'):
        scores = scores-0.28*aisle_risk
    if is_aircon:
        ideal=float(profile.get('ceiling_relation_rule',AIRCON_CEILING_RULE)
                    .get('ideal_gap_image_ratio',0.045))
        scores += 0.30*np.exp(-np.abs(ceiling_gap-ideal)/0.055)
    if rule in ('upper_wall_clear','upper_wall_below_ceiling','ceiling_clear_of_fixtures'):
        scores = scores + 0.14*(1-ys/h)
    if rule=='floor_edge_docking':
        scores += 0.12*np.clip(np.abs(xs/w-0.5)*2,0,1)
    rx, ry = xs / max(w,1), ys / max(h,1)
    xpref = profile.get('preferred_x','any')
    ypref = profile.get('preferred_y','any')
    x_bonus = {'center':1.0-2.0*np.abs(rx-0.5),
               'left':1.0-rx, 'right':rx,
               'edge':2.0*np.abs(rx-0.5),
               'any':np.full_like(rx,0.5,dtype=float)}.get(xpref)
    y_bonus = {'top':1.0-ry,'middle':1.0-2.0*np.abs(ry-0.5),
               'bottom':ry,
               'any':np.full_like(ry,0.5,dtype=float)}.get(ypref)
    if x_bonus is not None and y_bonus is not None:
        scores = 0.80*scores + 0.13*x_bonus + 0.07*y_bonus
    if anchor:
        ax1, ay1, ax2, ay2 = anchor['bbox']
        ax = (ax1 + ax2) * w / (2 * W)
        ay = (ay2 if surface in ('floor', 'table') else (ay1 + ay2) * 0.5) * h / H
        proximity = np.exp(-np.hypot(xs - ax, ys - ay) / max(1.0, 0.30 * np.hypot(w, h)))
        scores = 0.65 * scores + 0.35 * proximity

    # Spatial constraints take precedence over the old nearest-anchor preference.
    # Learning from COCO (if enough samples) contributes at most 10% of this term.
    scores = scores + 0.42 * np.clip(rel_bonus, 0, 2.0)

    # V8.1 TV: do not leave a correctly scaled screen high in a corner just
    # because that wall patch scored well on mask coverage. Prefer the room's
    # entertainment zone; all collision/surface/obstacle hard rules still apply.
    if profile.get('group_name') == 'TV' and surface == 'wall':
        cabinet = [o for o in detections if o.get('label') in ('Cabinet', 'Shelf')
                   and o.get('bbox') and o['bbox'][2]-o['bbox'][0] >= 0.25*W
                   and o['bbox'][1] >= 0.36*H]
        if cabinet:
            anchor_tv=max(cabinet,key=lambda o: (o['bbox'][2]-o['bbox'][0]) *
                          float(o.get('confidence',0.6)))
            preferred_cx=0.5*(anchor_tv['bbox'][0]+anchor_tv['bbox'][2])*w/W
            # A wall-mounted screen sits above the cabinet, not inside it.
            preferred_cy=np.clip((anchor_tv['bbox'][1]/H - 0.14)*h,0.31*h,0.61*h)
        else:
            preferred_cx=0.5*w
            preferred_cy=0.50*h
        scores += (0.60*np.exp(-np.abs(xs-preferred_cx)/max(1.,0.18*w)) +
                   0.34*np.exp(-np.abs(ys-preferred_cy)/max(1.,0.16*h)))

    # V8.2: a television should be placed at a comfortable height within its
    # OWN wall span (ceiling edge to floor edge), not image center or ceiling.
    if profile.get('group_name') == 'TV' and surface == 'wall':
        position = _tv_wall_height_metrics(strict_masks, (w,h), x1,y1,x2,y2)
        gap = position['floor_gap_wall_fraction']
        preferred = TV_WALL_HEIGHT_RULE['preferred_floor_gap_wall_fraction']
        scores += np.where(position['valid_boundary'] & np.isfinite(gap),
                           1.20*np.exp(-np.abs(gap-preferred)/0.05),0.0)

    # Verify at native resolution; nearest resize may invent pixels at boundaries.
    orig_band = max(2, int(np.ceil(band * max(H / h, W / w))))
    orig_margin = max(1, int(np.ceil(margin * max(H / h, W / w))))
    # Support mask for table top should be verified against upper table mask, not all table pixels.
    original_support = ((_table_upper_support(np.asarray(masks.get('table',np.zeros_like(raw)),bool)) |
                         inferred_support) if surface == 'table' else raw)
    ranked = np.argsort(-scores)[:min(5000, scores.size)]
    for i in ranked:
        bx1 = max(0, int(np.floor(x1[i] * W / w)))
        bx2 = min(W, int(np.ceil(x2[i] * W / w)))
        by1 = max(0, int(np.floor(y1[i] * H / h)))
        by2 = min(H, int(np.ceil(y2[i] * H / h)))
        b = bx1, by1, bx2, by2
        if not _valid_support(original_support, surface, b, orig_band):
            continue
        if not bool(_strict_candidate_validity(
                profile,masks,(W,H),np.array([bx1]),np.array([by1]),
                np.array([bx2]),np.array([by2]))[0]):
            continue
        if _collision_with_yolo_excluding_support(b, detections, surface, margin=orig_margin):
            continue
        ac_relation=None
        if is_aircon:
            native_good,native_gap,native_coverage=_aircon_ceiling_geometry(
                masks['ceiling'],np.array([bx1]),np.array([by1]),
                np.array([bx2]),np.array([by2]),profile)
            if not bool(native_good[0]):
                continue
            ac_relation={'kind':'below_ceiling_on_wall',
                         'ceiling_gap_px':round(float(native_gap[0]*H),1),
                         'ceiling_gap_image_ratio':round(float(native_gap[0]),4),
                         'ceiling_column_coverage':round(float(native_coverage[0]),3),
                         'source':'Mask2Former ceiling boundary',
                         'verified':False}
        # Native pixel validation: don't trust a relation found only after downsampling.
        final_bonus,final_valid,final_matches,final_missing=spatial_relation_candidates(
            profile,detections,(W,H),
            np.array([bx1]),np.array([by1]),np.array([bx2]),np.array([by2]))
        if not bool(final_valid[0]):
            continue
        cx, cy = (bx1 + bx2) // 2, (by1 + by2) // 2
        sy = by1 if surface == 'ceiling' else (by2 if surface in ('floor','table') else cy)
        point = {'x':cx, 'y':sy}
        relation = _surface_relation_at(original_support, surface, (cx,sy))
        return {
            'status': 'suggested_unverified',
            'bbox_px': {'x1': bx1, 'y1': by1, 'x2': bx2, 'y2': by2},
            'bbox_norm': {'x1': round(bx1/W, 5), 'y1': round(by1/H, 5),
                          'x2': round(bx2/W, 5), 'y2': round(by2/H, 5)},
            'point_px': {'x': cx, 'y': cy},
            'point_norm': {'x': round(cx/W, 5), 'y': round(cy/H, 5)},
            'support_point_px': {'x': cx, 'y': sy},
            'surface': surface, 'anchor': anchor['label'] if anchor else None,
            'placement_score': round(float(scores[i]), 4), 'verified': False,
            'relative_to_surface': (relation['relative_to_surface'] if relation else 'Chưa định vị được trong vùng surface'),
            'surface_component_bbox_px': (relation['surface_component_bbox_px'] if relation else None),
            'surface_position': (relation['surface_position'] if relation else None),
            'surface_relation': (relation['surface_relation'] if relation else None),
            'surface_mask_pixels': int(raw.sum()),
            'surface_support_source': ('mask2former+YOLO_inferred_shelf_top' if inferred_support.any()
                                       else 'mask2former'),
            'spatial_relationships':final_matches,
            'ceiling_relation':ac_relation,
            'no_overlap_with_detected_objects':True,
            'missing_relation_anchors':final_missing,
            'relationship_bonus':round(float(final_bonus[0]),4),
            'spatial_learning':'disabled_api_only',
            'reason': 'Du doan 2D: bbox khong chong YOLO, tiep xuc mat ' + surface +
                      '; chua xac minh kich thuoc/vung 3D/lap dat',
        }
    return _unavailable_placement('no_safe_region',
                   'Ung vien khong vuot qua kiem tra mask/bbox tren anh goc', surface)

# ===== CELL 111 =====
def _attach_failure_context(placement, group, profile, masks):
    """Attach meaningful labels even when a product has NO valid bbox."""
    surface = (profile or {}).get('surface')
    placement['surface'] = surface
    placement['surface_mask_pixels'] = (int(np.asarray(masks[surface],bool).sum())
                                        if surface in masks else None)
    placement['assumed_profile'] = bool(group in {
        'Đèn điện/đèn sạc','Loa karaoke','Loa',
        'Khay/kệ đựng đồ','Quạt điện','Máy hút bụi'})
    if placement.get('bbox_px') is None:
        sn = SURFACE_NAMES_VI.get(surface, 'chưa xác định')
        status=placement.get('status')
        if status=='no_safe_region':
            placement['relative_to_surface'] = f'Chưa có vị trí hợp lệ trên {sn}; không gán bbox giả'
        elif status=='surface_not_allowed':
            placement['relative_to_surface'] = CEILING_REJECT_REASON
        elif status=='no_surface_detected':
            placement['relative_to_surface'] = f'Mask2Former không tìm thấy {sn}'
        elif status=='unsupported_product':
            placement['relative_to_surface'] = 'Chưa có metadata surface cho sản phẩm'
        else:
            placement['relative_to_surface'] = f'Chưa xác định trên {sn}'
    return placement


# ===== 10c.4 v4: SCALE BY OBJECT CLASS + LOCAL PERSPECTIVE =====
# Warning: without calibrated camera/depth, this is a relative 2D proposal.
# Avoid median(widths/heights) across unrelated YOLO categories.
KNOWN_REFERENCE_DIMENSIONS_CM={
    'Sofa':(200,85),'Chair':(52,88),'Armchair':(72,93),
    'TV':(110,65),'Television':(110,65),'Table':(110,55),
    'Dining table':(140,75),'Desk':(120,75),'Bed':(160,55),
    'Refrigerator':(70,170),'Door':(85,205),'Window':(120,125),
}

def _valid_yolo_reference_objects(detections):
    result=[]
    for d in detections:
        label=str(d.get('label',''))
        if label.startswith('Đề xuất:') or label not in KNOWN_REFERENCE_DIMENSIONS_CM:
            continue
        b=d.get('bbox')
        if not b or len(b)!=4:continue
        x1,y1,x2,y2=map(float,b)
        if not np.isfinite([x1,y1,x2,y2]).all() or x2-x1<6 or y2-y1<6:continue
        result.append(d)
    return result

# ===== CELL 113 =====
def _relative_closeness(depth_map, x, y, image_size):
    """Relative depth normalized by robust image percentiles; larger ~ closer.

    This interpretation follows Depth Anything V2's relative inverse-depth convention.
    It must NOT be used as metric distance or safety verification.
    """
    if depth_map is None:return None
    W,H=image_size
    d=np.asarray(depth_map)
    if d.shape!=(H,W):raise ValueError('depth_map shape mismatch')
    vals=d[np.isfinite(d)]
    if vals.size<20:return None
    low,high=np.percentile(vals,[5,95])
    if high-low<1e-6:return None
    ix=int(np.clip(round(x),0,W-1));iy=int(np.clip(round(y),0,H-1))
    y1=max(0,iy-2);y2=min(H,iy+3)
    x1=max(0,ix-2);x2=min(W,ix+3)
    local=d[y1:y2,x1:x2];local=local[np.isfinite(local)]
    if local.size==0:return None
    return float(np.clip((np.median(local)-low)/(high-low),0,1))

def _reference_pixel_per_cm(detections, image_size, surface, candidate_xy,
                            depth_map=None):
    W,H=image_size
    refs=_valid_yolo_reference_objects(detections)
    if not refs:return None,[],None
    candidate_closeness=_relative_closeness(depth_map,*candidate_xy,image_size)
    entries=[]
    for obj in refs:
        b=obj['bbox']; label=obj['label'];rw,rh=KNOWN_REFERENCE_DIMENSIONS_CM[label]
        w=(b[2]-b[0])/rw;h=(b[3]-b[1])/rh
        # Only use plausible dimensions, robust to TV/wall or furniture occlusion.
        if min(w,h)<=0:continue
        scale=float(np.sqrt(w*h))
        objx=(b[0]+b[2])/2;objy=b[3]
        ref_close=_relative_closeness(depth_map,objx,objy,image_size)
        if ref_close is not None and candidate_closeness is not None:
            relative=float(np.clip(1+0.80*(candidate_closeness-ref_close),0.62,1.55))
            depth_difference=abs(candidate_closeness-ref_close)
        elif surface in ('floor','table'):
            # Perspective fallback solely from image y, NOT real distance.
            relative=float(np.clip(1+0.70*(candidate_xy[1]-objy)/H,0.66,1.45))
            depth_difference=abs(candidate_xy[1]-objy)/H
        else:
            relative=1.0;depth_difference=0.5
        confidence=float(obj.get('confidence',0.5))
        weight=(0.1+confidence)/(0.1+depth_difference)
        entries.append({'label':label,'scale':scale*relative,'weight':weight,
                        'reference_bbox':b,'relative_factor':round(relative,3)})
    if not entries:return None,[],candidate_closeness
    # Weighted median among known-size instances, not indiscriminate median.
    entries.sort(key=lambda o:o['scale'])
    wt=np.array([o['weight'] for o in entries]);pos=np.searchsorted(np.cumsum(wt),wt.sum()/2)
    return float(entries[min(pos,len(entries)-1)]['scale']),entries,candidate_closeness

# ===== CELL 115 =====
def estimate_product_bbox_size(profile, detections, image_size,
                               depth_map=None, placement_point=None, surface=None):
    """Pixel size at a candidate point; preserves width:height from size_cm.

    Category-size priors are unverified. For unknown SKU size return a
    conservative image-ratio fallback with explicit review flag.
    """
    W,H=map(int,image_size)
    surf=surface or profile.get('surface') or 'floor'
    if placement_point is None:placement_point=(0.5*W,0.60*H)
    px_per_cm, references, closeness=_reference_pixel_per_cm(
        detections,(W,H),surf,placement_point,depth_map=depth_map)
    dims=profile.get('size_cm')
    use_cm=isinstance(dims,dict) and all(isinstance(dims.get(k),(int,float)) and dims[k]>0
                                             for k in ('width','height','depth'))
    if use_cm and px_per_cm is not None:
        estimated_w=px_per_cm*dims['width']
        estimated_h=px_per_cm*dims['height']
        method='known_category_reference_at_candidate_depth'
    else:
        # Explicit fallback, not a scale measurement.
        rw,rh=DEFAULT_REGION_RATIOS.get(surf,(0.10,0.15))
        estimated_w=W*rw*float(profile.get('relative_width',1))
        estimated_h=H*rh*float(profile.get('relative_height',1))
        if use_cm:
            # Preserve projected silhouette ratio as far as a generic prior allows.
            aspect=float(np.clip(dims['width']/dims['height'],0.18,6.0))
            prior_area=max(estimated_w*estimated_h,64)
            estimated_w=float(np.sqrt(prior_area*aspect))
            estimated_h=float(np.sqrt(prior_area/aspect))
        method='uncalibrated_image_fallback'
    # Visible dimensions can't be asserted to mm/cm without camera calibration.
    # Caps prevent enormous candidates but do not silently shrink them to fit space.
    bw=int(round(np.clip(estimated_w,max(7,0.010*W),0.58*W)))
    bh=int(round(np.clip(estimated_h,max(7,0.014*H),0.62*H)))
    return bw,bh,{'n_objects':len(detections),'method':method,
                  'reference_objects':references,'candidate_depth_closeness':closeness,
                  'size_source':profile.get('size_source','unknown'),
                  'size_cm':dims, 'mount_type':profile.get('mount_type'),
                  'candidate_point_px':{'x':int(placement_point[0]),'y':int(placement_point[1])},
                  'reference_scale_px_per_cm':px_per_cm}

# ===== CELL 117 =====
# V8.4 NOTE: wall-mounted TV/AC sizing is OVERRIDDEN in D.5.4; edit the V8.4 policy instead.
# D.3.1 — TV screen-size policies. These are 2D priors, not physical dimensions.
import re

TV_SCALE_POLICY = {
    'assumed_diagonal_inch': 55,
    'min_diagonal_inch': 24,
    'max_diagonal_inch': 100,
    'width_ratio_for_55inch': 0.32,
    'min_width_image_ratio': 0.22,
    'max_width_image_ratio': 0.54,
    'aspect_ratio': 16 / 9,
    'cabinet_min_width_image_ratio': 0.28,
    'cabinet_max_tv_width_fraction': 0.92,
    'min_alpha_silhouette_aspect_ratio': 1.30,
    'max_alpha_silhouette_aspect_ratio': 2.80,
    'min_rendered_fraction_of_placement_width': 0.82,
}


def tv_diagonal_inches(product_name):
    """Return (inches, source). Never claim that a model-code guess is verified."""
    name = str(product_name or '')
    # Explicit measurements are stronger evidence than product model numbers.
    labeled = re.search(r'(?<!\d)(\d{2,3})\s*(?:inch(?:es)?|in\b|[\"”])',
                        name, flags=re.I)
    if labeled:
        diagonal=int(labeled.group(1))
        if 24 <= diagonal <= 100:
            return diagonal, 'sku_name_explicit_inch'
    # Common TV model codes UA55U..., K-65S..., 55QNED..., 65C6KS.
    code = re.search(r'(?<!\d)(100|98|86|85|83|82|80|77|75|70|65|60|58|55|50|49|48|43|42|40|39|32|28|24)(?=[A-Z0-9])',
                     name.upper())
    if code:
        return int(code.group(1)), 'sku_name_model_code_guess'
    return int(TV_SCALE_POLICY['assumed_diagonal_inch']), 'category_default_unverified'


def tv_wall_console_reference(detections, image_size):
    """Only use confidently wide *low* cabinet/shelf boxes as optional scale anchors."""
    W,H=map(int,image_size)
    anchors=[]
    for obj in detections:
        if obj.get('label') not in ('Cabinet','Shelf'):continue
        bbox=obj.get('bbox')
        if not isinstance(bbox,(list,tuple)) or len(bbox)!=4:continue
        x1,y1,x2,y2=map(float,bbox)
        if (x2-x1) >= TV_SCALE_POLICY['cabinet_min_width_image_ratio']*W and y1 >= 0.36*H:
            anchors.append((float(obj.get('confidence',0.5))*(x2-x1),obj))
    return max(anchors,key=lambda a:a[0])[1] if anchors else None


def tv_screen_bbox_size(profile, detections, image_size, surface='wall'):
    """Propose a TV footprint independent of unrelated object px/cm estimates.

    One image cannot measure TV physical size. Keep the 16:9 footprint and
    return 'no_safe_region' through the existing placement search if it won't fit.
    """
    W,H=map(int,image_size)
    inch,source=tv_diagonal_inches(profile.get('product_name',''))
    scale=TV_SCALE_POLICY
    ratio=float(np.clip(
        scale['width_ratio_for_55inch']*(inch/55),
        scale['min_width_image_ratio'],scale['max_width_image_ratio']))
    anchor=tv_wall_console_reference(detections,(W,H))
    if surface=='table' and anchor is None:
        # A tabletop TV must have a real shelf/table support; the existing
        # table-surface rules decide whether such a placement is possible.
        candidates=[o for o in detections if o.get('label') in ('Cabinet','Shelf','Table','Desk','Side Table')
                    and isinstance(o.get('bbox'),(list,tuple)) and len(o['bbox'])==4]
        anchor=max(candidates,key=lambda o:o['bbox'][2]-o['bbox'][0],default=None)
    if anchor is not None:
        console_ratio=(float(anchor['bbox'][2])-float(anchor['bbox'][0]))/W
        cap=scale['cabinet_max_tv_width_fraction']*console_ratio
        # Never create an unrealistically tiny screen from a narrow cabinet.
        # If the console is insufficiently wide, placement still uses the scene
        # prior rather than trying to shrink an arbitrarily large SKU.
        if cap >= scale['min_width_image_ratio']:
            ratio=min(ratio,cap)
    bw=max(1,int(round(W*ratio)))
    bh=max(1,int(round(bw/scale['aspect_ratio'])))
    metadata={
        'method':'tv_wall_sku_diagonal_image_ratio_prior',
        'size_source':source,
        'diagonal_inch_estimate':inch,
        'tv_width_image_ratio':round(bw/W,4),
        'tv_screen_aspect_ratio':scale['aspect_ratio'],
        'cabinet_anchor':anchor.get('label') if anchor else None,
        'measurement_verified':False,
    }
    return bw,bh,metadata

# ===== CELL 118 =====
# D.3.2 — Route only TVs to the dedicated scale policy.
# Every other category still uses the original depth-conditioned estimator.
_estimate_product_bbox_size_before_tv=estimate_product_bbox_size


def estimate_product_bbox_size(profile, detections, image_size,
                               depth_map=None, placement_point=None, surface=None):
    surf=surface or profile.get('surface') or 'floor'
    if profile.get('group_name') != 'TV' or surf not in ('wall','table'):
        return _estimate_product_bbox_size_before_tv(
            profile,detections,image_size,depth_map=depth_map,
            placement_point=placement_point,surface=surface)
    width,height,metadata=tv_screen_bbox_size(profile,detections,image_size,surf)
    metadata.update({
        'candidate_point_px':{'x':int(placement_point[0]),'y':int(placement_point[1])}
            if placement_point is not None else None,
        'reference_objects':[],
        'reference_scale_px_per_cm':None,
        'candidate_depth_closeness':None,
        'size_cm':profile.get('size_cm'),
        'mount_type':profile.get('mount_type'),
        'n_objects':len(detections),
    })
    return width,height,metadata

# ===== CELL 120 =====
# V8.4 NOTE: wall-mounted TV/AC sizing is OVERRIDDEN in D.5.4; edit the V8.4 policy instead.
# ===== D.5.1 — V8.2: wall-space height, depth-conditioned TV and AC scale =====
# All distance fractions below are IMAGE-SPACE HEURISTICS. Depth Anything V2
# yields RELATIVE inverse depth, never meters. We do NOT invent physical distances.
TV_WALL_HEIGHT_RULE = {
    'floor_gap_min_wall_fraction': 0.080,   # 8% of detected wall height
    'floor_gap_max_wall_fraction': 0.200,   # 20% of detected wall height
    'preferred_floor_gap_wall_fraction': 0.150,  # target 15%; depth scale unchanged
    'ceiling_gap_min_wall_fraction': 0.12,   # sufficient wall above the TV
    'ceiling_gap_max_wall_fraction': 0.79,   # don't place TV near the floor
    'min_wall_height_image_fraction': 0.30,
    'minimum_valid_columns_fraction': 0.67,
}
DEPTH_SCALE_POLICY_V82 = {
    'tv_factor_bounds': (0.94, 1.08),   # preserve the TV size that already looks right
    'aircon_factor_bounds': (0.82, 1.24),
    'depth_scale_strength': 0.80,
    'aircon_reference_width_image_ratio': 0.215,  # ≈95cm category size prior
    'aircon_min_width_image_ratio': 0.17,
    'aircon_max_width_image_ratio': 0.32,
    'aircon_aspect_bounds': (2.20, 4.80),
    'aircon_min_rendered_footprint_fraction': 0.80,
}


def _tv_wall_height_metrics(masks, image_size, x1,y1,x2,y2):
    """Vectorized wall-height measurement below candidate TV, at its own x.

    Ceiling boundary = lowest ceiling pixel in upper 65% of image.
    Floor boundary   = first floor pixel in lower 70% of image.
    If either mask is missing or ambiguous, return invalid (no guessed seam).
    """
    W,H=map(int,image_size)
    xx1,yy1,xx2,yy2=np.broadcast_arrays(*[np.asarray(v,float)
                                          for v in (x1,y1,x2,y2)])
    nan=np.full(xx1.shape,np.nan)
    result={'valid_boundary':np.zeros(xx1.shape,bool),'valid':np.zeros(xx1.shape,bool),
            'floor_gap_wall_fraction':nan.copy(),'ceiling_gap_wall_fraction':nan.copy(),
            'wall_height_px':nan.copy(),'ceiling_y_px':nan.copy(),
            'floor_y_px':nan.copy(),'tv_center_wall_fraction':nan.copy(),
            'floor_gap_px':nan.copy(),'ceiling_gap_px':nan.copy()}
    if W<1 or H<1:return result
    ceiling=np.asarray(masks.get('ceiling',[]),bool)
    floor=np.asarray(masks.get('floor',[]),bool)
    wall=np.asarray(masks.get('wall',[]),bool)
    if any(a.shape != (H,W) for a in (ceiling,floor,wall)):
        return result
    if not ceiling.any() or not floor.any() or not wall.any():
        return result
    top_search=max(1,int(H*0.65))
    floor_start=int(H*0.30)
    yy=np.arange(top_search)[:,None]
    ceiling_edge=np.where(ceiling[:top_search],yy,-1).max(axis=0)
    fy=np.arange(floor_start,H)[:,None]
    floor_edge=np.where(floor[floor_start:],fy,H).min(axis=0)
    wall_height=floor_edge-ceiling_edge
    good_column=(ceiling_edge>=0)&(floor_edge<H)&(
        wall_height >= TV_WALL_HEIGHT_RULE['min_wall_height_image_fraction']*H)
    # Robust local edge read: choose three columns spread over the TV body.
    sample=np.stack([np.clip(np.rint(xx1+(xx2-xx1)*q).astype(int),0,W-1)
                     for q in (0.22,0.50,0.78)],axis=0)
    present=good_column[sample]
    count=present.sum(axis=0)
    enough=(count>=2)
    # Compute median via sorting finite values; no warnings for all-NaN candidates.
    ceil_values=np.sort(np.where(present,ceiling_edge[sample],np.inf),axis=0)
    floor_values=np.sort(np.where(present,floor_edge[sample],np.inf),axis=0)
    middle=np.maximum(count-1,0)//2
    ceil=np.take_along_axis(ceil_values,middle[None,...],axis=0)[0]
    grd=np.take_along_axis(floor_values,middle[None,...],axis=0)[0]
    ceil=np.where(enough,ceil,np.nan)
    grd=np.where(enough,grd,np.nan)
    height=grd-ceil
    gap_floor=grd-yy2
    gap_top=yy1-ceil
    f=gap_floor/np.maximum(height,1)
    t=gap_top/np.maximum(height,1)
    center=(0.5*(yy1+yy2)-ceil)/np.maximum(height,1)
    rule=TV_WALL_HEIGHT_RULE
    okay=(enough & (f >= rule['floor_gap_min_wall_fraction']) &
          (f <= rule['floor_gap_max_wall_fraction']) &
          (t >= rule['ceiling_gap_min_wall_fraction']) &
          (t <= rule['ceiling_gap_max_wall_fraction']) &
          (yy1>=ceil) & (yy2<=grd))
    result.update(valid_boundary=enough,valid=okay,
                  floor_gap_wall_fraction=f,ceiling_gap_wall_fraction=t,
                  wall_height_px=height,ceiling_y_px=ceil,floor_y_px=grd,
                  tv_center_wall_fraction=center,
                  floor_gap_px=gap_floor,ceiling_gap_px=gap_top)
    return result


def _depth_scale_from_wall(profile, depth_map, image_size, placement_point, product_group):
    """Depth-relative multiplier against THE SAME scene's wall mask.

    Reject when wall/depth cannot be trusted. Do not claim scale in cm.
    """
    W,H=map(int,image_size)
    data=np.asarray(depth_map,dtype=float) if depth_map is not None else np.asarray([])
    wall=np.asarray(profile.get('_wall_depth_mask',[]),bool)
    if data.shape!=(H,W) or wall.shape!=(H,W):
        raise ValueError('missing_depth_or_wall_mask_for_scale')
    valid=np.isfinite(data)
    wall_values=data[valid & wall]
    if wall_values.size<max(30,round(.003*W*H)):
        raise ValueError('insufficient_wall_depth_pixels')
    global_vals=data[valid]
    p05,p95=np.percentile(global_vals,[5,95])
    if not np.isfinite(p95-p05) or p95-p05<1e-8:
        raise ValueError('flat_depth_prediction')
    ref=float(np.median(wall_values))
    x,y=(W/2,H/2) if placement_point is None else placement_point
    ix=int(np.clip(round(x),0,W-1)); iy=int(np.clip(round(y),0,H-1))
    r=max(2,int(round(min(W,H)*0.012)))
    xa,xb=max(0,ix-r),min(W,ix+r+1)
    ya,yb=max(0,iy-r),min(H,iy+r+1)
    local=data[ya:yb,xa:xb]
    mask=valid[ya:yb,xa:xb] & wall[ya:yb,xa:xb]
    # Never read window or furniture depth as wall depth.
    if np.count_nonzero(mask)<max(3,round(0.20*mask.size)):
        raise ValueError('candidate_depth_not_on_wall')
    candidate=float(np.median(local[mask]))
    diff=float((candidate-ref)/(p95-p05))
    raw=float(np.exp(DEPTH_SCALE_POLICY_V82['depth_scale_strength']*np.clip(diff,-2.0,2.0)))
    lo,hi=DEPTH_SCALE_POLICY_V82['tv_factor_bounds' if product_group=='TV'
                                    else 'aircon_factor_bounds']
    factor=float(np.clip(raw,lo,hi))
    return factor,{'depth_method':'relative_inverse_depth_vs_wall_median',
                   'relative_wall_depth_delta':round(diff,4),
                   'depth_scale_factor':round(factor,4),
                   'depth_factor_clipped':bool(raw<lo or raw>hi),
                   'depth_is_metric':False}

# Wrap existing size estimator so only the TV/AC use dedicated depth scaling.
_estimate_product_bbox_size_pre_v82=estimate_product_bbox_size

def estimate_product_bbox_size(profile,detections,image_size,
                               depth_map=None,placement_point=None,surface=None):
    group=(profile or {}).get('group_name','')
    surf=surface or (profile or {}).get('surface','floor')
    if group not in ('TV','Máy điều hòa') or surf!='wall':
        return _estimate_product_bbox_size_pre_v82(
            profile,detections,image_size,depth_map=depth_map,
            placement_point=placement_point,surface=surface)
    W,H=map(int,image_size)
    factor,metrics=_depth_scale_from_wall(profile,depth_map,(W,H),placement_point,group)
    if group=='TV':
        bw,bh,base=tv_screen_bbox_size(profile,detections,(W,H),surf)
        # TV appearance is already reasonable: avoid a dramatic rescale.
        target_ratio=float(np.clip(bw/W*factor,
                                  TV_SCALE_POLICY['min_width_image_ratio'],
                                  TV_SCALE_POLICY['max_width_image_ratio']))
        bw=max(1,int(round(W*target_ratio)))
        bh=max(1,int(round(bw/TV_SCALE_POLICY['aspect_ratio'])))
        base['method']='tv_sku_plus_local_wall_relative_depth_v82'
    else:
        size=profile.get('size_cm') or {}
        width_cm=float(size.get('width',95) or 95)
        height_cm=float(size.get('height',31) or 31)
        # Use 95x31cm category prior unless a SKU dimension override exists.
        base_ratio=DEPTH_SCALE_POLICY_V82['aircon_reference_width_image_ratio']*(width_cm/95.0)
        target_ratio=float(np.clip(base_ratio*factor,
                      DEPTH_SCALE_POLICY_V82['aircon_min_width_image_ratio'],
                      DEPTH_SCALE_POLICY_V82['aircon_max_width_image_ratio']))
        bw=max(1,int(round(W*target_ratio)))
        aspect=float(np.clip(width_cm/max(height_cm,1),
                            *DEPTH_SCALE_POLICY_V82['aircon_aspect_bounds']))
        bh=max(1,int(round(bw/aspect)))
        base={'method':'aircon_size_prior_plus_local_wall_relative_depth_v82',
              'size_source':profile.get('size_source','category_prior_unverified'),
              'size_cm':size,'reference_width_ratio_before_depth':round(base_ratio,4),
              'projected_aspect':round(aspect,3), 'measurement_verified':False}
    base.update(metrics)
    base.update({'candidate_point_px':{'x':int(placement_point[0]),
                                       'y':int(placement_point[1])}
                if placement_point is not None else None,
                 'placed_width_image_ratio':round(bw/W,4),
                 'reference_objects':[],'reference_scale_px_per_cm':None,
                 'n_objects':len(detections), 'mount_type':profile.get('mount_type')})
    return bw,bh,base

# Extend hard checks: native and downsampled candidate boxes are both checked.
_strict_candidate_validity_pre_v82=_strict_candidate_validity

def _strict_candidate_validity(profile,masks,image_size,x1,y1,x2,y2):
    valid=_strict_candidate_validity_pre_v82(profile,masks,image_size,x1,y1,x2,y2)
    if (profile or {}).get('group_name')=='TV' and profile.get('surface')=='wall':
        geometry=_tv_wall_height_metrics(masks,image_size,x1,y1,x2,y2)
        valid &= geometry['valid']
    if (profile or {}).get('group_name')=='Máy điều hòa':
        xx1,yy1,xx2,yy2=np.broadcast_arrays(*[np.asarray(v,float)
                                              for v in (x1,y1,x2,y2)])
        W,H=map(int,image_size)
        width=(xx2-xx1)/max(1,W)
        policy=DEPTH_SCALE_POLICY_V82
        valid &= ((width>=policy['aircon_min_width_image_ratio']-0.013)&
                  (width<=policy['aircon_max_width_image_ratio']+0.013))
    return valid

# ===== CELL 122 =====
def choose_placement_region(profile,masks,detections,image_size,
                            max_processing_side=640,depth_map=None):
    """Depth-conditioned multi-band search with hard surface/collision checks.

    Exact physical fit, clearances in centimeters, and structural installation
    safety cannot be verified from an uncalibrated single image.
    """
    if not profile:return _unavailable_placement('unsupported_product','Thiếu profile')
    if profile.get('placement_blocked_status') == 'surface_not_allowed':
        return _unavailable_placement('surface_not_allowed',
                    profile.get('placement_blocked_reason', CEILING_REJECT_REASON),
                    profile.get('blocked_surface', 'ceiling'))
    W,H=map(int,image_size)
    surfaces=profile.get('surface_options') or [profile.get('surface')]
    if _is_air_conditioner_profile(profile): surfaces=['wall']
    if 'ceiling' in surfaces and not _ceiling_profile_is_allowed(profile):
        surfaces=[s for s in surfaces if s != 'ceiling']
        if not surfaces:
            return _unavailable_placement('surface_not_allowed',
                                          CEILING_REJECT_REASON, 'ceiling')
    failures=[]; successes=[]
    bands={
        'floor':[(0.43,0.59),(0.59,0.75),(0.75,0.91),(0.91,0.985)],
        'table':[(0.20,0.38),(0.38,0.56),(0.56,0.75),(0.75,0.91)],
        'wall':[(0.11,0.27),(0.27,0.44),(0.44,0.62),(0.62,0.82)],
        'ceiling':[(0.03,0.17),(0.17,0.31),(0.31,0.47)],
    }
    for surf in surfaces:
        if surf not in ALLOWED_SURFACES:continue
        for band in bands[surf]:
            cy=(band[0]+band[1])*0.5*H
            try:
                bw,bh,scene=estimate_product_bbox_size(profile,detections,(W,H),
                          depth_map=depth_map,placement_point=(W/2,cy),surface=surf)
            except ValueError as exc:
                if profile.get('group_name') in ('TV','Máy điều hòa'):
                    failures.append((surf,band,'invalid_depth_at_candidate',str(exc)))
                    continue
                raise
            # Never silently shrink dramatically: incorrect physical size is worse than no box.
            p=deepcopy(profile)
            p.update({'surface':surf,'region_width_ratio':bw/W,
                      'region_height_ratio':bh/H,'candidate_y_range':band})
            suggestion=_choose_placement_region_fixed(
                p,masks,detections,(W,H),max_processing_side=max_processing_side)
            if not suggestion.get('bbox_px'):
                failures.append((surf,band,suggestion['status'],suggestion.get('reason')))
                continue
            b=suggestion['bbox_px'];support=suggestion['support_point_px']
            # Use local candidate depth to recheck sizing; if scene changes too much,
            # reject rather than displaying a falsely sized bbox.
            aw,ah=b['x2']-b['x1'],b['y2']-b['y1']
            try:
                sw,sh,_=estimate_product_bbox_size(profile,detections,(W,H),
                    depth_map=depth_map,placement_point=(support['x'],support['y']),surface=surf)
            except ValueError as exc:
                if profile.get('group_name') in ('TV','Máy điều hòa'):
                    failures.append((surf,band,'invalid_depth_at_selected_bbox',str(exc)))
                    continue
                raise
            local_error=max(abs(aw-sw)/max(sw,1),abs(ah-sh)/max(sh,1))
            # Significant depth discrepancy: retry next spatial band (no invented bbox).
            if local_error>0.55:
                failures.append((surf,band,'local_depth_size_mismatch',round(local_error,3)))
                continue
            scene.update({'placed_size_px':{'width':aw,'height':ah},
                         'estimated_size_px':{'width':bw,'height':bh},
                         'local_size_disagreement':round(local_error,3),
                         'detected_objects_count':len(detections),
                         'ratios_to_each_object':[]})
            for obj in detections:
                bb=obj.get('bbox');
                if not bb or len(bb)!=4:continue
                ww,hh=float(bb[2]-bb[0]),float(bb[3]-bb[1])
                if min(ww,hh)<=0:continue
                scene['ratios_to_each_object'].append({
                    'label':obj.get('label','?'),'object_width_px':round(ww,1),
                    'object_height_px':round(hh,1),
                    'bbox_width_over_object_width':round(aw/ww,3),
                    'bbox_height_over_object_height':round(ah/hh,3)})
            scene['source']=scene['method']
            scene['effective_width_vs_reference']=None
            scene['effective_height_vs_reference']=None
            suggestion['scale_vs_scene']=scene
            suggestion['profile_source']=profile.get('profile_source','catalog_group_rule')
            suggestion['needs_review']=profile.get('needs_review',True)
            suggestion['reason'] += '; perspective-aware 2D estimate (not a measured installation region)'
            # Small reward for high confidence, real object geometry refs.
            confidence=0.04 if scene['method'].startswith('known') else 0.0
            penalty=0.13*local_error
            quality=float(suggestion['placement_score'])+confidence-penalty
            successes.append((quality,suggestion))
    if successes:
        successes.sort(key=lambda x:x[0],reverse=True)
        return successes[0][1]
    chosen=next((f for f in failures if f[2]=='no_ceiling_satisfying_region'),
                 next((f for f in failures if f[2]=='no_ceiling_detected'),
                 next((f for f in failures if f[2]=='no_relation_satisfying_region'),
                next((f for f in failures if f[2]=='no_safe_region'),
                     failures[0] if failures else None))))
    surf,status,reason=(chosen[0],chosen[2],chosen[3]) if chosen else (profile.get('surface'),'unsupported_product','Surface không hợp lệ')
    p=_unavailable_placement(status,str(reason),surf)
    p['failed_candidates']=failures
    p['profile_source']=profile.get('profile_source','catalog_group_rule')
    p['needs_review']=profile.get('needs_review',True)
    return p

# ===== CELL 124 =====
# ===== 10c.4.10b: final hard guard + rule provenance =====
_choose_placement_region_v6=choose_placement_region

def choose_placement_region(profile,masks,detections,image_size,
                            max_processing_side=640,depth_map=None):
    placement=_choose_placement_region_v6(profile,masks,detections,image_size,
                                          max_processing_side=max_processing_side,
                                          depth_map=depth_map)
    if profile is None or profile.get('group_name') not in NINE_CATEGORY_RULEBOOK_V7:
        return placement
    placement['rulebook_version']=profile.get('rulebook_version','V7.1')
    placement['rule_variant']=profile.get('speaker_type',profile.get('mount_type'))
    placement['hard_rules']=list(profile.get('hard_rules_descriptions',[]))
    placement['soft_rules']=list(profile.get('soft_rules_descriptions',[]))
    placement['matched_rules']=[]
    placement['failed_rules']=[]
    box=placement.get('bbox_px')
    if box:
        W,H=map(int,image_size)
        bx=[int(box[k]) for k in ('x1','y1','x2','y2')]
        valid=_strict_candidate_validity(profile,masks,(W,H),
                    *[np.array([v]) for v in bx])
        if not bool(valid[0]) or _collision_with_yolo_excluding_support(
                bx,detections,profile.get('surface'),margin=0):
            placement=_unavailable_placement('blocked_by_scene_rules',
                        'V7 final check: kích thước/bề mặt/vật cản không hợp lệ',
                        profile.get('surface'))
            placement['rulebook_version']='V7.1'
            placement['failed_rules']=['kiểm tra lại Hard Rules ở ảnh gốc']
        else:
            placement['matched_rules']=['surface', 'silhouette',
                'scene_obstacles', 'detected_object_non_overlap']
            for r in placement.get('spatial_relationships',[]):
                if r.get('satisfied') is True:
                    placement['matched_rules'].append(r.get('name','object_relation'))
    else:
        placement['failed_rules']=[placement.get('reason','no_safe_region')]
    placement['placement_status']=placement.get('status')
    return placement

# ===== CELL 126 =====
# ===== D.5.2 — Require valid depth + wall/floor/ceiling masks for TV/AC =====
_choose_placement_region_pre_v82=choose_placement_region

def choose_placement_region(profile,masks,detections,image_size,
                            max_processing_side=640,depth_map=None):
    group=(profile or {}).get('group_name','')
    if group in ('TV','Máy điều hòa') and (profile or {}).get('surface')=='wall':
        W,H=map(int,image_size)
        raw_wall=np.asarray(masks.get('wall',[]),bool)
        raw_depth=np.asarray(depth_map) if depth_map is not None else np.asarray([])
        if raw_wall.shape!=(H,W) or not raw_wall.any():
            return _unavailable_placement('no_wall_depth_reference','Thiếu wall mask để định scale theo độ sâu','wall')
        if raw_depth.shape!=(H,W) or not np.isfinite(raw_depth).any():
            return _unavailable_placement('no_depth_for_scale','Thiếu depth hợp lệ; không ước lượng TV/máy lạnh','wall')
        if group=='TV':
            floor_ref=np.asarray(masks.get('floor',[]),bool)
            if floor_ref.shape!=(H,W) or not floor_ref.any():
                return _unavailable_placement('no_floor_boundary',
                        'Thiếu mask sàn; không thể định vị chiều cao TV theo sàn','wall')
            # Ceiling is OPTIONAL: D.5.6 uses the visible front-wall top + floor seam
            # when Mask2Former sees no ceiling. Never invent a ceiling mask.
        local_profile=deepcopy(profile)
        local_profile['_wall_depth_mask']=raw_wall
        try:
            placement=_choose_placement_region_pre_v82(
                local_profile,masks,detections,image_size,
                max_processing_side=max_processing_side,depth_map=depth_map)
        except ValueError as exc:
            if any(tag in str(exc) for tag in ('missing_depth_or_wall_mask_for_scale',
                          'insufficient_wall_depth_pixels','flat_depth_prediction',
                          'candidate_depth_not_on_wall')):
                return _unavailable_placement('invalid_depth_for_scale',str(exc),'wall')
            raise
        if placement.get('bbox_px'):
            if group=='TV':
                b=placement['bbox_px']; geom=_tv_wall_height_metrics(masks,(W,H),
                    [b['x1']],[b['y1']],[b['x2']],[b['y2']])
                if not bool(geom['valid'][0]):
                    return _unavailable_placement('invalid_tv_wall_height',
                        'TV không đạt khoảng cách tới sàn/trần','wall')
                placement['tv_wall_height']={k:round(float(geom[k][0]),4)
                    for k in ('floor_gap_px','ceiling_gap_px','floor_gap_wall_fraction',
                              'ceiling_gap_wall_fraction','wall_height_px',
                              'tv_center_wall_fraction')}
                placement['tv_wall_height']['measurement_verified']=False
            else:
                placement['aircon_depth_scale']=placement.get('scale_vs_scene',{})
        return placement
    return _choose_placement_region_pre_v82(
        profile,masks,detections,image_size,
        max_processing_side=max_processing_side,depth_map=depth_map)


def validate_aircon_api_silhouette(planned_bbox,rendered_bbox,image_size):
    """Fail closed when the API cutout would make an AC tiny or portrait.

    Do not stretch the object, use the raw product image or substitute another SKU.
    """
    W,H=map(int,image_size)
    pbw=int(planned_bbox['x2'])-int(planned_bbox['x1'])
    width=int(rendered_bbox['x2'])-int(rendered_bbox['x1'])
    height=int(rendered_bbox['y2'])-int(rendered_bbox['y1'])
    ratio=width/max(height,1)
    p=DEPTH_SCALE_POLICY_V82
    if not (p['aircon_aspect_bounds'][0] <= ratio <= p['aircon_aspect_bounds'][1]):
        raise ValueError(f'Ảnh máy lạnh sau API có tỷ lệ sai: {ratio:.2f}')
    if width < p['aircon_min_rendered_footprint_fraction']*pbw:
        raise ValueError(f'Ảnh máy lạnh sau API quá nhỏ: {width}px so với bbox {pbw}px')
    if width < (p['aircon_min_width_image_ratio']*0.80)*W:
        raise ValueError('Máy lạnh quá nhỏ so với độ rộng ảnh sau khi tách nền')
    return {'rendered_width_px':width,'rendered_height_px':height,
            'rendered_aspect_ratio':round(ratio,3),
            'rendered_width_image_ratio':round(width/max(W,1),4)}

# ===== CELL 128 =====
# ===== D.5.3: Hard Rule -> mot bbox treo tuong chi thuoc MOT mat tuong =====
_wall_plane_previous_candidate_check = _strict_candidate_validity

def _wall_plane_allowed_names():
    return (['wall_plane_front','wall_plane_left','wall_plane_right']
            if WALL_PLANE_POLICY_V83['allow_side_wall_mount']
            else ['wall_plane_front'])


def _wall_plane_box_check(masks,image_size,bbox,threshold=None):
    W,H=map(int,image_size)
    x1,y1,x2,y2=[int(round(v)) for v in bbox]
    area=(x2-x1)*(y2-y1)
    if area<=0 or x1<0 or y1<0 or x2>W or y2>H:
        return False,{'reason':'bbox_outside_image'}
    threshold=(WALL_PLANE_POLICY_V83['bbox_same_plane_min_fraction']
               if threshold is None else float(threshold))
    junction=np.asarray(masks.get('wall_junction_forbidden',[]),bool)
    if junction.shape!=(H,W):return False,{'reason':'wall_plane_unavailable'}
    if np.any(junction[y1:y2,x1:x2]):
        return False,{'reason':'bbox_intersects_wall_junction'}
    for name in _wall_plane_allowed_names():
        plane=np.asarray(masks.get(name,[]),bool)
        if plane.shape!=(H,W):continue
        coverage=float(plane[y1:y2,x1:x2].sum())/area
        if coverage>=threshold:
            return True,{'plane_id':name,'plane_coverage':round(coverage,5),
                         'corner_margin_px':max(WALL_PLANE_POLICY_V83['corner_band_min_px'],
                         round(WALL_PLANE_POLICY_V83['corner_band_image_fraction']*W)),
                         'method':'per_plane_full_bbox_support'}
    return False,{'reason':'bbox_not_contained_in_one_wall_plane'}


def _strict_candidate_validity(profile,masks,image_size,x1,y1,x2,y2):
    base=_wall_plane_previous_candidate_check(profile,masks,image_size,x1,y1,x2,y2)
    if profile is None or profile.get('surface')!='wall':return base
    W,H=map(int,image_size)
    xx1,yy1,xx2,yy2=np.broadcast_arrays(*[np.asarray(z,dtype=float)
                                          for z in (x1,y1,x2,y2)])
    xa=np.clip(np.floor(xx1).astype(int),0,W)
    ya=np.clip(np.floor(yy1).astype(int),0,H)
    xb=np.clip(np.ceil(xx2).astype(int),0,W)
    yb=np.clip(np.ceil(yy2).astype(int),0,H)
    area=np.maximum(1,(xb-xa)*(yb-ya))
    junction=np.asarray(masks.get('wall_junction_forbidden',[]),bool)
    if junction.shape!=(H,W):return base & False
    j_sum=_rect_counts(_integral(junction),xa,ya,xb,yb)
    valid=(j_sum==0)
    accepted=np.zeros(xa.shape,dtype=bool)
    for name in _wall_plane_allowed_names():
        plane=np.asarray(masks.get(name,[]),bool)
        if plane.shape!=(H,W):continue
        count=_rect_counts(_integral(plane),xa,ya,xb,yb)
        accepted |= count/area>=WALL_PLANE_POLICY_V83['bbox_same_plane_min_fraction']
    return base & valid & accepted & (xb>xa) & (yb>ya)


_choose_placement_region_pre_wall_planes = choose_placement_region

def choose_placement_region(profile,masks,detections,image_size,
                            max_processing_side=640,depth_map=None):
    result=_choose_placement_region_pre_wall_planes(
        profile,masks,detections,image_size,
        max_processing_side=max_processing_side,depth_map=depth_map)
    if not result.get('bbox_px') or result.get('surface')!='wall':
        return result
    box=result['bbox_px']
    check,diagnostic=_wall_plane_box_check(masks,image_size,
                    [box[k] for k in ('x1','y1','x2','y2')])
    if not check:
        blocked=_unavailable_placement('wall_plane_rejected',diagnostic['reason'],'wall')
        blocked['failed_rules']=['wall_one_plane_only','no_wall_intersection']
        return blocked
    result['wall_plane']=diagnostic
    result.setdefault('matched_rules',[]).append('wall_one_plane_only')
    return result

# ===== CELL 130 =====
STATUS_NAMES_VI.update({'overlap_rejected':'Trùng vật thể/vùng đã đặt', 'no_safe_region':'Không có vùng an toàn', 'no_ceiling_satisfying_region':'Máy lạnh không sát dưới trần', 'blocked_by_scene_rules':'Không đạt rule bề mặt/vật cản'})

# ===== CELL 132 =====
# D.5.4a - V8.5: size calibration config (edit here).
# Relative monocular depth DOES NOT provide meters. Never claim measured size.
TV_AC_WALL_SCALE_V84 = {
    'measured_front_wall_width_cm': None,  # Set the real measurement when known.
    'assumed_front_wall_width_cm': 350.0, # Scene prior ONLY; unverified.
    'tv_wall_width_fraction_bounds': (0.13, 0.56),
    'aircon_wall_width_fraction_bounds': (0.15, 0.43),
    'tv_relative_depth_factor_bounds': (0.90, 1.10),
    'aircon_relative_depth_factor_bounds': (0.88, 1.17),
    'relative_depth_strength': 0.45,
    'minimum_front_wall_width_image_fraction': 0.23,
    'min_wall_depth_pixels': 35,
    'min_reliable_rows': 12,
    'wall_width_quantile': 0.78,
    'debug_scale': False,
}
TV_SCALE_POLICY['min_width_image_ratio'] = 0.055  # Alpha validation, not size prior.
DEPTH_SCALE_POLICY_V82['aircon_min_width_image_ratio'] = 0.065
DEPTH_SCALE_POLICY_V82['aircon_max_width_image_ratio'] = 0.36
_V84_MASK_CONTEXT = {}

# ===== CELL 134 =====
# D.5.4b - V8.5: infer the width of ONE front wall consistently for all SKUs.
def _v85_front_wall_geometry(masks, image_size):
    """Estimate one stable front-wall horizontal span from several wall rows.

    Never derive its scale from a single row that may have segmentation holes.
    Reject noisy or incomplete wall planes; never merge separate wall planes.
    """
    W,H=map(int,image_size)
    cached=masks.get('_front_wall_geometry_v85')
    if isinstance(cached,dict) and cached.get('_shape')==(H,W):
        return cached
    front=np.asarray(masks.get('wall_plane_front',[]),dtype=bool)
    if front.shape!=(H,W) or not front.any():
        raise ValueError('front_wall_plane_unavailable')
    occupied_rows=np.flatnonzero(front.sum(axis=1)>=max(20,round(W*.16)))
    if len(occupied_rows)<TV_AC_WALL_SCALE_V84['min_reliable_rows']:
        raise ValueError('insufficient_front_wall_rows')
    # Exclude the ceiling/floor seams and inspect multiple interior heights.
    low,high=np.quantile(occupied_rows,[.17,.83])
    row_ids=np.unique(np.rint(np.linspace(low,high,33)).astype(int))
    lefts=[];rights=[];widths=[]
    for y in row_ids:
        # 5-row voting removes isolated holes without making a different plane.
        ya=max(0,y-2);yb=min(H,y+3)
        hits=front[ya:yb].sum(axis=0)
        xx=np.flatnonzero(hits>=max(1,int(np.ceil(.40*(yb-ya)))))
        if len(xx)<max(20,round(W*.15)):
            continue
        xleft=int(np.quantile(xx,.01))
        xright=int(np.quantile(xx,.99))+1
        if xright<=xleft:
            continue
        lefts.append(xleft);rights.append(xright);widths.append(xright-xleft)
    if len(widths)<TV_AC_WALL_SCALE_V84['min_reliable_rows']:
        raise ValueError('insufficient_reliable_wall_spans')
    median_width=float(np.median(widths))
    wall_span=float(np.quantile(widths,TV_AC_WALL_SCALE_V84['wall_width_quantile']))
    if wall_span<TV_AC_WALL_SCALE_V84['minimum_front_wall_width_image_fraction']*W:
        raise ValueError('front_wall_span_too_narrow')
    if np.percentile(widths,90)>1.85*max(1,np.percentile(widths,10)):
        raise ValueError('front_wall_geometry_inconsistent_across_rows')
    # Use robust front-wall edges; these are NOT the overall image edges.
    left=int(np.quantile(lefts,.35))
    right=int(np.quantile(rights,.65))
    span=int(round(max(wall_span,median_width,right-left)))
    span=min(span,W)
    result={
        '_shape':(H,W),
        'front_wall_span_px':span,
        'front_wall_x_extent':(left,right),
        'front_wall_span_median_px':round(median_width,1),
        'front_wall_width_fraction_of_image':round(span/max(W,1),4),
        'wall_rows_used':len(widths),
        'wall_span_method':'multi_row_front_wall_plane_quantile',
        'reliability':'estimated_2d_not_metric',
    }
    masks['_front_wall_geometry_v85']=result
    return result


def _v84_wall_span(masks, image_size, y):
    """Backward-compatible geometry access used by the AC placement scanner."""
    g=_v85_front_wall_geometry(masks,image_size)
    return g['front_wall_x_extent']


def _v84_depth_stats(depth_map,plane,image_size):
    W,H=map(int,image_size)
    arr=np.asarray(depth_map,dtype=float) if depth_map is not None else np.asarray([])
    if arr.shape!=(H,W):raise ValueError('missing_relative_depth')
    mask=np.asarray(plane,bool)
    if mask.shape!=(H,W):raise ValueError('front_wall_depth_mask_unavailable')
    finite=np.isfinite(arr)
    valid=arr[finite]
    wall=arr[finite & mask]
    if valid.size<100 or wall.size<TV_AC_WALL_SCALE_V84['min_wall_depth_pixels']:
        raise ValueError('insufficient_front_wall_depth_samples')
    lo,hi=np.percentile(valid,[5,95])
    if hi-lo<1e-8:raise ValueError('constant_relative_depth_prediction')
    return {'range':float(hi-lo),'reference':float(np.median(wall))}


def _v84_depth_factor(depth_map,plane,image_size,point,group,stats=None):
    W,H=map(int,image_size)
    data=np.asarray(depth_map,dtype=float)
    wall=np.asarray(plane,bool)
    if stats is None:stats=_v84_depth_stats(data,wall,(W,H))
    x,y=map(lambda z:int(round(z)),point)
    x=int(np.clip(x,0,W-1));y=int(np.clip(y,0,H-1))
    radius=max(3,int(round(min(W,H)*.015)))
    xa,xb=max(0,x-radius),min(W,x+radius+1)
    ya,yb=max(0,y-radius),min(H,y+radius+1)
    crop=data[ya:yb,xa:xb]
    usable=np.isfinite(crop)&wall[ya:yb,xa:xb]
    if usable.sum()<max(5,int(round(.15*crop.size))):
        raise ValueError('depth_not_on_front_wall_at_candidate')
    local=float(np.median(crop[usable]))
    delta=(local-stats['reference'])/stats['range']
    bounds=(TV_AC_WALL_SCALE_V84['tv_relative_depth_factor_bounds'] if group=='TV'
            else TV_AC_WALL_SCALE_V84['aircon_relative_depth_factor_bounds'])
    raw=float(np.exp(TV_AC_WALL_SCALE_V84['relative_depth_strength']*
                     np.clip(delta,-1.5,1.5)))
    return float(np.clip(raw,*bounds)),round(float(delta),5)

# ===== CELL 136 =====
# D.5.4c - V8.5: SKU physical dimensions -> front wall pixels -> relative depth.
def _v84_size_by_wall_width(profile,image_size,depth_map,point,group,masks,stats=None):
    W,H=map(int,image_size)
    geometry=_v85_front_wall_geometry(masks,(W,H))
    span=geometry['front_wall_span_px']
    plane=np.asarray(masks['wall_plane_front'],bool)
    factor,delta=_v84_depth_factor(depth_map,plane,(W,H),point,group,stats)
    calibrated=TV_AC_WALL_SCALE_V84['measured_front_wall_width_cm']
    wall_cm=float(calibrated if calibrated is not None else
                  TV_AC_WALL_SCALE_V84['assumed_front_wall_width_cm'])
    if not 120<=wall_cm<=1500:raise ValueError('invalid_front_wall_width_cm')
    if group=='TV':
        inch,source=tv_diagonal_inches(profile.get('product_name',''))
        width_cm=float(inch)*2.54*16.0/((16.0**2+9.0**2)**0.5)
        aspect=16.0/9.0
        bounds=TV_AC_WALL_SCALE_V84['tv_wall_width_fraction_bounds']
    else:
        sku=profile.get('size_cm') or {}
        width_cm=float(sku.get('width',95) or 95)
        height_cm=float(sku.get('height',31) or 31)
        aspect=float(np.clip(width_cm/max(height_cm,1),
                  *DEPTH_SCALE_POLICY_V82['aircon_aspect_bounds']))
        source=profile.get('size_source','category_size_unverified')
        bounds=TV_AC_WALL_SCALE_V84['aircon_wall_width_fraction_bounds']
    if not 15<=width_cm<=320:raise ValueError('unrealistic_product_width_cm')
    raw_ratio=width_cm/wall_cm*factor
    # Clipping is a safeguard, not the means of making a small TV into a big SKU.
    ratio=float(np.clip(raw_ratio,*bounds))
    bw=max(1,int(round(span*ratio)))
    bh=max(1,int(round(bw/aspect)))
    meta={
        'method':'v85_multiline_front_wall_sku_depth_relative',
        **geometry,
        'size_source':source,'measurement_verified':bool(calibrated is not None),
        'wall_width_cm_used':wall_cm,
        'wall_width_calibrated':bool(calibrated is not None),
        'product_width_cm_estimate':round(width_cm,2),
        'raw_wall_fraction':round(raw_ratio,4),
        'ratio_clipped_by_safety':bool(abs(ratio-raw_ratio)>1e-8),
        'depth_is_metric':False,'relative_depth_delta':delta,
        'relative_depth_factor':round(factor,4),
        'wall_width_fraction':round(bw/max(span,1),4),
        'image_width_fraction':round(bw/max(W,1),4),
        'product_aspect':round(aspect,3),
        'size_not_physically_verified':bool(calibrated is None),
        'estimated_size_px':{'width':bw,'height':bh},
    }
    if group=='TV':
        meta.update({'diagonal_inch_estimate':int(inch),
                    'sku_small_screen_below_43_inch':bool(inch<43)})
    if TV_AC_WALL_SCALE_V84['debug_scale']:
        print(f'[SCALE V8.5] {group} | wall={span}px / {wall_cm:g}cm'
              f' | SKU={width_cm:.1f}cm | depth_factor={factor:.3f}'
              f' | result={bw}x{bh}px | verified={calibrated is not None}')
    return bw,bh,meta

_v84_size_before=estimate_product_bbox_size

def estimate_product_bbox_size(profile,detections,image_size,
                               depth_map=None,placement_point=None,surface=None):
    group=(profile or {}).get('group_name','')
    surf=surface or (profile or {}).get('surface','floor')
    if group not in ('TV','M\u00e1y \u0111i\u1ec1u h\u00f2a') or surf!='wall':
        return _v84_size_before(profile,detections,image_size,
                   depth_map=depth_map,placement_point=placement_point,surface=surface)
    masks=_V84_MASK_CONTEXT.get((profile or {}).get('_v84_mask_context_key'))
    if not isinstance(masks,dict):
        raise ValueError('v85_surface_masks_not_attached')
    W,H=map(int,image_size)
    point=placement_point if placement_point is not None else (W/2,H*.39)
    return _v84_size_by_wall_width(profile,(W,H),depth_map,point,group,masks)

# ===== CELL 138 =====
# D.5.5 - V8.4: AC independent guided wall search, measured from ceiling AND floor.
# TV can obstruct the placement rectangle, but TV position NEVER determines AC height.
AIRCON_WALL_CLEARANCE_V84 = {
    'top_gap_min_wall_fraction': 0.035,
    'top_gap_preferred_wall_fraction': 0.075,
    'top_gap_max_wall_fraction': 0.155,
    'bottom_gap_min_wall_fraction': 0.43,
    'wall_plane_coverage_min': 0.975,
    'ceiling_floor_required': True,
    'max_scan_candidates': 240,
    'debug_failure_counts': True,
}


def _v84_wall_ceiling_floor_edges(masks, image_size, x):
    """Use the actual *front-wall segment* (its top/bottom) in one column.

    Mask2Former must provide wall, ceiling, floor. The wall segment supplies
    local ceiling-wall and wall-floor seams; no image-global y thresholds.
    """
    W,H=map(int,image_size)
    plane=np.asarray(masks.get('wall_plane_front',[]),bool)
    ce=np.asarray(masks.get('ceiling',[]),bool)
    fl=np.asarray(masks.get('floor',[]),bool)
    if any(m.shape!=(H,W) for m in (plane,ce,fl)):
        raise ValueError('missing_surface_masks_for_ac_clearance')
    if AIRCON_WALL_CLEARANCE_V84['ceiling_floor_required'] and (not ce.any() or not fl.any()):
        raise ValueError('missing_ceiling_or_floor_mask')
    cx=int(np.clip(round(x),0,W-1))
    rad=max(3,int(round(W*.008)))
    strip=plane[:,max(0,cx-rad):min(W,cx+rad+1)]
    # Robust against single-pixel holes around the upper/lower wall boundary.
    counts=strip.sum(axis=1)
    ys=np.flatnonzero(counts>=max(2,int(round(strip.shape[1]*.42))))
    if ys.size<max(30,int(.16*H)):
        raise ValueError('incomplete_wall_column_at_ac_candidate')
    # Avoid isolated segmentation pixels at edges: robust 1st/99th percentile.
    top=int(np.quantile(ys,.008));bottom=int(np.quantile(ys,.992))
    if bottom-top < .27*H:
        raise ValueError('unreliable_wall_height_at_ac_candidate')
    return top,bottom


def _v84_aircon_wall_clearance(masks,image_size,bbox):
    W,H=map(int,image_size)
    x1,y1,x2,y2=map(int,bbox)
    cx=(x1+x2)/2
    try:
        top,bottom=_v84_wall_ceiling_floor_edges(masks,(W,H),cx)
    except ValueError as exc:
        return False,{'reason':str(exc)}
    height=max(1,bottom-top)
    gap_top=(y1-top)/height
    gap_bottom=(bottom-y2)/height
    policy=AIRCON_WALL_CLEARANCE_V84
    good=(policy['top_gap_min_wall_fraction'] <= gap_top <=
          policy['top_gap_max_wall_fraction'] and
          gap_bottom >= policy['bottom_gap_min_wall_fraction'] and
          y1>=top and y2<=bottom)
    return bool(good),{
        'ceiling_wall_y_px':int(top), 'wall_floor_y_px':int(bottom),
        'wall_height_px':int(height),
        'top_gap_wall_fraction':round(float(gap_top),4),
        'bottom_gap_wall_fraction':round(float(gap_bottom),4),
        'top_gap_px':int(y1-top),'bottom_gap_px':int(bottom-y2),
        'source':'Mask2Former/front_wall_plane_top_bottom',
        'measurement_verified':False,
    }


def _v84_find_aircon_placement(profile,masks,detections,image_size,depth_map=None):
    """Search upper front wall at wall-relative ceiling/surface bands.

    This bypasses the older image-global vertical bands and avoids mandatory
    relation to TV/Sofa. Keeps forbidden regions and non-overlap hard checks.
    """
    W,H=map(int,image_size)
    p=deepcopy(profile)
    p['surface']='wall'
    p['surface_options']=['wall']
    plane=np.asarray(masks.get('wall_plane_front',[]),bool)
    if plane.shape!=(H,W) or not plane.any():
        return _unavailable_placement('no_front_wall_plane','No reliable front wall plane','wall')
    try:
        stats=_v84_depth_stats(depth_map,plane,(W,H))
        xlo,xhi=_v84_wall_span(masks,(W,H),H*.34)
    except ValueError as exc:
        return _unavailable_placement('insufficient_wall_geometry_or_depth',str(exc),'wall')
    max_n=int(AIRCON_WALL_CLEARANCE_V84['max_scan_candidates'])
    xmargin=max(10,round((xhi-xlo)*.04))
    xs=np.linspace(xlo+xmargin,xhi-xmargin,35)
    heights=[.075,.055,.100,.130,.040,.150]
    counts={}; ranked=[]; examined=0
    def failed(reason):counts[reason]=counts.get(reason,0)+1
    for cx in xs:
        try:
            top,bottom=_v84_wall_ceiling_floor_edges(masks,(W,H),cx)
        except ValueError as exc:
            failed(str(exc));continue
        wall_height=bottom-top
        for frac in heights:
            if examined>=max_n:break
            examined+=1
            yy=top+frac*wall_height
            try:
                bw,bh,scene=_v84_size_by_wall_width(
                    p,(W,H),depth_map,(cx,yy), 'Máy điều hòa',masks,stats=stats)
            except ValueError as exc:
                failed(str(exc));continue
            x1=int(round(cx-bw/2));x2=x1+bw
            y1=int(round(top+frac*wall_height));y2=y1+bh
            if min(x1,y1)<0 or x2>W or y2>H:
                failed('out_of_frame');continue
            b=(x1,y1,x2,y2)
            good,clearance=_v84_aircon_wall_clearance(masks,(W,H),b)
            if not good:failed('outside_ceiling_floor_clearance');continue
            # One physical wall plane; never cross detected wall junctions.
            plane_ok,diag=_wall_plane_box_check(masks,(W,H),b,
                threshold=AIRCON_WALL_CLEARANCE_V84['wall_plane_coverage_min'])
            if not plane_ok:failed(diag.get('reason','crosses_wall_corner'));continue
            # Existing V6/V7 hard geometry: windows, doors, edge margins, ratio.
            geometry=_wall_plane_previous_candidate_check(p,masks,(W,H),
                            *[np.array([u]) for u in b])
            if not bool(geometry[0]):
                failed('semantic_obstacle_or_wrong_aspect');continue
            # Wall semantic support must be overwhelming, not an empty patch.
            raw_wall=np.asarray(masks['wall'],bool)
            if raw_wall[y1:y2,x1:x2].mean() < .965:
                failed('insufficient_real_wall_support');continue
            ceiling=np.asarray(masks['ceiling'],bool)
            if np.mean(ceiling[y1:y2,x1:x2])>.005:
                failed('overlaps_ceiling');continue
            if _collision_with_yolo_excluding_support(b,detections,'wall',margin=0):
                failed('overlap_detected_or_reserved');continue
            # No position or scale relationship with TV; only actual 2D collisions.
            preferred=AIRCON_WALL_CLEARANCE_V84['top_gap_preferred_wall_fraction']
            score=(2.0-5.0*abs(frac-preferred))
            score+=.35*(1.0-abs(cx-(xlo+xhi)/2)/max(1,(xhi-xlo)/2))
            score+=.12*(1.0-abs(scene['relative_depth_delta']))
            ranked.append((score,b,scene,clearance,diag))
        if examined>=max_n:break
    if not ranked:
        out=_unavailable_placement('no_safe_ac_wall_region',
            'No front-wall bbox satisfies ceiling/floor clearance, geometry and obstacles','wall')
        out['failed_candidates']=sorted(counts.items(),key=lambda z:-z[1])
        out['candidates_scanned']=examined
        return out
    ranked.sort(key=lambda it:it[0],reverse=True)
    score,b,scene,clearance,diag=ranked[0]
    x1,y1,x2,y2=b;cx=(x1+x2)//2;cy=(y1+y2)//2
    scene.update({'placed_size_px':{'width':x2-x1,'height':y2-y1},
                  'estimated_size_px':{'width':x2-x1,'height':y2-y1}})
    return {'status':'suggested_unverified',
        'bbox_px':dict(x1=x1,y1=y1,x2=x2,y2=y2),
        'bbox_norm':dict(x1=round(x1/W,5),y1=round(y1/H,5),
                         x2=round(x2/W,5),y2=round(y2/H,5)),
        'point_px':{'x':cx,'y':cy},
        'support_point_px':{'x':cx,'y':cy},
        'surface':'wall','anchor':None,
        'placement_score':round(float(score),4),'verified':False,
        'scale_vs_scene':scene,'ceiling_relation':clearance,
        'aircon_wall_clearance':clearance,'wall_plane':diag,
        'matched_rules':['front_wall_only','ceiling_floor_clearance',
                         'non_overlap','sku_depth_relative_scale'],
        'failed_rules':[],'placement_status':'suggested_unverified',
        'rulebook_version':'V8.4',
        'searched_candidates':examined,
        'failed_candidate_counts':sorted(counts.items(),key=lambda z:-z[1]),
        'reason':'Front-wall geometry: wall-relative ceiling/floor clearance; no TV anchor'}

# Wrapper: TV goes through existing V8.3 placement with wall span sizing;
# AC gets its own search with wall-relative ceiling/floor geometry.
_v84_choose_before=choose_placement_region

def choose_placement_region(profile,masks,detections,image_size,
                            max_processing_side=640,depth_map=None):
    if not profile:return _unavailable_placement('unsupported_product','Missing product profile')
    group=profile.get('group_name')
    if group=='Máy điều hòa':
        return _v84_find_aircon_placement(profile,masks,detections,image_size,depth_map)
    if group=='TV' and profile.get('surface')=='wall':
        p=deepcopy(profile)
        _V84_MASK_CONTEXT.clear()
        _V84_MASK_CONTEXT[id(masks)]=masks
        p['_v84_mask_context_key']=id(masks)
        return _v84_choose_before(p,masks,detections,image_size,
                                  max_processing_side=max_processing_side,
                                  depth_map=depth_map)
    return _v84_choose_before(profile,masks,detections,image_size,
                              max_processing_side=max_processing_side,
                              depth_map=depth_map)

# ===== CELL 140 =====
# V8.6 - Floor-only wall-height reference (no ceiling visible)
# Geometry fallback ONLY; the background-removal API still has NO fallback.
FLOOR_ONLY_WALL_POLICY_V86 = {
    'minimum_floor_columns_fraction': 0.32,
    'minimum_wall_height_image_fraction': 0.26,
    'minimum_wall_coverage_in_column': 0.14,
    'upper_visible_wall_gap_fraction': 0.025,
    'ac_floor_gap_min_wall_fraction': 0.52,
    'ac_floor_gap_preferred_wall_fraction': 0.73,
    'ac_floor_gap_max_wall_fraction': 0.86,
    'ac_candidate_floor_gap_options': (0.73, 0.68, 0.78, 0.63, 0.83, 0.57),
    'ac_min_wall_mask_coverage': 0.965,
    'max_scan_candidates': 240,
}
_V86_FLOOR_WALL_CACHE = {}


def _v86_has_visible_ceiling(masks, image_size):
    """A tiny stray ceiling pixel is not evidence of a usable ceiling plane."""
    W,H=map(int,image_size)
    raw=np.asarray(masks.get('ceiling',[]),dtype=bool)
    if raw.shape!=(H,W):return False
    upper=raw[:max(1,int(.67*H))]
    return bool(upper.sum() >= max(16,int(round(.002*W*H))))


def _v86_floor_wall_columns(masks,image_size):
    """Derive local floor seam + *visible* wall-top from existing masks.

    The wall-top is an image crop/semantic boundary, NOT a measured ceiling.
    Require real floor evidence; no synthetic ceiling or floor is fabricated.
    """
    W,H=map(int,image_size)
    front=np.asarray(masks.get('wall_plane_front',[]),bool)
    floor=np.asarray(masks.get('floor',[]),bool)
    if front.shape!=(H,W) or floor.shape!=(H,W) or not front.any() or not floor.any():
        raise ValueError('no_front_wall_or_floor_mask')
    key=(id(front),id(floor),W,H)
    cache=_V86_FLOOR_WALL_CACHE.get(key)
    if (cache is not None and cache.get('_wall_ref') is front and
        cache.get('_floor_ref') is floor):return cache
    # Only seek the floor in the lower part; ignore isolated upper false positives.
    lower=floor.copy()
    lower[:int(round(.30*H))]=False
    floor_columns=lower.sum(axis=0)>=max(5,int(round(.020*H)))
    if floor_columns.sum()<max(8,int(round(FLOOR_ONLY_WALL_POLICY_V86['minimum_floor_columns_fraction']*W))):
        raise ValueError('floor_boundary_not_visible_across_wall')
    first_floor=np.where(floor_columns,np.argmax(lower,axis=0),H).astype(int)
    wall_columns=front.sum(axis=0)>=max(6,int(round(
        FLOOR_ONLY_WALL_POLICY_V86['minimum_wall_coverage_in_column']*H)))
    first_wall=np.where(wall_columns,np.argmax(front,axis=0),H).astype(int)
    # Smooth the seam locally using column medians; never fill unsupported columns.
    from scipy.ndimage import median_filter
    known=np.flatnonzero(floor_columns)
    estimate=np.interp(np.arange(W),known,first_floor[known])
    seam=np.rint(median_filter(estimate,size=max(3,int(round(.025*W))|1),mode='nearest')).astype(int)
    reliable=floor_columns & wall_columns & (np.abs(first_floor-seam)<=.07*H)
    reliable &= (seam-first_wall)>=FLOOR_ONLY_WALL_POLICY_V86['minimum_wall_height_image_fraction']*H
    reliable &= seam < H
    if reliable.sum()<max(8,int(round(.16*W))):
        raise ValueError('floor_wall_reference_not_reliable')
    result={'top_y':first_wall,'floor_y':seam,'valid_columns':reliable,
            'reference_mode':'floor_and_visible_wall_top_no_ceiling',
            'metric_height_verified':False,'_wall_ref':front,'_floor_ref':floor}
    if len(_V86_FLOOR_WALL_CACHE)>3:_V86_FLOOR_WALL_CACHE.clear()
    _V86_FLOOR_WALL_CACHE[key]=result
    return result


def display_floor_wall_reference_v86(image,masks,output_path=None):
    """Debug: green=visible wall top; orange=detected floor seam (not ceiling)."""
    from PIL import Image,ImageDraw
    src=image.convert('RGB').copy() if isinstance(image,Image.Image) else Image.fromarray(np.asarray(image)).convert('RGB')
    W,H=src.size
    info=_v86_floor_wall_columns(masks,(W,H))
    draw=ImageDraw.Draw(src)
    valid=info['valid_columns']
    for x in range(W-1):
        if valid[x] and valid[x+1]:
            draw.line([(x,int(info['top_y'][x])),(x+1,int(info['top_y'][x+1]))],fill=(24,175,90),width=3)
            draw.line([(x,int(info['floor_y'][x])),(x+1,int(info['floor_y'][x+1]))],fill=(245,139,42),width=3)
    if output_path is not None:src.save(output_path)
    return src

# ===== CELL 142 =====
# V8.6 - A TV can use the floor seam when no ceiling is visible.
_tv_wall_height_metrics_v85=_tv_wall_height_metrics


def _tv_wall_height_metrics(masks,image_size,x1,y1,x2,y2):
    W,H=map(int,image_size)
    if _v86_has_visible_ceiling(masks,(W,H)):
        return _tv_wall_height_metrics_v85(masks,(W,H),x1,y1,x2,y2)
    xx1,yy1,xx2,yy2=np.broadcast_arrays(*[
        np.asarray(v,dtype=float) for v in (x1,y1,x2,y2)])
    nan=np.full(xx1.shape,np.nan,dtype=float)
    out={'valid_boundary':np.zeros(xx1.shape,bool),'valid':np.zeros(xx1.shape,bool),
         'floor_gap_wall_fraction':nan.copy(),'ceiling_gap_wall_fraction':nan.copy(),
         'wall_height_px':nan.copy(),'ceiling_y_px':nan.copy(),
         'floor_y_px':nan.copy(),'tv_center_wall_fraction':nan.copy(),
         'floor_gap_px':nan.copy(),'ceiling_gap_px':nan.copy(),
         'reference_mode':'floor_and_visible_wall_top_no_ceiling',
         'ceiling_detected':False}
    try:ref=_v86_floor_wall_columns(masks,(W,H))
    except ValueError:return out
    sample=np.stack([np.clip(np.rint(xx1+(xx2-xx1)*q).astype(int),0,W-1)
                     for q in (0.20,0.50,0.80)],axis=0)
    present=ref['valid_columns'][sample]
    count=present.sum(axis=0)
    enough=count>=2
    take=np.maximum(count-1,0)//2
    top_vals=np.sort(np.where(present,ref['top_y'][sample],np.inf),axis=0)
    low_vals=np.sort(np.where(present,ref['floor_y'][sample],np.inf),axis=0)
    top=np.take_along_axis(top_vals,take[None,...],axis=0)[0]
    bottom=np.take_along_axis(low_vals,take[None,...],axis=0)[0]
    top=np.where(enough,top,np.nan)
    bottom=np.where(enough,bottom,np.nan)
    wall_height=bottom-top
    floor_gap=bottom-yy2
    upper_gap=yy1-top
    fraction=floor_gap/np.maximum(1,wall_height)
    upper_fraction=upper_gap/np.maximum(1,wall_height)
    rule=TV_WALL_HEIGHT_RULE
    okay=(enough & (fraction>=rule['floor_gap_min_wall_fraction']) &
          (fraction<=rule['floor_gap_max_wall_fraction']) &
          (upper_fraction>=rule['ceiling_gap_min_wall_fraction']) &
          (upper_fraction<=rule['ceiling_gap_max_wall_fraction']) &
          (yy1>=top) & (yy2<=bottom))
    out.update(valid_boundary=enough,valid=okay,
               floor_gap_wall_fraction=fraction,
               # Legacy key refers to the visible wall top, NOT a ceiling.
               ceiling_gap_wall_fraction=upper_fraction,
               wall_height_px=wall_height,ceiling_y_px=top,floor_y_px=bottom,
               tv_center_wall_fraction=(.5*(yy1+yy2)-top)/np.maximum(1,wall_height),
               floor_gap_px=floor_gap,ceiling_gap_px=upper_gap)
    return out

# ===== CELL 144 =====
# V8.6 - Air conditioner placement anchored to floor if ceiling is absent.
_v84_aircon_wall_clearance_v85=_v84_aircon_wall_clearance
_v84_find_aircon_placement_v85=_v84_find_aircon_placement


def _v84_aircon_wall_clearance(masks,image_size,bbox):
    W,H=map(int,image_size)
    if _v86_has_visible_ceiling(masks,(W,H)):
        good,meta=_v84_aircon_wall_clearance_v85(masks,(W,H),bbox)
        meta['reference_mode']='ceiling_wall_floor_detected'
        meta['ceiling_detected']=True
        return good,meta
    x1,y1,x2,y2=map(int,bbox)
    cx=int(round(.5*(x1+x2)))
    try:
        ref=_v86_floor_wall_columns(masks,(W,H))
        if not ref['valid_columns'][int(np.clip(cx,0,W-1))]:
            raise ValueError('floor_seam_missing_at_aircon')
        top=int(ref['top_y'][cx]);bottom=int(ref['floor_y'][cx])
    except ValueError as exc:
        return False,{'reason':str(exc),'reference_mode':'floor_only_no_ceiling'}
    height=max(1,bottom-top)
    gap_floor=(bottom-y2)/height
    gap_upper=(y1-top)/height
    p=FLOOR_ONLY_WALL_POLICY_V86
    good=(p['ac_floor_gap_min_wall_fraction']<=gap_floor<=
          p['ac_floor_gap_max_wall_fraction'] and
          gap_upper>=p['upper_visible_wall_gap_fraction'] and
          y1>=top and y2<=bottom)
    return bool(good),{
        'reference_mode':'floor_and_visible_wall_top_no_ceiling',
        'ceiling_detected':False,
        'ceiling_wall_y_px':None,'wall_floor_y_px':bottom,
        'visible_wall_top_y_px':top,'wall_height_px':height,
        'floor_gap_wall_fraction':round(float(gap_floor),5),
        'upper_visible_wall_gap_fraction':round(float(gap_upper),5),
        'floor_gap_px':int(bottom-y2),'upper_visible_wall_gap_px':int(y1-top),
        'measurement_verified':False,
        'note':'Height is relative to the visible wall and floor. Ceiling not inferred.'}


def _v86_find_ac_using_floor(profile,masks,detections,image_size,depth_map=None):
    W,H=map(int,image_size)
    p=deepcopy(profile)
    p['surface']='wall';p['surface_options']=['wall']
    plane=np.asarray(masks.get('wall_plane_front',[]),bool)
    if plane.shape!=(H,W) or not plane.any():
        return _unavailable_placement('no_front_wall_plane','Cannot identify a front wall','wall')
    try:
        floor_ref=_v86_floor_wall_columns(masks,(W,H))
        geometry=_v85_front_wall_geometry(masks,(W,H))
        stats=_v84_depth_stats(depth_map,plane,(W,H))
    except ValueError as exc:
        return _unavailable_placement('no_floor_wall_depth_reference',str(exc),'wall')
    left,right=geometry['front_wall_x_extent']
    margin=max(9,int(round((right-left)*.045)))
    xs=np.linspace(left+margin,right-margin,33)
    options=FLOOR_ONLY_WALL_POLICY_V86['ac_candidate_floor_gap_options']
    max_n=FLOOR_ONLY_WALL_POLICY_V86['max_scan_candidates']
    scanned=0;reasons={};ranked=[]
    def reject(msg):reasons[msg]=reasons.get(msg,0)+1
    for x in xs:
        ix=int(np.clip(round(x),0,W-1))
        if not floor_ref['valid_columns'][ix]:reject('floor_seam_unreliable');continue
        top=int(floor_ref['top_y'][ix]);floor_y=int(floor_ref['floor_y'][ix])
        hwall=floor_y-top
        for floor_frac in options:
            if scanned>=max_n:break
            scanned+=1
            # The product's BOTTOM is positioned using floor distance only.
            target_y2=floor_y-floor_frac*hwall
            try:
                bw,bh,scene=_v84_size_by_wall_width(
                    p,(W,H),depth_map,(x,target_y2),
                    'Máy điều hòa',masks,stats=stats)
            except ValueError as exc:reject(str(exc));continue
            x1=int(round(x-bw*.5));x2=x1+bw
            y2=int(round(target_y2));y1=y2-bh
            b=x1,y1,x2,y2
            if x1<0 or y1<0 or x2>W or y2>H:reject('outside_image');continue
            ok,clearance=_v84_aircon_wall_clearance(masks,(W,H),b)
            if not ok:reject('floor_clearance_invalid');continue
            plane_ok,plane_diag=_wall_plane_box_check(
                masks,(W,H),b,threshold=AIRCON_WALL_CLEARANCE_V84['wall_plane_coverage_min'])
            if not plane_ok:reject(plane_diag.get('reason','crosses_wall_plane'));continue
            raw_wall=np.asarray(masks.get('wall',[]),bool)
            if raw_wall.shape!=(H,W) or raw_wall[y1:y2,x1:x2].mean()<FLOOR_ONLY_WALL_POLICY_V86['ac_min_wall_mask_coverage']:
                reject('insufficient_real_wall_support');continue
            # Reuse original obstacle, proportion, and image-edge checks.
            # Do not reuse V8.2's image-wide minimum AC size, since the scale
            # now comes from this wall plane and its physical-width estimate.
            valid=_strict_candidate_validity_v6(p,masks,(W,H),
                np.array([x1]),np.array([y1]),np.array([x2]),np.array([y2]))
            if not bool(valid[0]):reject('obstacle_aspect_or_image_edge');continue
            lo,hi=p.get('aspect_ratio_bounds',(1.,12.))
            if not lo <= (x2-x1)/max(1,y2-y1) <= hi:
                reject('sku_aspect_constraint');continue
            if _collision_with_yolo_excluding_support(b,detections,'wall',margin=0):
                reject('overlap_detected_or_reserved');continue
            pref=FLOOR_ONLY_WALL_POLICY_V86['ac_floor_gap_preferred_wall_fraction']
            score=2.0-4.5*abs(floor_frac-pref)
            score+=.25*(1-abs(x-(left+right)*.5)/max((right-left)*.5,1))
            score+=.1*(1-min(1,abs(scene['relative_depth_delta'])))
            ranked.append((score,b,scene,clearance,plane_diag))
        if scanned>=max_n:break
    if not ranked:
        out=_unavailable_placement('no_safe_ac_floor_anchored_region',
            'Floor visible but no front-wall bbox meets geometry and collision constraints','wall')
        out['candidates_scanned']=scanned
        out['failed_candidate_counts']=sorted(reasons.items(),key=lambda z:-z[1])
        return out
    ranked.sort(key=lambda z:z[0],reverse=True)
    score,b,scene,clearance,plane_diag=ranked[0]
    x1,y1,x2,y2=b
    scene['placed_size_px']={'width':x2-x1,'height':y2-y1}
    scene['estimated_size_px']={'width':x2-x1,'height':y2-y1}
    return {'status':'suggested_unverified',
        'bbox_px':dict(x1=x1,y1=y1,x2=x2,y2=y2),
        'bbox_norm':dict(x1=round(x1/W,5),y1=round(y1/H,5),
                         x2=round(x2/W,5),y2=round(y2/H,5)),
        'point_px':{'x':(x1+x2)//2,'y':(y1+y2)//2},
        'support_point_px':{'x':(x1+x2)//2,'y':(y1+y2)//2},
        'surface':'wall','anchor':None,'verified':False,
        'placement_score':round(float(score),4),
        'scale_vs_scene':scene,'wall_plane':plane_diag,
        'aircon_wall_clearance':clearance,'ceiling_relation':None,
        'height_reference_mode':'floor_and_visible_wall_top_no_ceiling',
        'matched_rules':['front_wall_only','floor_anchored_no_ceiling',
                          'non_overlap','sku_depth_relative_scale'],
        'failed_rules':[],'placement_status':'suggested_unverified',
        'rulebook_version':'V8.6','searched_candidates':scanned,
        'reason':'Floor-only geometry; no ceiling assumed or invented.'}


def _v84_find_aircon_placement(profile,masks,detections,image_size,depth_map=None):
    if _v86_has_visible_ceiling(masks,image_size):
        return _v84_find_aircon_placement_v85(profile,masks,detections,image_size,depth_map)
    return _v86_find_ac_using_floor(profile,masks,detections,image_size,depth_map)


# V8.5 API accepted-size check still used a global image-width AC minimum,
# which contradicts V8.5's single-wall sizing in narrow-wall photographs.
def validate_aircon_api_silhouette(planned_bbox,rendered_bbox,image_size):
    pw=int(planned_bbox['x2'])-int(planned_bbox['x1'])
    rw=int(rendered_bbox['x2'])-int(rendered_bbox['x1'])
    rh=int(rendered_bbox['y2'])-int(rendered_bbox['y1'])
    aspect=rw/max(1,rh)
    lo,hi=DEPTH_SCALE_POLICY_V82['aircon_aspect_bounds']
    if not (lo<=aspect<=hi):
        raise ValueError(f'AC PNG silhouette aspect ratio invalid: {aspect:.2f}')
    if rw < .80*pw or rh<6:
        raise ValueError(f'AC API silhouette too small: {rw}px vs planned {pw}px')
    return {'rendered_width_px':rw,'rendered_height_px':rh,
            'rendered_aspect_ratio':round(aspect,3),
            'rendered_width_vs_planned':round(rw/max(pw,1),3),
            'sized_against':'wall_plane_not_full_image'}

# ===== CELL 147 =====
# D.5.7 — V8.11: CHỈ TV - Wall-first placement (independent of AC)
# Mục tiêu: chọn mặt tường trước, rồi mới tính bbox. Mask2Former chỉ cung cấp
# semantic "wall", không cung cấp instance tường. Các mặt tường ở đây là ước lượng
# từ mask + đường giao RGB/geometry, KHÔNG khẳng định đã đo 3D chính xác.
from collections import Counter
from scipy.ndimage import binary_closing, distance_transform_edt
import numpy as np
import cv2
from PIL import Image, ImageDraw

TV_WALL_FIRST_POLICY_V811 = {
    'enabled': True,
    'allow_side_wall': True,               # sửa hạn chế V8.3 chỉ cho wall_plane_front
    'minimum_plane_area_fraction': 0.035,
    'minimum_plane_width_fraction': 0.16,
    'min_bbox_plane_coverage': 0.93,     # mask semantic có thể bị thủng bởi bóng/đồ
    'max_forbidden_overlap_fraction': 0.005,
    'corner_band_image_fraction': 0.028,
    'corner_band_min_px': 10,
    'unknown_corner_edge_fraction': 0.06,
    'visual_corner_min_score': 21.0,
    'preferred_floor_gap_fraction': 0.15,
    'floor_gap_options': (0.15, 0.13, 0.17, 0.10, 0.19, 0.08, 0.20),
    'min_floor_gap_fraction': 0.08,
    'max_floor_gap_fraction': 0.20,
    'min_top_gap_fraction': 0.08,
    'tv_wall_width_floor_ratio': 0.13,   # ngưỡng theo MẶT TƯỜNG, không theo toàn ảnh
    'tv_wall_width_cap_ratio': 0.57,
    'horizontal_grid_image_fraction': 0.009,
    'min_free_corner_distance_px': 4,
    'debug': True,
}


def v811_prepare_tv_planes(image, masks):
    """Dựng từng mặt tường ứng viên từ WALL semantic và bằng chứng góc thật.

    Chỉ bổ sung khóa tv_*; KHÔNG thay đổi wall_plane_* cho các sản phẩm khác.
    Nếu RGB không tìm ra góc, chỉ dùng các mặt tường được phân tách bằng hình học
    hoặc phần trong của mask wall với trạng thái uncertain, KHÔNG bịa góc.
    """
    img=np.asarray(image.convert('RGB') if isinstance(image,Image.Image) else image)
    H,W=img.shape[:2]
    raw=np.asarray(masks.get('wall',[]),dtype=bool)
    if raw.shape!=(H,W) or not raw.any():
        masks['tv_plane_meta_v811']={'status':'no_wall_mask','planes':[]}
        return masks['tv_plane_meta_v811']
    # Chỉ lấp các lỗ lẻ nhỏ 3x3 do noise semantic; không nối hai tường qua góc.
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    cleaned=cv2.morphologyEx(raw.astype('uint8'),cv2.MORPH_CLOSE,kernel).astype(bool)
    visual={}
    for side in ('left','right'):
        try: e=_v87_detect_visual_corner(image,masks,side)
        except Exception: e=None
        if e and float(e.get('score',0)) >= TV_WALL_FIRST_POLICY_V811['visual_corner_min_score']:
            visual[side]=float(e['x'])
    # Nếu không có bằng chứng RGB, dùng góc từ V8.3/V8.7 đã đánh dấu chắc chắn.
    geo={}
    old_names=('wall_plane_left','wall_plane_right')
    for side, name in zip(('left','right'),old_names):
        if side in visual: continue
        old=np.asarray(masks.get(name,[]),dtype=bool)
        front=np.asarray(masks.get('wall_plane_front',[]),dtype=bool)
        if old.shape==(H,W) and front.shape==(H,W) and old.sum()>.025*W*H:
            xs=np.where(old)[1]
            if len(xs)>100:
                geo[side]=float(np.percentile(xs,97 if side=='left' else 3))
    corners={**geo,**visual}
    # Ranh giới giao tường được nới rộng; bbox không bao giờ vượt qua dải này.
    band=max(int(TV_WALL_FIRST_POLICY_V811['corner_band_min_px']),
             int(round(W*TV_WALL_FIRST_POLICY_V811['corner_band_image_fraction'])))
    xx=np.arange(W)[None,:]
    barred=np.zeros((H,W),dtype=bool)
    for x in corners.values():
        barred |= np.broadcast_to(np.abs(xx-x)<=band,(H,W))
    planes={}
    if 'left' in corners:
        planes['tv_plane_left']=cleaned & (xx<corners['left']-band)
    if 'right' in corners:
        planes['tv_plane_right']=cleaned & (xx>corners['right']+band)
    lo=corners.get('left',None);hi=corners.get('right',None)
    if lo is None and hi is None:
        # Không thấy đường giao: không khẳng định front đã được nhận diện.
        # Chọn vùng an toàn ở trong phần nhìn thấy của wall semantic.
        inset=int(W*TV_WALL_FIRST_POLICY_V811['unknown_corner_edge_fraction'])
        inner=np.zeros((H,W),bool);inner[:,inset:max(inset+1,W-inset)]=True
        planes['tv_plane_unresolved']=cleaned & inner
    else:
        if lo is not None and hi is not None and lo+band>=hi-band:
            planes['tv_plane_unresolved']=np.zeros_like(cleaned)
        else:
            left=max(0,int(lo+band)) if lo is not None else 0
            right=min(W,int(hi-band)) if hi is not None else W
            inner=np.zeros((H,W),bool)
            if right>left:inner[:,left:right]=True
            planes['tv_plane_front']=cleaned & inner
    # Loại trừ cửa sổ/cửa/cầu thang khỏi nơi đặt TV, kể cả khi Mask2Former
    # xếp nhầm cửa sổ vào mask WALL.
    obstacles=np.zeros((H,W),bool)
    for name in ('window','door','stairs','fireplace','light_fixture'):
        arr=np.asarray(masks.get(name,[]),bool)
        if arr.shape==(H,W):obstacles |= arr
    planes={k:v for k,v in planes.items()
            if np.count_nonzero(v) >= TV_WALL_FIRST_POLICY_V811['minimum_plane_area_fraction']*W*H
            and np.count_nonzero(v.sum(axis=0)>H*.12) >= W*TV_WALL_FIRST_POLICY_V811['minimum_plane_width_fraction']}
    if not TV_WALL_FIRST_POLICY_V811['allow_side_wall']:
        planes={k:v for k,v in planes.items() if k in ('tv_plane_front','tv_plane_unresolved')}
    for k in list(masks):
        if k.startswith('tv_plane_') and k!='tv_plane_meta_v811': del masks[k]
    masks.update(planes)
    masks['tv_corner_forbidden']=barred
    masks['tv_forbidden_obstacles']=obstacles
    meta={'status':'ok' if planes else 'no_usable_wall_plane',
          'planes':list(planes),'plane_areas':{k:int(np.sum(v)) for k,v in planes.items()},
          'corners_px':{k:round(v,1) for k,v in corners.items()},
          'corner_evidence':{k:('rgb' if k in visual else 'segmentation_geometry') for k in corners},
          'confidence':'visual_or_geometric_split' if corners else 'wall_mask_only_uncertain',
          'corner_band_px':band,'not_metric_3d':True}
    masks['tv_plane_meta_v811']=meta
    return meta


def v811_tv_plane_box_check(masks,image_size,bbox,plane_id=None):
    """Xác minh TOÀN BỘ bbox trên đúng một mặt; cho phép noise mask nhỏ.

    Góc tường, cửa sổ, cửa đi là hard reject. Không dùng giới hạn cũ
    99.5% và không dùng chiều rộng tối thiểu 22% của TOÀN ẢNH.
    """
    W,H=map(int,image_size)
    b=[int(round(v)) for v in bbox]
    x1,y1,x2,y2=b
    if x2<=x1 or y2<=y1 or min(x1,y1)<0 or x2>W or y2>H:
        return False,{'reason':'bbox_outside_image'}
    bar=np.asarray(masks.get('tv_corner_forbidden',[]),bool)
    if bar.shape==(H,W) and bar[y1:y2,x1:x2].any():
        return False,{'reason':'crosses_wall_wall_junction'}
    forbidden=np.asarray(masks.get('tv_forbidden_obstacles',[]),bool)
    if forbidden.shape==(H,W):
        forbidden_ratio=float(forbidden[y1:y2,x1:x2].mean())
        if forbidden_ratio>TV_WALL_FIRST_POLICY_V811['max_forbidden_overlap_fraction']:
            return False,{'reason':'intersects_window_door_or_fixture','obstacle_fraction':round(forbidden_ratio,3)}
    names=([plane_id] if plane_id is not None else
           list(masks.get('tv_plane_meta_v811',{}).get('planes',[])))
    for name in names:
        plane=np.asarray(masks.get(name,[]),bool)
        if plane.shape!=(H,W):continue
        coverage=float(plane[y1:y2,x1:x2].mean())
        if coverage>=TV_WALL_FIRST_POLICY_V811['min_bbox_plane_coverage']:
            # Kiểm tra cả bốn mép, tránh bbox chỉ có tâm nằm trên wall.
            pad=max(1,int(min(x2-x1,y2-y1)*.055))
            edges=[plane[y1:y1+pad,x1:x2],plane[y2-pad:y2,x1:x2],
                   plane[y1:y2,x1:x1+pad],plane[y1:y2,x2-pad:x2]]
            if min(float(a.mean()) for a in edges)<0.80:continue
            return True,{'plane_id':name,'plane_coverage':round(coverage,4),
                         'method':'single_semantic_plane_with_corner_exclusion',
                         'surface_geometry_verified':False}
    return False,{'reason':'not_contained_in_one_tv_plane','candidate_planes':names}


def v811_tv_vertical_reference(masks,image_size,x,plane_id):
    """Ranh giới trên/dưới của MẶT TƯỜNG tại x, ưu tiên sàn nếu quan sát được.

    Trong ảnh không thấy trần, wall_top chỉ là cạnh trên phần tường nhìn thấy.
    """
    W,H=map(int,image_size)
    x=int(np.clip(round(x),0,W-1))
    plane=np.asarray(masks.get(plane_id,[]),bool)
    if plane.shape!=(H,W):return None
    # Lấy median của các cột lân cận để tránh lỗ segmentation.
    r=max(2,int(round(W*.009)))
    top=[];bottom=[];floor_y=[]
    floor=np.asarray(masks.get('floor',[]),bool)
    for col in range(max(0,x-r),min(W,x+r+1)):
        ys=np.flatnonzero(plane[:,col])
        if ys.size < H*.12:continue
        top.append(int(np.quantile(ys,.025)))
        bottom.append(int(np.quantile(ys,.975)))
        if floor.shape==(H,W):
            fys=np.flatnonzero(floor[int(H*.30):,col])
            if fys.size >= max(3,int(H*.009)):
                floor_y.append(int(H*.30)+int(fys[0]))
    if len(top)<max(2,r//2):return None
    t=float(np.median(top))
    b=float(np.median(floor_y)) if len(floor_y)>=max(2,r//2) else float(np.median(bottom))
    mode='observed_floor_seam' if len(floor_y)>=max(2,r//2) else 'visible_wall_bottom_estimate'
    if b-t < H*.23:return None
    return {'top_y':t,'floor_y':b,'wall_height':b-t,'reference_mode':mode,
            'top_is_real_ceiling':False,'metric_height_verified':False}


def v811_tv_height_check(masks,image_size,bbox,plane_id):
    x1,y1,x2,y2=map(int,bbox)
    ref=v811_tv_vertical_reference(masks,image_size,.5*(x1+x2),plane_id)
    if ref is None:return False,{'reason':'missing_reliable_wall_vertical_extent'}
    gap=(ref['floor_y']-y2)/ref['wall_height']
    topgap=(y1-ref['top_y'])/ref['wall_height']
    cfg=TV_WALL_FIRST_POLICY_V811
    good=cfg['min_floor_gap_fraction']-0.012<=gap<=cfg['max_floor_gap_fraction']+0.012
    good &= topgap>=cfg['min_top_gap_fraction']
    out={**ref,'floor_gap_wall_fraction':round(float(gap),4),
         'top_gap_wall_fraction':round(float(topgap),4),
         'preferred_floor_gap_wall_fraction':cfg['preferred_floor_gap_fraction']}
    if not good:out['reason']='TV_height_outside_floor_or_top_clearance'
    return bool(good),out


def v811_tv_size_for_plane(profile,image_size,plane,depth_map=None):
    W,H=map(int,image_size)
    cols=np.flatnonzero(plane.sum(axis=0)>H*.15)
    if len(cols)<W*.08:raise ValueError('wall_plane_too_narrow_for_tv')
    span=int(np.quantile(cols,.98)-np.quantile(cols,.02)+1)
    inches,size_source=tv_diagonal_inches(profile.get('product_name',''))
    tv_width_cm=inches*2.54*16.0/(16.0**2+9.0**2)**.5
    measured=TV_AC_WALL_SCALE_V84.get('measured_front_wall_width_cm')
    wall_cm=float(measured if measured is not None else
                  TV_AC_WALL_SCALE_V84.get('assumed_front_wall_width_cm',350.0))
    # Số đo front-wall không áp dụng vật lý chính xác cho tường bên.
    physical_fraction=tv_width_cm/wall_cm
    rel_factor=1.0;rel_delta=None
    if depth_map is not None:
        try:
            arr=np.asarray(depth_map,float)
            if arr.shape==(H,W):
                ys,xs=np.where(plane)
                mid=len(ys)//2
                mid_x=int(np.median(xs));mid_y=int(np.median(ys))
                rel_factor,rel_delta=_v84_depth_factor(arr,plane,(W,H),(mid_x,mid_y),'TV')
        except (ValueError,KeyError,IndexError):
            pass
    width=int(round(span*np.clip(physical_fraction*rel_factor,
                           TV_WALL_FIRST_POLICY_V811['tv_wall_width_floor_ratio'],
                           TV_WALL_FIRST_POLICY_V811['tv_wall_width_cap_ratio'])))
    height=int(round(width*9/16))
    return width,height,{'method':'v811_per_wall_plane_sku_size_depth',
        'size_source':size_source,'diagonal_inch_estimate':int(inches),
        'product_width_cm_estimate':round(tv_width_cm,2),
        'wall_width_cm_used':wall_cm,'wall_width_calibrated':measured is not None,
        'selected_plane_span_px':span,'relative_depth_factor':round(rel_factor,4),
        'relative_depth_delta':rel_delta,'depth_is_metric':False,
        'estimated_size_px':{'width':width,'height':height}}


def v811_choose_tv_wall(profile,masks,detections,image_size,depth_map=None):
    """Chọn tường tốt nhất, sau đó chọn TV bbox phù hợp trên tường đó.

    Không yêu cầu YOLO phải tìm TV/kệ, không giới hạn ở tường chính,
    không dùng hard rule chiều rộng >=22% toàn ảnh gây false rejection.
    """
    W,H=map(int,image_size)
    cfg=TV_WALL_FIRST_POLICY_V811
    planes=masks.get('tv_plane_meta_v811',{}).get('planes',[])
    if not planes:
        return _unavailable_placement('tv_no_wall_plane',
                  'No wall plane with enough semantic/visual evidence','wall')
    failures=Counter();proposals=[]
    # Cấm giao toàn phần các bbox YOLO và sản phẩm đã ghép.
    obstacle_rects=[]
    for det in detections:
        box=det.get('bbox')
        if box is not None and len(box)==4:
            try:obstacle_rects.append(tuple(map(float,box)))
            except (ValueError,TypeError):pass
    step=max(4,int(round(W*cfg['horizontal_grid_image_fraction'])))
    total_wall_area=max(1,sum(np.asarray(masks[p],bool).sum() for p in planes))
    for plane_id in planes:
        mask=np.asarray(masks.get(plane_id,[]),bool)
        try: bw,bh,scale=v811_tv_size_for_plane(profile,(W,H),mask,depth_map)
        except ValueError as exc:
            failures[str(exc)]+=1;continue
        if min(bw,bh)<12 or bw>=W or bh>=H:
            failures['tv_bbox_unreasonable']+=1;continue
        cols=np.flatnonzero(mask.sum(axis=0)>=max(7,int(H*.11)))
        if not len(cols):continue
        left=int(np.quantile(cols,.02));right=int(np.quantile(cols,.98))
        centers=range(left+bw//2+2,right-bw//2-1,step)
        for cx in centers:
            ref=v811_tv_vertical_reference(masks,(W,H),cx,plane_id)
            if ref is None:failures['missing_wall_vertical_reference']+=1;continue
            for fraction in cfg['floor_gap_options']:
                y2=int(round(ref['floor_y']-fraction*ref['wall_height']))
                y1=y2-bh
                x1=int(round(cx-bw/2));x2=x1+bw
                b=(x1,y1,x2,y2)
                if min(b[:2])<0 or x2>W or y2>H:
                    failures['out_of_bounds']+=1;continue
                good,diag=v811_tv_plane_box_check(masks,(W,H),b,plane_id)
                if not good:
                    failures[diag['reason']]+=1;continue
                height_ok,vertical=v811_tv_height_check(masks,(W,H),b,plane_id)
                if not height_ok:
                    failures[vertical.get('reason','invalid_height')]+=1;continue
                if any((x1<o[2] and x2>o[0] and y1<o[3] and y2>o[1])
                       for o in obstacle_rects):
                    failures['overlaps_existing_object']+=1;continue
                # Favor high mask support, generous wall space and wall center;
                # not the center of the FULL IMAGE (off-center side walls are OK).
                center_penalty=abs(cx-(left+right)/2)/max(1,right-left)
                gap_penalty=abs(fraction-cfg['preferred_floor_gap_fraction'])
                plane_area=float(mask.sum())/total_wall_area
                score=(1.55*diag['plane_coverage']+0.32*np.log1p(plane_area*9)
                       -0.55*center_penalty-2.5*gap_penalty)
                proposals.append((score,plane_id,b,scale,diag,vertical))
    if not proposals:
        result=_unavailable_placement('tv_no_safe_wall_region',
                'No candidate on a single wall meets corners, windows, floor-height and collision constraints',
                'wall')
        result['failed_candidate_counts']=dict(failures)
        result['tv_wall_planes_evaluated']=planes
        return result
    proposals.sort(reverse=True,key=lambda x:x[0])
    score,plane_id,b,scale,diag,vertical=proposals[0]
    x1,y1,x2,y2=b; cx=(x1+x2)//2;cy=(y1+y2)//2
    return {'status':'suggested_unverified','bbox_px':dict(zip(('x1','y1','x2','y2'),b)),
        'bbox_norm':dict(x1=round(x1/W,5),y1=round(y1/H,5),x2=round(x2/W,5),y2=round(y2/H,5)),
        'point_px':{'x':cx,'y':cy},'support_point_px':{'x':cx,'y':cy},
        'surface':'wall','anchor':None,'placement_score':round(float(score),4),
        'wall_plane':diag,'tv_wall_height':vertical,'scale_vs_scene':scale,
        'tv_wall_selection':{'chosen_plane':plane_id,'planes_evaluated':list(planes),
            'candidate_count':len(proposals),'failed_candidate_counts':dict(failures),
            'mode':'single_plane_select_then_position','size_reference':'plane_width_not_full_image'},
        'matched_rules':['single_wall_plane','corners_excluded','no_detected_object_overlap',
            'tv_floor_clearance','no_window_door_overlap'],
        'reason':'TV wall-first selection from segmented wall planes; 2D estimated, not 3D certified',
        'verified':False}


_choose_placement_region_before_tv_v811=choose_placement_region

def choose_placement_region(profile,masks,detections,image_size,
                            max_processing_side=640,depth_map=None):
    if (TV_WALL_FIRST_POLICY_V811['enabled'] and (profile or {}).get('group_name')=='TV'
            and (profile or {}).get('surface')=='wall'):
        return v811_choose_tv_wall(profile,masks,detections,image_size,depth_map)
    return _choose_placement_region_before_tv_v811(
        profile,masks,detections,image_size,
        max_processing_side=max_processing_side,depth_map=depth_map)


def v811_save_tv_wall_debug(image,masks,output_path=None,selected=None):
    """Ảnh QA: màu từng plane, góc cấm màu đỏ, bbox TV cuối màu vàng."""
    base=image.convert('RGB') if isinstance(image,Image.Image) else Image.fromarray(np.asarray(image)).convert('RGB')
    rgb=np.asarray(base).copy();overlay=rgb.copy()
    tones={'tv_plane_front':(53,175,110),'tv_plane_left':(70,130,237),
           'tv_plane_right':(235,157,67),'tv_plane_unresolved':(165,125,217)}
    for k,col in tones.items():
        plane=np.asarray(masks.get(k,[]),bool)
        if plane.shape==rgb.shape[:2]:overlay[plane]=np.asarray(col,dtype=np.uint8)
    bar=np.asarray(masks.get('tv_corner_forbidden',[]),bool)
    if bar.shape==rgb.shape[:2]:overlay[bar]=[230,45,45]
    result=Image.fromarray(cv2.addWeighted(rgb,.66,overlay,.34,0))
    if selected and selected.get('bbox_px'):
        d=ImageDraw.Draw(result)
        b=selected['bbox_px'];d.rectangle([b['x1'],b['y1'],b['x2'],b['y2']],outline=(255,210,30),width=4)
        p=selected.get('tv_wall_selection',{}).get('chosen_plane','tv_plane')
        d.text((b['x1'],max(2,b['y1']-15)),p,fill=(240,185,20))
    if output_path:result.save(output_path)
    return result

# ===== CELL 149 =====
# D.5.8 — V8.12: TV WALL-HEIGHT CALIBRATION + PERSPECTIVE COMPOSITING
# The physical scale is provisional unless measured_wall_height_cm is provided.
# Depth Anything V2 is relative; it must not be treated as metric distance.
import math
from collections import Counter
import cv2
import numpy as np
from PIL import Image, ImageDraw

TV_WALL_GEOMETRY_V812 = {
    'enabled': True,
    'measured_wall_height_cm': None,      # Real measurement, if available
    'assumed_wall_height_cm': 265.0,     # Explicit unverified scene prior
    'seated_eye_level_cm': 108.0,        # TV screen CENTER, not TV bottom
    'minimum_eye_level_cm': 85.0,
    'maximum_eye_level_cm': 130.0,
    'eye_level_trials_cm': (108., 102., 115., 95., 122.),
    'horizontal_foreshortening_strength': 0.36,
    'foreshortening_bounds': (0.80, 1.0),
    'depth_correction_bounds': (0.94, 1.06),
    'perspective_warp': True,
    'perspective_strength': 0.80,
    'maximum_tv_plane_width_fraction': 0.69,
    'minimum_usable_tv_width_px': 45,
    'minimum_plane_coverage': 0.94,
    'edge_clearance_image_fraction': 0.012,
    'candidate_step_image_fraction': 0.008,
    'prefer_front_wall_score_bonus': 0.05,
    'debug': True,
}


def v812_tv_physical_dimensions(profile):
    inches, source = tv_diagonal_inches((profile or {}).get('product_name', ''))
    diag_cm = float(inches) * 2.54
    width_cm = diag_cm * 16.0 / math.sqrt(16.0**2 + 9.0**2)
    height_cm = diag_cm * 9.0 / math.sqrt(16.0**2 + 9.0**2)
    # Product listing dimensions, when available, take precedence over the nominal screen.
    sku = ((profile or {}).get('size_cm') or {}) if (profile or {}).get('size_source')=='sku_verified_input' else {}
    try:
        sw = float(sku.get('width') or 0)
        sh = float(sku.get('height') or 0)
        if 45 <= sw <= 250 and 25 <= sh <= 150 and 1.20 < sw / sh < 2.50:
            width_cm, height_cm = sw, sh
            source = 'sku_body_dimensions_cm'
    except (TypeError, ValueError, ZeroDivisionError):
        pass
    return width_cm, height_cm, int(inches), source


def v812_tv_local_wall_reference(masks, image_size, x, plane_id):
    ref = v811_tv_vertical_reference(masks, image_size, x, plane_id)
    if ref is None:
        return None
    if ref['wall_height'] <= 15 or not np.isfinite(ref['wall_height']):
        return None
    return ref


def v812_tv_estimate_foreshortening(masks, image_size, plane_id, left, right):
    # Large changes in apparent wall height across the same plane indicate an oblique view.
    # This is a bounded visual prior; not a calibrated camera angle.
    span = max(1, right - left)
    refs = []
    for q in (0.18, 0.5, 0.82):
        r = v812_tv_local_wall_reference(masks, image_size, left + q * span, plane_id)
        if r:
            refs.append(r['wall_height'])
    if len(refs) < 2:
        return 1., {'wall_height_variation': None}
    change = (max(refs) - min(refs)) / max(1., np.median(refs))
    cfg = TV_WALL_GEOMETRY_V812
    correction = float(np.clip(1. - cfg['horizontal_foreshortening_strength'] * change,
                               *cfg['foreshortening_bounds']))
    return correction, {'wall_height_variation': round(float(change), 4),
                        'correction_method': 'bounded_vertical_wall_gradient_prior'}


def v812_tv_depth_factor(depth_map, plane, image_size, point):
    if depth_map is None:
        return 1.0, 'no_relative_depth'
    try:
        f, delta = _v84_depth_factor(depth_map, plane, image_size, point, 'TV')
        return float(np.clip(f, *TV_WALL_GEOMETRY_V812['depth_correction_bounds'])), delta
    except (ValueError, TypeError, IndexError, KeyError):
        return 1.0, 'unreliable_relative_depth'


def v812_quad_bbox(quad):
    q = np.asarray(quad, dtype=float)
    return (int(np.floor(q[:,0].min())), int(np.floor(q[:,1].min())),
            int(np.ceil(q[:,0].max())), int(np.ceil(q[:,1].max())))


def v812_tv_projected_quad(masks, image_size, plane_id, center_x, physical_width_cm,
                            physical_height_cm, eye_level_cm, width_px, reference):
    W, H = map(int, image_size)
    x1 = int(round(center_x - width_px / 2))
    x2 = x1 + int(width_px)
    if not 0 <= x1 < x2 < W:
        return None
    assumed_height_cm = float(TV_WALL_GEOMETRY_V812['measured_wall_height_cm'] or
                              TV_WALL_GEOMETRY_V812['assumed_wall_height_cm'])
    local_left = v812_tv_local_wall_reference(masks, (W,H), x1, plane_id)
    local_right = v812_tv_local_wall_reference(masks, (W,H), x2, plane_id)
    if local_left is None or local_right is None:
        return None

    def vertical_placement(r):
        px_per_cm = r['wall_height'] / assumed_height_cm
        mid = r['floor_y'] - eye_level_cm * px_per_cm
        half_h = physical_height_cm * 0.5 * px_per_cm
        return mid - half_h, mid + half_h
    mid_top, mid_bottom = vertical_placement(reference)
    left_top, left_bottom = vertical_placement(local_left)
    right_top, right_bottom = vertical_placement(local_right)
    strength = float(TV_WALL_GEOMETRY_V812['perspective_strength']) if TV_WALL_GEOMETRY_V812['perspective_warp'] else 0.0
    # Blend with the center pose to suppress erratic segmentation boundaries.
    left_top = mid_top + strength * (left_top-mid_top)
    left_bottom = mid_bottom + strength * (left_bottom-mid_bottom)
    right_top = mid_top + strength * (right_top-mid_top)
    right_bottom = mid_bottom + strength * (right_bottom-mid_bottom)
    return [(float(x1),float(left_top)), (float(x2),float(right_top)),
            (float(x2),float(right_bottom)), (float(x1),float(left_bottom))]


def v812_tv_quad_check(masks, image_size, quad, plane_id):
    W,H = map(int, image_size)
    vertices = np.asarray(quad,dtype=np.float32)
    if not np.isfinite(vertices).all():
        return False, {'reason':'tv_perspective_nonfinite'}
    x1,y1,x2,y2 = v812_quad_bbox(vertices)
    if x1<0 or y1<0 or x2>W or y2>H:
        return False, {'reason':'tv_outside_image'}
    qmask=np.zeros((H,W),dtype=np.uint8)
    cv2.fillConvexPoly(qmask,np.rint(vertices).astype(np.int32),1)
    area=int(qmask.sum())
    if area<150:
        return False, {'reason':'tv_quad_too_small'}
    plane=np.asarray(masks.get(plane_id,[]),bool)
    if plane.shape!=(H,W):
        return False, {'reason':'selected_tv_plane_missing'}
    coverage=float(plane[qmask>0].mean())
    if coverage<TV_WALL_GEOMETRY_V812['minimum_plane_coverage']:
        return False, {'reason':'tv_not_on_one_wall','plane_coverage':round(coverage,3)}
    for key in ('tv_corner_forbidden','tv_forbidden_obstacles'):
        forbidden=np.asarray(masks.get(key,[]),bool)
        if forbidden.shape==(H,W) and np.any(forbidden & (qmask>0)):
            return False, {'reason':f'tv_intersects_{key}'}
    return True,{'plane_id':plane_id,'plane_coverage':round(coverage,4),
                 'geometry':'perspective_quad_inside_one_wall',
                 'physical_accuracy_verified':False}


def v812_choose_tv_wall(profile,masks,detections,image_size,depth_map=None):
    W,H=map(int,image_size)
    cfg=TV_WALL_GEOMETRY_V812
    planes=masks.get('tv_plane_meta_v811',{}).get('planes',[])
    if not planes:
        return _unavailable_placement('tv_no_wall_plane','No separated wall plane for TV','wall')
    wall_height_cm=float(cfg['measured_wall_height_cm'] or cfg['assumed_wall_height_cm'])
    if not 190. <= wall_height_cm <= 450.:
        raise ValueError('TV wall height calibration must be 190..450 cm')
    width_cm,height_cm,inches,source=v812_tv_physical_dimensions(profile)
    obstacle_rects=[]
    for det in detections:
        bb=det.get('bbox')
        if bb is not None and len(bb)==4:
            try:obstacle_rects.append(tuple(map(float,bb)))
            except (ValueError,TypeError):pass
    candidates=[];fails=Counter()
    eye_trials=tuple(dict.fromkeys((float(cfg['seated_eye_level_cm']),) + tuple(cfg['eye_level_trials_cm'])))
    step=max(4,int(round(W*cfg['candidate_step_image_fraction'])))
    for plane_id in planes:
        plane=np.asarray(masks.get(plane_id,[]),bool)
        if plane.shape!=(H,W):continue
        cols=np.flatnonzero(plane.sum(axis=0)>=max(8,int(H*.12)))
        if len(cols)<W*.11:continue
        left,right=int(np.quantile(cols,.025)),int(np.quantile(cols,.975))
        span=max(1,right-left)
        angle_factor,angle_meta=v812_tv_estimate_foreshortening(masks,(W,H),plane_id,left,right)
        for cx in range(left+10,right-9,step):
            ref=v812_tv_local_wall_reference(masks,(W,H),cx,plane_id)
            if ref is None:
                fails['tv_wall_height_unavailable']+=1;continue
            depth_factor,depth_delta=v812_tv_depth_factor(depth_map,plane,(W,H),
                                                            (cx,ref['top_y']+ref['wall_height']*.45))
            width_px=int(round(width_cm / wall_height_cm * ref['wall_height']
                               * angle_factor * depth_factor))
            width_px=max(1,width_px)
            if width_px<cfg['minimum_usable_tv_width_px']:
                fails['tv_too_small_for_image_resolution']+=1;continue
            if width_px>span*cfg['maximum_tv_plane_width_fraction']:
                fails['selected_sku_too_large_for_visible_wall']+=1;continue
            if cx-width_px/2<=left+5 or cx+width_px/2>=right-5:
                fails['tv_too_close_to_wall_edge']+=1;continue
            for eye_cm in eye_trials:
                if not cfg['minimum_eye_level_cm'] <= eye_cm <= cfg['maximum_eye_level_cm']:
                    continue
                quad=v812_tv_projected_quad(masks,(W,H),plane_id,cx,width_cm,height_cm,
                                            eye_cm,width_px,ref)
                if quad is None:
                    fails['tv_quad_projection_failed']+=1;continue
                b=v812_quad_bbox(quad)
                if b[0]<0 or b[1]<0 or b[2]>W or b[3]>H:
                    fails['tv_outside_image']+=1;continue
                ok,check=v812_tv_quad_check(masks,(W,H),quad,plane_id)
                if not ok:
                    fails[check['reason']]+=1;continue
                if any(b[0]<o[2] and b[2]>o[0] and b[1]<o[3] and b[3]>o[1]
                       for o in obstacle_rects):
                    fails['tv_overlaps_detected_object']+=1;continue
                center_penalty=abs(cx-(left+right)/2)/max(1,span)
                eye_penalty=abs(eye_cm-cfg['seated_eye_level_cm'])/30
                # Wall center, eye-level comfort, plane coverage and robust scale.
                score=(1.45*check['plane_coverage']-0.55*center_penalty-
                       0.30*eye_penalty+0.10*min(1,width_px/max(1,span))+
                       (cfg['prefer_front_wall_score_bonus'] if 'front' in plane_id else 0))
                meta={'method':'v812_local_wall_height_sku_and_perspective',
                      'diagonal_inch_estimate':inches,'size_source':source,
                      'product_width_cm_estimate':round(width_cm,2),
                      'product_height_cm_estimate':round(height_cm,2),
                      'wall_height_cm_used':round(wall_height_cm,2),
                      'wall_height_calibrated':cfg['measured_wall_height_cm'] is not None,
                      'local_wall_height_px':round(ref['wall_height'],2),
                      'relative_depth_factor':round(depth_factor,4),
                      'relative_depth_delta':depth_delta,
                      'horizontal_foreshortening':round(angle_factor,4),
                      **angle_meta,'estimated_size_px':{'width':width_px,
                        'height_at_center':round(height_cm/wall_height_cm*ref['wall_height'],1)},
                      'depth_is_metric':False,'metric_size_verified':False}
                candidates.append((score,plane_id,b,quad,meta,eye_cm,check,ref))
    if not candidates:
        result=_unavailable_placement('tv_no_safe_wall_region',
                   'No wall supports the selected TV dimensions and perspective safely','wall')
        result['failed_candidate_counts']=dict(fails)
        result['tv_wall_planes_evaluated']=list(planes)
        return result
    candidates.sort(key=lambda x:x[0],reverse=True)
    score,plane_id,b,quad,meta,eye_cm,check,ref=candidates[0]
    x1,y1,x2,y2=b
    return {'status':'suggested_unverified','bbox_px':dict(zip(('x1','y1','x2','y2'),b)),
      'bbox_norm':{'x1':round(x1/W,5),'y1':round(y1/H,5),'x2':round(x2/W,5),'y2':round(y2/H,5)},
      'point_px':{'x':int((x1+x2)/2),'y':int((y1+y2)/2)},
      'support_point_px':{'x':int((x1+x2)/2),'y':int((y1+y2)/2)},
      'surface':'wall','anchor':None,'placement_score':round(float(score),4),
      'tv_projective_quad':[[round(v,2) for v in pair] for pair in quad],
      'wall_plane':check,'scale_vs_scene':meta,
      'tv_wall_height':{'reference_mode':ref['reference_mode'],
                        'wall_height_px':round(ref['wall_height'],2),
                        'target_tv_center_above_floor_cm':eye_cm,
                        'wall_height_calibrated':cfg['measured_wall_height_cm'] is not None},
      'tv_wall_selection':{'chosen_plane':plane_id,'planes_evaluated':list(planes),
        'candidate_count':len(candidates),'failed_candidate_counts':dict(fails),
        'mode':'v812_wall_height_perspective_quad'},
      'matched_rules':['single_wall_plane','corner_clearance','depth_relative_scale',
                       'physical_sku_nominal','seated_eye_level','no_overlap'],
      'reason':'Calibrated TV size from local wall height (estimate unless measured); perspective quad',
      'verified':False}

_v811_choose_placement_before_v812=choose_placement_region

def choose_placement_region(profile,masks,detections,image_size,
                            max_processing_side=640,depth_map=None):
    if (TV_WALL_GEOMETRY_V812['enabled'] and (profile or {}).get('group_name')=='TV'
        and (profile or {}).get('surface')=='wall'):
        return v812_choose_tv_wall(profile,masks,detections,image_size,depth_map)
    return _v811_choose_placement_before_v812(profile,masks,detections,image_size,
               max_processing_side=max_processing_side,depth_map=depth_map)


def v812_composite_tv_cutout(room,cutout,placement,masks):
    '''Project an API-validated RGBA TV onto the wall quad; no background-removal fallback.'''
    if cutout.mode!='RGBA' or 'tv_projective_quad' not in placement:
        raise ValueError('TV requires an API RGBA cutout and a calibrated wall quad')
    alpha=cutout.getchannel('A')
    crop=alpha.point(lambda a:255 if a>8 else 0).getbbox()
    if crop is None:
        raise ValueError('API TV cutout has no visible alpha')
    rgba=cutout.crop(crop)
    sw,sh=rgba.size
    source_aspect=sw/max(1,sh)
    if not 1.20 <= source_aspect <= 3.00:
        raise ValueError(f'API TV cutout aspect {source_aspect:.2f} is not screen-like')
    W,H=room.size
    quad=np.asarray(placement['tv_projective_quad'],dtype=np.float32)
    src=np.asarray([(0,0),(sw-1,0),(sw-1,sh-1),(0,sh-1)],dtype=np.float32)
    Hmat=cv2.getPerspectiveTransform(src,quad)
    # Premultiplied-alpha warp prevents dark/black halos at transformed edges.
    api_rgba=np.asarray(rgba,dtype=np.float32) / 255.0
    premul_rgb=api_rgba[:,:,:3] * api_rgba[:,:,3:4]
    premul=np.concatenate([premul_rgb,api_rgba[:,:,3:4]],axis=2)
    warped_pm=cv2.warpPerspective(premul,Hmat,(W,H),flags=cv2.INTER_LINEAR,
                   borderMode=cv2.BORDER_CONSTANT,borderValue=(0.,0.,0.,0.))
    alpha_out=np.clip(warped_pm[:,:,3:4],0.,1.)
    rgb_out=np.divide(warped_pm[:,:,:3],np.maximum(alpha_out,1e-6))
    warped=np.uint8(np.round(np.clip(np.concatenate([rgb_out,alpha_out],axis=2),0.,1.)*255.))
    visible=(warped[:,:,3]>8)
    if visible.sum()<100:
        raise ValueError('TV perspective cutout becomes too small')
    plane_id=placement['tv_wall_selection']['chosen_plane']
    plane=np.asarray(masks.get(plane_id,[]),bool)
    if plane.shape!=(H,W):raise ValueError('TV wall plane absent after API')
    coverage=float(plane[visible].mean())
    if coverage<TV_WALL_GEOMETRY_V812['minimum_plane_coverage']:
        raise ValueError('TV alpha cutout crosses another wall')
    for key in ('tv_corner_forbidden','tv_forbidden_obstacles'):
        forbidden=np.asarray(masks.get(key,[]),bool)
        if forbidden.shape==(H,W) and np.any(forbidden & visible):
            raise ValueError(f'TV cutout intersects forbidden region: {key}')
    ys,xs=np.where(visible)
    actual={'x1':int(xs.min()),'y1':int(ys.min()),'x2':int(xs.max()+1),'y2':int(ys.max()+1)}
    out=room.convert('RGBA').copy()
    out.alpha_composite(Image.fromarray(warped,mode='RGBA'),(0,0))
    return out,actual,{'wall_plane':plane_id,'alpha_wall_coverage':round(coverage,4),
                       'projective_quad':placement['tv_projective_quad'],
                       'method':'cv2_projective_warp_of_API_RGBA_only'}



def v812_validate_projected_tv_scale(placement, rendered_bbox, image_size):
    """Use projective-quad dimensions rather than axis-aligned 16:9 thresholds."""
    W,H=map(int,image_size)
    proposed=v812_quad_bbox(placement['tv_projective_quad'])
    pw=max(1,proposed[2]-proposed[0]);ph=max(1,proposed[3]-proposed[1])
    rw=int(rendered_bbox['x2']-rendered_bbox['x1'])
    rh=int(rendered_bbox['y2']-rendered_bbox['y1'])
    if not (.82*pw <= rw <= 1.08*pw and .82*ph <= rh <= 1.08*ph):
        raise ValueError('Warped API TV silhouette does not match calibrated quad')
    if rw<28 or rh<20:
        raise ValueError('Warped API TV is too small')
    return {'placement_width_px':pw,'rendered_width_px':rw,
            'rendered_width_image_ratio':round(rw/max(1,W),4),
            'rendered_bbox_aspect_ratio':round(rw/max(1,rh),3),
            'projected_tv_geometry':True,'scale_method':'wall_height_perspective_v812'}

def v812_debug_tv_geometry(image,masks,placement,output_path=None):
    base=image.convert('RGB') if isinstance(image,Image.Image) else Image.fromarray(np.asarray(image)).convert('RGB')
    preview=v811_save_tv_wall_debug(base,masks,selected=placement)
    d=ImageDraw.Draw(preview)
    q=placement.get('tv_projective_quad')
    if q:
        coords=[tuple(int(round(v)) for v in pair) for pair in q]
        d.line(coords+[coords[0]],fill=(255,238,0),width=4)
        meta=placement.get('scale_vs_scene',{})
        label=f'{meta.get("diagonal_inch_estimate","?")}in | wall {meta.get("local_wall_height_px","?")}px'
        d.text((coords[0][0],max(4,coords[0][1]-18)),label,fill=(255,238,0))
    if output_path:
        preview.save(output_path)
    return preview

# ===== CELL 151 =====
# D.5.9 — TV: FRONT WALL FIRST -> SIDE WALL ONLY IF FRONT INVALID.
# Keep V8.12 SKU dimensions, Depth Anything relative scaling, perspective quad,
# strict API cutout and photometric rendering UNCHANGED.
# Center on the selected physical wall span, NOT on the entire photo.
from collections import Counter
import numpy as np
from PIL import Image, ImageDraw

TV_WALL_SELECTION_V818 = {
    'enabled': True,
    'front_plane_names': ('tv_plane_front',),
    'unresolved_as_primary_only_when_no_front': True,
    'side_plane_names': ('tv_plane_left', 'tv_plane_right'),
    'max_wall_center_offset_fraction': 0.10,  # max distance from estimated wall midpoint
    'center_trials_fraction': (0.0, -0.02, 0.02, -0.04, 0.04,
                               -0.06, 0.06, -0.08, 0.08, -0.10, 0.10),
    'min_wall_columns': 16,
    'min_bbox_width_px': 45,
    'min_visible_wall_width_fraction': 0.10,  # only quality guard, not TV size ratio
    'wall_center_sample_columns': 160,
    'minimum_wall_height_fraction_image': 0.18,
    'debug': True,
}


def v818_wall_metric_center(masks, image_size, plane_id):
    """Estimate the midpoint along ONE perspective wall plane.

    The apparent local vertical wall size is a bounded proxy for image pixels per
    physical centimeter. Integrating dx / wall_height(x) approximates equal
    physical distance along an oblique wall. This remains monocular 2D geometry,
    not a metric 3D reconstruction. Depth is used for TV size at each candidate.
    """
    W, H = map(int, image_size)
    plane = np.asarray(masks.get(plane_id, []), dtype=bool)
    if plane.shape != (H, W):
        return None
    qualifying = np.flatnonzero(plane.sum(axis=0) >= max(6, int(H * 0.10)))
    if qualifying.size < TV_WALL_SELECTION_V818['min_wall_columns']:
        return None
    left = int(np.quantile(qualifying, .015))
    right = int(np.quantile(qualifying, .985))
    if right - left < max(TV_WALL_SELECTION_V818['min_wall_columns'],
                          int(W * TV_WALL_SELECTION_V818['min_visible_wall_width_fraction'])):
        return None
    sampling = np.unique(np.rint(np.linspace(
        left, right, min(TV_WALL_SELECTION_V818['wall_center_sample_columns'],
                         right-left+1))).astype(int))
    xs = []
    heights = []
    for x in sampling:
        ref = v812_tv_local_wall_reference(masks, (W,H), int(x), plane_id)
        if ref is None or ref['wall_height'] < H * TV_WALL_SELECTION_V818['minimum_wall_height_fraction_image']:
            continue
        xs.append(float(x))
        heights.append(float(ref['wall_height']))
    if len(xs) < 5 or xs[-1] - xs[0] < .50 * (right-left):
        return None
    xs = np.asarray(xs, float)
    heights = np.asarray(heights, float)
    # Cap local variation to avoid a segmentation hole dominating the integral.
    middle_height = float(np.median(heights))
    heights = np.clip(heights, .55*middle_height, 1.65*middle_height)
    segments = np.diff(xs) * .5 * (1.0/heights[:-1] + 1.0/heights[1:])
    cumulative = np.r_[0.0, np.cumsum(segments)]
    if cumulative[-1] <= 1e-8:
        return None
    fractions = cumulative/cumulative[-1]
    return {'x_samples': xs, 'distance_fractions':fractions,
            'center_x':float(np.interp(.5, fractions, xs)),
            'extent_px':[left,right],
            'projection_method':'inverse_local_wall_height_integral',
            'depth_is_metric':False,
            'width_px':right-left}


def v818_choose_tv_wall(profile, masks, detections, image_size, depth_map=None):
    """Strict ordered search: main wall, then secondary walls; align to wall midpoint.

    A secondary wall is NEVER considered while a safe centered front-wall
    placement exists. No shrinking the chosen SKU to force it onto a wall.
    """
    W,H = map(int,image_size)
    cfg=TV_WALL_SELECTION_V818
    v12=TV_WALL_GEOMETRY_V812
    available=list(masks.get('tv_plane_meta_v811',{}).get('planes',[]))
    if not available:
        return _unavailable_placement('tv_no_wall_plane',
            'No wall plane identified for the TV','wall')
    main=[p for p in cfg['front_plane_names'] if p in available]
    if not main and cfg['unresolved_as_primary_only_when_no_front']:
        main=[p for p in available if p=='tv_plane_unresolved']
    sides=[p for p in cfg['side_plane_names'] if p in available and p not in main]
    # If a detector yields an additional explicitly named plane, check it only
    # after the main wall and known side walls.
    remaining=[p for p in available if p not in main and p not in sides]
    phases=[('main_wall',main),('side_wall',sides+remaining)]
    wall_height_cm=float(v12['measured_wall_height_cm'] or v12['assumed_wall_height_cm'])
    if not (190.0<=wall_height_cm<=450.0):
        raise ValueError('TV wall calibration must be between 190 and 450 cm')
    tv_width_cm,tv_height_cm,tv_inches,source=v812_tv_physical_dimensions(profile)
    eye_trials=tuple(dict.fromkeys((float(v12['seated_eye_level_cm']),)
                                   +tuple(map(float,v12['eye_level_trials_cm']))))
    occupied=[]
    for d in detections:
        bbox=d.get('bbox')
        if bbox is not None and len(bbox)==4:
            try: occupied.append(tuple(map(float,bbox)))
            except (ValueError, TypeError): pass
    fail_counts=Counter()
    by_plane={}
    wall_centers={}
    # Ordered phases are the priority contract (not a soft score bonus).
    for phase_name,phase_planes in phases:
        proposals=[]
        for plane_id in phase_planes:
            plane=np.asarray(masks.get(plane_id,[]),bool)
            if plane.shape!=(H,W):
                fail_counts['wall_mask_missing']+=1
                continue
            geometry=v818_wall_metric_center(masks,(W,H),plane_id)
            if geometry is None:
                fail_counts['wall_extent_unreliable']+=1
                by_plane[plane_id]={'status':'unreliable_wall_extent'}
                continue
            wall_centers[plane_id]=geometry['center_x']
            left,right=geometry['extent_px']
            span=geometry['width_px']
            angle_factor,angle_diag=v812_tv_estimate_foreshortening(
                masks,(W,H),plane_id,left,right)
            local_fail=Counter();accepted=0
            for offset in cfg['center_trials_fraction']:
                if abs(offset)>cfg['max_wall_center_offset_fraction']+1e-8:
                    continue
                center_fraction=.5+float(offset)
                cx=float(np.interp(center_fraction, geometry['distance_fractions'],
                                   geometry['x_samples']))
                ref=v812_tv_local_wall_reference(masks,(W,H),cx,plane_id)
                if ref is None:
                    local_fail['missing_wall_height']+=1
                    continue
                depth_factor,depth_delta=v812_tv_depth_factor(
                    depth_map,plane,(W,H),
                    (cx,ref['top_y']+ref['wall_height']*.45))
                tv_w=int(round(tv_width_cm/wall_height_cm*ref['wall_height']
                               *angle_factor*depth_factor))
                if tv_w < max(cfg['min_bbox_width_px'],v12['minimum_usable_tv_width_px']):
                    local_fail['tv_too_small_at_depth']+=1
                    continue
                if tv_w > span*v12['maximum_tv_plane_width_fraction']:
                    local_fail['tv_too_wide_for_wall']+=1
                    continue
                edge_gap=max(5,int(round(W*v12['edge_clearance_image_fraction'])))
                if cx-tv_w*.5<left+edge_gap or cx+tv_w*.5>right-edge_gap:
                    local_fail['tv_crosses_plane_edge']+=1
                    continue
                for eye_cm in eye_trials:
                    if not v12['minimum_eye_level_cm']<=eye_cm<=v12['maximum_eye_level_cm']:
                        continue
                    quad=v812_tv_projected_quad(
                        masks,(W,H),plane_id,cx,tv_width_cm,tv_height_cm,
                        eye_cm,tv_w,ref)
                    if quad is None:
                        local_fail['perspective_projection_failed']+=1
                        continue
                    rect=v812_quad_bbox(quad)
                    if rect[0]<0 or rect[1]<0 or rect[2]>W or rect[3]>H:
                        local_fail['out_of_bounds']+=1
                        continue
                    valid,diag=v812_tv_quad_check(masks,(W,H),quad,plane_id)
                    if not valid:
                        local_fail[diag['reason']]+=1
                        continue
                    if any(rect[0]<b[2] and rect[2]>b[0] and
                           rect[1]<b[3] and rect[3]>b[1] for b in occupied):
                        local_fail['collision_with_existing_object']+=1
                        continue
                    # Higher score for the ACTUAL geometric center of the wall.
                    # Distances here are estimated wall-distance fractions.
                    position_error=abs(offset)
                    eye_error=abs(eye_cm-v12['seated_eye_level_cm'])/30.0
                    score=(2.0*diag['plane_coverage']-4.0*position_error
                           -.35*eye_error+.07*(tv_w/max(1,span)))
                    scale={'method':'v812_local_wall_height_sku_depth_v818_centered',
                           'diagonal_inch_estimate':tv_inches,'size_source':source,
                           'product_width_cm_estimate':round(tv_width_cm,2),
                           'product_height_cm_estimate':round(tv_height_cm,2),
                           'wall_height_cm_used':round(wall_height_cm,2),
                           'wall_height_calibrated':v12['measured_wall_height_cm'] is not None,
                           'local_wall_height_px':round(ref['wall_height'],2),
                           'relative_depth_factor':round(depth_factor,4),
                           'relative_depth_delta':depth_delta,
                           'horizontal_foreshortening':round(angle_factor,4),
                           **angle_diag,
                           'estimated_size_px':{'width':tv_w,
                             'height_at_center':round(tv_height_cm/wall_height_cm*ref['wall_height'],1)},
                           'depth_is_metric':False,'metric_size_verified':False}
                    proposals.append((score,plane_id,rect,quad,scale,eye_cm,diag,ref,
                                      center_fraction,cx,geometry))
                    accepted+=1
            fail_counts.update(local_fail)
            by_plane[plane_id]={'status':'has_centered_candidates' if accepted else 'no_safe_centered_candidate',
                'valid_candidates':accepted,'failures':dict(local_fail),
                'wall_center_x_px':round(geometry['center_x'],2),
                'wall_x_extent_px':[left,right],'estimated_wall_width_px':span}
        if proposals:
            proposals.sort(key=lambda p:p[0], reverse=True)
            score,plane_id,b,quad,scale,eye_cm,check,ref,center_fraction,cx,geometry=proposals[0]
            x1,y1,x2,y2=b
            return {'status':'suggested_unverified',
                'bbox_px':dict(zip(('x1','y1','x2','y2'),b)),
                'bbox_norm':{'x1':round(x1/W,5),'y1':round(y1/H,5),
                             'x2':round(x2/W,5),'y2':round(y2/H,5)},
                'point_px':{'x':int((x1+x2)/2),'y':int((y1+y2)/2)},
                'support_point_px':{'x':int((x1+x2)/2),'y':int((y1+y2)/2)},
                'surface':'wall','anchor':None,'placement_score':round(float(score),4),
                'tv_projective_quad':[[round(v,2) for v in pair] for pair in quad],
                'wall_plane':check,'scale_vs_scene':scale,
                'tv_wall_height':{'reference_mode':ref['reference_mode'],
                    'wall_height_px':round(ref['wall_height'],2),
                    'target_tv_center_above_floor_cm':eye_cm,
                    'wall_height_calibrated':v12['measured_wall_height_cm'] is not None},
                'tv_wall_selection':{
                    'chosen_plane':plane_id,
                    'selection_phase':phase_name,
                    'ordered_phases':['main_wall','side_wall'],
                    'planes_evaluated':available,
                    'plane_decisions':by_plane,
                    'chosen_wall_center_x_px':round(geometry['center_x'],2),
                    'chosen_tv_center_x_px':round(cx,2),
                    'tv_center_fraction_of_estimated_wall_distance':round(center_fraction,4),
                    'center_offset_fraction':round(abs(center_fraction-.5),4),
                    'wall_center_method':geometry['projection_method'],
                    'depth_metric':False,
                    'candidate_count_in_selected_phase':len(proposals),
                    'failed_candidate_counts':dict(fail_counts),
                    'mode':'v818_primary_wall_first_side_if_no_safe_center',
                },
                'matched_rules':['main_wall_first','side_wall_only_if_main_invalid',
                    'depth_relative_scale','wall_center_preference',
                    'single_wall_plane','corner_clearance','no_overlap',
                    'seated_eye_level','api_rgba_required'],
                'reason':('Centered TV on main wall' if phase_name=='main_wall'
                          else 'No safe centered position on main wall; use side wall'),
                'verified':False}
    result=_unavailable_placement('tv_no_safe_wall_region',
       'Neither main wall nor side walls have a safe CENTERED TV placement','wall')
    result['failed_candidate_counts']=dict(fail_counts)
    result['tv_wall_planes_evaluated']=available
    result['tv_wall_plane_decisions']=by_plane
    return result

# Override TV selection ONLY. The existing V8.12 choose_placement_region wrapper
# resolves v812_choose_tv_wall at call time. No modifications to other products.
v812_choose_tv_wall = v818_choose_tv_wall

# Retain V8.12 perspective/debug, add the estimated physical wall center marker.
_v812_debug_tv_geometry_before_v818 = v812_debug_tv_geometry

def v812_debug_tv_geometry(image, masks, placement, output_path=None):
    preview=_v812_debug_tv_geometry_before_v818(image,masks,placement)
    info=(placement or {}).get('tv_wall_selection',{})
    x=info.get('chosen_wall_center_x_px')
    if x is not None:
        draw=ImageDraw.Draw(preview)
        w,h=preview.size
        xx=int(round(float(x)))
        draw.line([(xx,0),(xx,h-1)],fill=(0,245,255),width=max(1,w//420))
        draw.text((min(w-155,xx+5),10),
                  'wall midpoint (estimated)',fill=(0,225,240))
    if output_path:preview.save(output_path)
    return preview

# ===== CELL 153 =====
def stage1_recommend(source=IMAGE_INPUT, random_seed=RECOMMEND_RANDOM_SEED,
                     show_result=True):
    image=load_image(source)
    response,response_source=request_external_recommendations(image)
    parsed=parse_and_random_select_api_response(response,top_k=None,seed=random_seed)
    parsed['api_endpoint']=RECOMMEND_API_URL
    parsed['api_query_params']=_recommend_query_params()
    parsed['api_response_source']=response_source
    parsed['input_image']=image
    parsed['input_size']={'width':image.width,'height':image.height}
    parsed['spatial_detections']=extract_api_spatial_detections(
        parsed.get('detected_objects',[]),image.size)
    # Cache metadata duy nhất để placement nhận tên group/size, KHÔNG chạy recommender.
    API_CATEGORY_PRODUCTS.clear()
    for entry in parsed['recommendations']:
        API_CATEGORY_PRODUCTS.setdefault(entry['placement_group'],[]).extend(entry['products'])
    for entry in parsed['recommendations']:
        apply_api_sku_surface_metadata(entry['placement_group'],entry['selected_product'])
    sync_placement_profiles_with_catalog(API_CATEGORY_PRODUCTS,display_audit=False)
    attach_group_name_to_profiles()
    rows=[]
    for rank,entry in enumerate(parsed['recommendations'],1):
        prod=entry['selected_product']
        rows.append({'rank':rank,'category_id':entry['category_id'],
          'group_name':entry['group_name'],'placement_group':entry['placement_group'],
          'final_score':entry.get('final_score'),
          'product_id':prod['product_id'],'product_code':prod['product_code'],
          'product_name':prod['product_name'],'price':prod['price'],
          'currency':prod['currency'],'image_url':prod['image_url'],
          'product_type':resolve_remove_bg_product_type(entry,prod),
          'mapping_status':entry.get('category_mapping',{}).get('status'),
          'official_category_name':entry.get('category_mapping',{}).get('official_category_name'),
          'id_has_collision':entry.get('category_mapping',{}).get('id_has_collision'),
          'reason':entry.get('reason','')})
    parsed['recommendation_table']=pd.DataFrame(rows)
    if show_result:
        detected=parsed.get('detected_objects',[])
        print(f'API detect: {len(detected)} nhãn; có bbox: {len(parsed["spatial_detections"])}')
        if detected:display(pd.DataFrame(detected))
        if detected and not parsed['spatial_detections']:
            print('API không trả bbox; không thể chặn vật thể đã phát hiện bằng bbox. '
                  'Spatial Placement vẫn sử dụng Mask2Former và các vùng cấm.')
        print('API recommend:',len(rows),'category có SKU; random 1 SKU/category')
        if rows:
            display(parsed['recommendation_table'])
            status_counts=parsed['recommendation_table']['mapping_status'].value_counts(dropna=False).to_dict()
            print('Mapping category:',status_counts)
        else:print('API trả danh sách gợi ý rỗng hoặc không có SKU khả dụng.')
    return parsed

# ===== CELL 155 =====
# F.1 — Backend tách nền độc lập với Recommend API.
import requests, json, re, uuid, mimetypes
from urllib.parse import urlparse,urljoin
from io import BytesIO
from datetime import datetime, timezone
from PIL import Image, ImageDraw

# CHỈ điền ở đây sau khi đối chiếu /openapi.json hoặc thông báo 422 của backend.
# Key có thể là '<category_id>:<group_name>', category_id, placement_group,
# hoặc product_type cũ (theo thứ tự ưu tiên trong resolver ở B.4.2).
# Mặc định theo curl của bạn: product_type đi trong multipart form.
# Nếu OpenAPI hiện ghi in=query, đổi thành query; KHÔNG tự thử fallback.
REMOVE_BG_PRODUCT_TYPE_LOCATION = os.environ.get('REMOVE_BG_PRODUCT_TYPE_LOCATION','form').strip().lower()
if REMOVE_BG_PRODUCT_TYPE_LOCATION not in ('form','query'):
    raise ValueError('REMOVE_BG_PRODUCT_TYPE_LOCATION phải là form hoặc query')
API_PRODUCT_TYPE_OVERRIDES = {
    # '38:Tivi': 'tv',
    # 'Máy điều hòa': 'air_conditioner',
}
PRODUCT_IMAGE_DOMAINS = ('tgdd.vn','dienmayxanh.com','mwgshop.vn')
MAX_PRODUCT_IMAGE_BYTES = 15*1024*1024
MAX_API_PNG_BYTES = 25*1024*1024
API_TIMEOUT = (10,120)
print('Remove Background endpoint:', REMOVE_BG_API_URL)

# ===== CELL 156 =====
# F.1b — KIỂM TRA SCHEMA BACKEND (chạy thủ công trên Colab nếu vẫn HTTP 422).
# Không gọi API tách nền thử với sản phẩm khác, không đổi loại sản phẩm tự động.
def inspect_remove_bg_api_schema(verbose=True):
    endpoint = REMOVE_BG_API_URL or (f"{REMOVE_BG_BASE_URL.rstrip('/')}/api/v1/products/remove-background" if REMOVE_BG_BASE_URL else "")
    if not endpoint:
        if verbose: print('Thiếu REMOVE_BG_BASE_URL hoặc REMOVE_BG_API_URL trong cấu hình.')
        return {'status': 'unavailable', 'reason': 'missing endpoint'}
    url = endpoint.split('/api/v1/',1)[0] + '/openapi.json'
    try:
        response = requests.get(url,timeout=(6,15),
                                headers={'Accept':'application/json',
                                         'ngrok-skip-browser-warning':'true'},
                                allow_redirects=False)
        if response.status_code != 200:
            raise RuntimeError(f'GET /openapi.json HTTP {response.status_code}: {response.text[:180]}')
        doc=response.json()
    except (requests.RequestException, ValueError, RuntimeError) as exc:
        if verbose: print('Không đọc được OpenAPI:',type(exc).__name__,str(exc)[:220])
        return {'status':'unavailable','reason':str(exc)}
    spec=(doc.get('paths',{}).get('/api/v1/products/remove-background',{})
          .get('post',{}))
    if not spec:
        if verbose: print('Không tìm thấy POST remove-background trong /openapi.json')
        return {'status':'path_not_found'}
    form=spec.get('requestBody',{}).get('content',{})
    fields={}
    for media_type,entry in form.items():
        schema=entry.get('schema',{})
        ref=schema.get('$ref')
        if ref and ref.startswith('#/components/schemas/'):
            schema=doc.get('components',{}).get('schemas',{}).get(ref.rsplit('/',1)[-1],{})
        fields[media_type]={'required':schema.get('required',[]),
                            'properties':schema.get('properties',{})}
    out={'status':'ok','parameters':spec.get('parameters',[]), 'form_schemas':fields}
    if verbose:
        print('POST',REMOVE_BG_API_URL)
        print('parameters:',json.dumps(out['parameters'],ensure_ascii=False,indent=2)[:2800])
        print('form schemas:',json.dumps(fields,ensure_ascii=False,indent=2)[:3500])
        print('Nếu product_type có enum khác, chỉnh API_PRODUCT_TYPE_OVERRIDES trong F.1.')
    return out

# ===== CELL 157 =====
# F.2 — Tải ĐÚNG image_url của SKU được gợi ý, không dùng ảnh thay thế.
def download_selected_product_image(image_url):
    url = str(image_url or '').strip()
    for redirect_idx in range(4):
        parsed = urlparse(url)
        host=(parsed.hostname or '').lower()
        if (parsed.scheme!='https' or not host or
            not any(host==d or host.endswith('.'+d) for d in PRODUCT_IMAGE_DOMAINS) or
            parsed.username or parsed.password or parsed.port not in (None,443)):
            raise ValueError('image_url ngoài domain CDN sản phẩm được cho phép')
        with requests.get(url,stream=True,timeout=(10,40),allow_redirects=False,
                          headers={'User-Agent':'Mozilla/5.0'}) as response:
            if response.status_code in (301,302,303,307,308):
                if redirect_idx==3:raise ValueError('Quá nhiều chuyển hướng image_url')
                location=response.headers.get('Location')
                if not location:raise ValueError('Image URL redirect không có Location')
                url=urljoin(url,location)
                continue
            response.raise_for_status()
            if not response.headers.get('Content-Type','').lower().startswith('image/'):
                raise ValueError('image_url không trả dữ liệu image')
            parts=[];total=0
            for piece in response.iter_content(65536):
                total+=len(piece)
                if total>MAX_PRODUCT_IMAGE_BYTES:raise ValueError('Ảnh sản phẩm vượt giới hạn 15MB')
                parts.append(piece)
        raw=b''.join(parts)
        with Image.open(BytesIO(raw)) as image:
            image.verify()
        with Image.open(BytesIO(raw)) as image:
            fmt=(image.format or '').upper()
        extension={'JPEG':'jpg','PNG':'png','WEBP':'webp'}.get(fmt)
        if not extension:raise ValueError('Chỉ chấp nhận ảnh nguồn JPEG, PNG, WebP')
        mime={'jpg':'image/jpeg','png':'image/png','webp':'image/webp'}[extension]
        return raw,'selected_product.'+extension,mime
    raise ValueError('Không tải được ảnh sản phẩm')

# ===== CELL 158 =====
# F.3 — Gọi ĐÚNG endpoint Remove Background theo Swagger: 2 form fields:
# image: file JPEG/PNG/WEBP ; product_type: chuỗi. KHÔNG fallback.
# Với HTTP 422 phải hiện response JSON của FastAPI để biết field nào sai.
def format_api_validation_error(response):
    ct=response.headers.get('Content-Type','')
    try:
        payload=response.json()
    except (ValueError, TypeError):
        return f'Content-Type={ct}; body={response.text[:1300]}'
    detail=payload.get('detail',payload) if isinstance(payload,dict) else payload
    if isinstance(detail,list):
        parts=[]
        for item in detail[:10]:
            if isinstance(item,dict):
                loc='.'.join(map(str,item.get('loc',[]))) or 'unknown'
                msg=item.get('msg','')
                kind=item.get('type','')
                extra=item.get('ctx',{})
                parts.append(f'{loc}: {msg} (type={kind}; ctx={extra})')
            else: parts.append(str(item))
        return '; '.join(parts)[:1600]
    return json.dumps(payload,ensure_ascii=False)[:1600]

def require_api_cutout(source_bytes, filename, mime, product_type, api_url=None,
                       product_id=None, product_name=None, group_name=None):
    """Gọi API Remove Background để tách nền sản phẩm thành ảnh PNG RGBA.

    Ghi log chi tiết cả khi gọi thành công lẫn khi gặp lỗi. Không dùng fallback.
    """
    prod_label = f"[{product_id or 'N/A'} - {product_name or 'N/A'}]"
    group_label = group_name or "N/A"

    api_url = (api_url or REMOVE_BG_API_URL or "").strip()
    if not api_url and REMOVE_BG_BASE_URL:
        _b = REMOVE_BG_BASE_URL.rstrip("/")
        api_url = _b if "/api/v1/products/remove-background" in _b else f"{_b}/api/v1/products/remove-background"
    if not api_url:
        err_msg = f"Thiếu endpoint Remove Background API cho sản phẩm {prod_label}. Vui lòng cấu hình REMOVE_BG_BASE_URL trong .env"
        logger.error("Gọi Remove Background API THẤT BẠI: %s", err_msg)
        raise ValueError(err_msg)

    pt_norm = str(product_type or "").strip().lower()
    if pt_norm in ("tv", "tivi"):
        product_type = "tv"
    elif pt_norm in ("air_conditioner", "may_lanh", "may lanh", "điều hòa", "dieu hoa"):
        product_type = "air_conditioner"
    else:
        product_type = "auto"

    if mime not in ('image/jpeg', 'image/png', 'image/webp'):
        err_msg = f"MIME ảnh không hợp lệ: '{mime}' cho sản phẩm {prod_label}"
        logger.error("Gọi Remove Background API THẤT BẠI: %s", err_msg)
        raise ValueError(err_msg)

    if not isinstance(source_bytes, bytes) or not source_bytes:
        err_msg = f"Không có dữ liệu bytes ảnh nguồn hợp lệ cho sản phẩm {prod_label}"
        logger.error("Gọi Remove Background API THẤT BẠI: %s", err_msg)
        raise ValueError(err_msg)

    # Chọn đúng vị trí theo schema backend, không tự retry bằng phương thức khác.
    product_type_location = REMOVE_BG_PRODUCT_TYPE_LOCATION
    form_data = {'product_type': product_type} if product_type_location == 'form' else {}
    query_params = {'product_type': product_type} if product_type_location == 'query' else {}

    logger.info(
        "Bắt đầu gọi Remove Background API cho sản phẩm %s (nhóm: %s, product_type: %s, file: %s, dung lượng: %d bytes) tới endpoint: %s",
        prod_label, group_label, product_type, filename, len(source_bytes), api_url,
    )

    t0 = time.perf_counter()
    try:
        r = requests.post(
            api_url,
            files={'image': (filename, source_bytes, mime)},
            data=form_data, params=query_params,
            headers={'Accept': 'image/png', 'ngrok-skip-browser-warning': 'true'},
            timeout=API_TIMEOUT, allow_redirects=False)
    except requests.RequestException as exc:
        elapsed = time.perf_counter() - t0
        logger.error(
            "Gọi Remove Background API THẤT BẠI (lỗi kết nối/timeout sau %.2fs) cho sản phẩm %s (nhóm: %s) tới %s: %s",
            elapsed, prod_label, group_label, api_url, exc, exc_info=True,
        )
        raise RuntimeError(f'Không kết nối được Remove Background API ({api_url}) sau {elapsed:.2f}s: {exc}') from exc

    elapsed = time.perf_counter() - t0

    if r.status_code != 200:
        detail = format_api_validation_error(r)
        message = (f'Remove Background HTTP {r.status_code} sau {elapsed:.2f}s tại {api_url}; '
                   f'sản phẩm={prod_label}, nhóm={group_label}, product_type={product_type!r}, '
                   f'image={filename!r}, bytes={len(source_bytes)}, transport={product_type_location}; '
                   f'API detail: {detail}')
        if r.status_code == 422:
            message += (' | HTTP 422: kiểm tra loc=body.image, '
                        'loc=body.product_type, enum product_type hoặc schema API.')
        logger.error(
            "Gọi Remove Background API THẤT BẠI (HTTP %d sau %.2fs) cho sản phẩm %s (nhóm: %s, product_type: %s) tại %s | Detail: %s",
            r.status_code, elapsed, prod_label, group_label, product_type, api_url, detail,
        )
        raise RuntimeError(message)

    if not r.content.startswith(b'\x89PNG\r\n\x1a\n'):
        logger.error(
            "Gọi Remove Background API THẤT BẠI: Response không phải ảnh PNG thật (nhận được %d bytes, Content-Type: %s) cho sản phẩm %s sau %.2fs tại %s",
            len(r.content), r.headers.get('Content-Type', ''), prod_label, elapsed, api_url,
        )
        raise ValueError('API không trả PNG thật; không được ghép ảnh gốc hoặc HTML')

    if len(r.content) > MAX_API_PNG_BYTES:
        logger.error(
            "Gọi Remove Background API THẤT BẠI: Ảnh PNG từ API vượt quá giới hạn 25MB (%d bytes) cho sản phẩm %s",
            len(r.content), prod_label,
        )
        raise ValueError('API PNG vượt giới hạn 25MB')

    with Image.open(BytesIO(r.content)) as image:
        if image.format != 'PNG' or ('A' not in image.getbands() and 'transparency' not in image.info):
            logger.error(
                "Gọi Remove Background API THẤT BẠI: Ảnh PNG trả về không có kênh Alpha (transparency) cho sản phẩm %s sau %.2fs tại %s",
                prod_label, elapsed, api_url,
            )
            raise ValueError('PNG của API không có alpha')
        cutout = image.convert('RGBA')
        cutout.load()

    hist = cutout.getchannel('A').histogram()
    total = cutout.width * cutout.height
    if sum(hist[32:]) < max(1, int(total * 0.005)) or sum(hist[:24]) < max(1, int(total * 0.005)):
        logger.error(
            "Gọi Remove Background API THẤT BẠI: Ảnh PNG trả về thiếu vùng sản phẩm rõ hoặc thiếu nền trong suốt cho sản phẩm %s sau %.2fs tại %s",
            prod_label, elapsed, api_url,
        )
        raise ValueError('PNG API thiếu vùng sản phẩm rõ hoặc nền trong suốt')

    logger.info(
        "Gọi Remove Background API THÀNH CÔNG cho sản phẩm %s (nhóm: %s, product_type: %s) tại %s trong %.2fs: "
        "nhận ảnh PNG RGBA (%dx%d, %d bytes, alpha channel hợp lệ)",
        prod_label, group_label, product_type, api_url, elapsed, cutout.width, cutout.height, len(r.content),
    )
    return cutout

# ===== CELL 159 =====
# F.4 — Composite đúng tỷ lệ silhouette, không tự xóa nền bằng Pillow.
# Pillow CHỈ dùng alpha đã được API tạo ra để ghép ảnh.
def composite_api_cutout(room,cutout,placement):
    bbox=(placement or {}).get('bbox_px')
    if not isinstance(bbox,dict):raise ValueError('Không có bounding box hợp lệ')
    x1,y1,x2,y2=(int(bbox[k]) for k in ('x1','y1','x2','y2'))
    W,H=room.size
    if not (0<=x1<x2<=W and 0<=y1<y2<=H):
        raise ValueError('Bounding box nằm ngoài ảnh')
    a=cutout.getchannel('A')
    content_bbox=a.point(lambda p:255 if p>8 else 0).getbbox()
    if not content_bbox:raise ValueError('Cutout không còn vật thể hiển thị')
    obj=cutout.crop(content_bbox)
    scale=min((x2-x1)/obj.width,(y2-y1)/obj.height)
    target_size=(max(1,round(obj.width*scale)),max(1,round(obj.height*scale)))
    obj=obj.resize(target_size,Image.Resampling.LANCZOS)
    px=x1+((x2-x1)-obj.width)//2
    surface=placement.get('surface')
    py=y1 if surface=='ceiling' else y2-obj.height if surface in ('floor','table') else y1+((y2-y1)-obj.height)//2
    out=room.convert('RGBA').copy()
    out.alpha_composite(obj,(px,py))
    return out,{'x1':px,'y1':py,'x2':px+obj.width,'y2':py+obj.height}


def boxes_intersect(a,b):
    def unpack(x):
        if isinstance(x,dict):return [float(x[k]) for k in ('x1','y1','x2','y2')]
        return list(map(float,x))
    ax1,ay1,ax2,ay2=unpack(a);bx1,by1,bx2,by2=unpack(b)
    return max(ax1,bx1)<min(ax2,bx2) and max(ay1,by1)<min(ay2,by2)


def validate_tv_cutout_scale(planned_bbox, rendered_bbox, image_size):
    """Validate the TV's *actual API-derived alpha silhouette* before committing.

    Raises ValueError on an undersized/malformed screen. No image fallback.
    """
    W,H=map(int,image_size)
    bbox_width=int(planned_bbox['x2'])-int(planned_bbox['x1'])
    drawn_width=int(rendered_bbox['x2'])-int(rendered_bbox['x1'])
    drawn_height=int(rendered_bbox['y2'])-int(rendered_bbox['y1'])
    aspect=drawn_width/max(1,drawn_height)
    p=TV_SCALE_POLICY
    if bbox_width<=0 or not (p['min_alpha_silhouette_aspect_ratio'] <= aspect <=
                            p['max_alpha_silhouette_aspect_ratio']):
        raise ValueError(f'TV cutout API có tỷ lệ không hợp lệ: {aspect:.3f}')
    if drawn_width < p['min_rendered_fraction_of_placement_width']*bbox_width:
        raise ValueError(f'TV hiển thị quá nhỏ: {drawn_width}px / bbox {bbox_width}px')
    # V8.11: the intended TV footprint is defined PER SELECTED WALL PLANE.
    # A narrow side-wall TV can validly occupy <22% of the overall photo.
    if drawn_width < 12 or drawn_height < 8:
        raise ValueError('TV cutout quá nhỏ để xem xét hiển thị')
    return {'placement_width_px':bbox_width,
            'rendered_width_px':drawn_width,
            'rendered_width_image_ratio':round(drawn_width/max(1,W),4),
            'rendered_aspect_ratio':round(aspect,3)}

# ===== CELL 161 =====
# F.4a — V8.8: PHOTOMETRIC MATCHING; chỉ xử lý cutout hợp lệ đã trả từ ngrok.
import numpy as np
from PIL import Image, ImageDraw

LIGHTING_POLICY_V88 = {
    'enabled': True,
    # Phòng quá tối/quá sáng mới cân bằng phơi sáng, tránh thay đổi ảnh bình thường.
    'auto_correct_room_exposure': True,
    'dark_scene_median_threshold': 68.0,     # [0,255]
    'bright_scene_median_threshold': 213.0,
    'target_dark_scene_median': 87.0,
    'target_bright_scene_median': 198.0,
    'room_exposure_gain_min': 0.91,
    'room_exposure_gain_max': 1.16,
    # Matching dựa vào ĐỘ CHÊNH ÁNH SÁNG giữa vùng đặt và cùng bề mặt,
    # KHÔNG ép màu trắng tường vào màu sắc vốn có của sản phẩm.
    'local_ring_fraction_of_bbox': 0.28,
    'local_ring_min_px': 16,
    'min_local_sample_pixels': 60,
    'object_gain_min': 0.74,
    'object_gain_max': 1.24,
    'local_vs_surface_strength': 0.72,
    'ambient_strength': 0.26,
    'neutral_scene_luma': 165.0,
    'object_contrast_min': 0.92,
    'object_contrast_max': 1.08,
    'color_cast_strength': 0.16,
    'max_color_channel_bias': 0.055,
    'tv_relighter_strength': 0.50,    # giữ nội dung/màu màn hình TV
    'debug': True,
}


def _v88_luma(rgb):
    return rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722


def _v88_robust_luma(rgb, mask=None):
    y = _v88_luma(rgb)
    values = y[np.asarray(mask, bool)] if mask is not None else y.reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError('Không đủ điểm ảnh để đo ánh sáng')
    p10, p25, p50, p75, p90 = np.percentile(values, (10, 25, 50, 75, 90))
    # Phần sáng trung tâm ít bị ảnh hưởng bởi bóng đổ/đèn quá chói.
    trimmed = values[(values >= p10) & (values <= p90)]
    return {'median': float(p50), 'mean_trimmed': float(trimmed.mean() if trimmed.size else p50),
            'p10': float(p10), 'p25': float(p25), 'p75': float(p75), 'p90': float(p90),
            'count': int(values.size)}


def v88_analyze_room_and_prepare_render(room, job_dir=None, policy=None):
    """Phân tích RGB đầu vào, tạo ảnh nền chỉ dành cho RENDER; không chỉnh ảnh infer."""
    p = policy or LIGHTING_POLICY_V88
    src = room.convert('RGB')
    before = np.asarray(src, dtype=np.float32)
    before_stats = _v88_robust_luma(before)
    med = before_stats['median']
    gain = 1.0
    if p['enabled'] and p['auto_correct_room_exposure']:
        if med < p['dark_scene_median_threshold']:
            gain = min(p['room_exposure_gain_max'], max(1.0, p['target_dark_scene_median'] / max(1, med)))
        elif med > p['bright_scene_median_threshold']:
            gain = max(p['room_exposure_gain_min'], min(1.0, p['target_bright_scene_median'] / max(1, med)))
    corrected = np.uint8(np.clip(np.rint(before * gain), 0, 255))
    scene_render = Image.fromarray(corrected, 'RGB').convert('RGBA')
    after_stats = _v88_robust_luma(corrected.astype(np.float32))
    record = {'mode': 'luminance_rgb_2d', 'room_exposure_gain': round(float(gain), 4),
              'room_corrected': bool(abs(gain - 1) > 1e-4),
              'before': before_stats, 'after': after_stats,
              'disclaimer': 'Độ sáng tương đối từ RGB; không suy ra cường độ đèn thực tế.'}
    if job_dir is not None and p.get('debug', False):
        path = Path(job_dir)
        path.mkdir(parents=True, exist_ok=True)
        # Luma debug biểu diễn mức sáng, không dùng lại làm nền render.
        intensity = np.uint8(np.clip(_v88_luma(before), 0, 255))
        Image.fromarray(intensity, 'L').convert('RGB').save(path / 'scene_lighting_debug.png')
        scene_render.convert('RGB').save(path / 'room_lighting_preview.png')
    return scene_render, record

# ===== CELL 163 =====
# F.4a.2 — Local ambient estimation + safe PNG relighting.
def _v88_surface_mask(masks, surface, shape):
    keys = {'wall': ('wall_front', 'wall'), 'floor': ('floor',),
            'ceiling': ('ceiling',), 'table': ('table',), 'shelf': ('table',)}.get(surface, (surface,))
    for key in keys:
        candidate = (masks or {}).get(key)
        if candidate is not None:
            arr = np.asarray(candidate, dtype=bool)
            if arr.shape == shape and int(arr.sum()) >= 100:
                return arr, key
    return np.ones(shape, dtype=bool), 'full_scene_no_surface_mask'


def v88_relight_api_cutout(cutout, scene_reference, placement, masks=None,
                            group_name=None, policy=None):
    """Khớp chiếu sáng cho PNG trong suốt do API trả về; alpha bất biến.

    Sample: vành ngoài bbox trên cùng bề mặt. Reference: median của bề mặt.
    Không tự tạo alpha/cutout, không tự thay ảnh khác nếu lỗi.
    """
    p = policy or LIGHTING_POLICY_V88
    if cutout.mode != 'RGBA':
        raise ValueError('V8.8 chỉ nhận cutout RGBA hợp lệ từ API')
    alpha = np.asarray(cutout.getchannel('A'))
    if not np.any(alpha < 24) or not np.any(alpha >= 32):
        raise ValueError('Cutout không chứa cả pixel trong suốt và vật thể')
    bbox = (placement or {}).get('bbox_px')
    if not isinstance(bbox, dict):
        raise ValueError('Không có bbox Placement để đo ánh sáng tại chỗ')
    W, H = scene_reference.size
    x1,y1,x2,y2 = (int(bbox[k]) for k in ('x1','y1','x2','y2'))
    if not (0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H):
        raise ValueError('BBox ngoài ảnh khi đo ánh sáng')
    rgb = np.asarray(scene_reference.convert('RGB'), np.float32)
    surface_mask, mask_source = _v88_surface_mask(masks, placement.get('surface'), (H,W))
    reference = _v88_robust_luma(rgb, surface_mask)
    width, height = x2-x1, y2-y1
    pad = max(p['local_ring_min_px'], round(max(width,height) * p['local_ring_fraction_of_bbox']))
    ax1,ay1,ax2,ay2 = max(0,x1-pad),max(0,y1-pad),min(W,x2+pad),min(H,y2+pad)
    local_mask = np.zeros((H,W),bool)
    local_mask[ay1:ay2,ax1:ax2] = True
    local_mask[y1:y2,x1:x2] = False
    local_mask &= surface_mask
    count = int(local_mask.sum())
    sampling_mode = 'bbox_ring_same_surface'
    if count < p['min_local_sample_pixels']:
        # Nếu không đủ vùng vòng bao, dùng đúng bề mặt chung thay vì đo nhầm nền khác.
        local_mask = surface_mask
        sampling_mode = 'surface_reference_insufficient_ring'
    local = _v88_robust_luma(rgb, local_mask)
    luma_local, luma_ref = local['mean_trimmed'],reference['mean_trimmed']
    exposure_ratio = np.clip(luma_local / max(luma_ref, 15.0), 0.25, 4.0)
    ambient_ratio = np.clip(luma_local / p['neutral_scene_luma'], 0.3, 3.0)
    gain_raw = (exposure_ratio ** p['local_vs_surface_strength'] *
                ambient_ratio ** p['ambient_strength'])
    gain = float(np.clip(gain_raw, p['object_gain_min'], p['object_gain_max']))
    contrast_raw = np.sqrt(max(1, local['p75']-local['p25']) /
                           max(1, reference['p75']-reference['p25']))
    contrast = float(np.clip(1 + (contrast_raw-1)*0.12,
                             p['object_contrast_min'],p['object_contrast_max']))
    # TV cần giữ nội dung ảnh màn hình, giảm mức hiệu chỉnh.
    strength = float(p['tv_relighter_strength']) if group_name == 'TV' else 1.0
    gain = 1 + (gain-1)*strength
    contrast = 1 + (contrast-1)*strength
    # Color cast dựa trên tỷ lệ vùng / bề mặt, KHÔNG ép màu vật thể giống màu tường.
    local_mean = rgb[local_mask].mean(axis=0)
    surface_mean = rgb[surface_mask].mean(axis=0)
    local_chroma = local_mean / max(1.0, float(local_mean.mean()))
    reference_chroma = surface_mean / max(1.0, float(surface_mean.mean()))
    tint_ratio = local_chroma / np.maximum(0.05, reference_chroma)
    color_gain = np.clip(1 + p['color_cast_strength']*strength*(tint_ratio-1),
                         1-p['max_color_channel_bias'],1+p['max_color_channel_bias'])
    src = np.array(cutout, dtype=np.uint8, copy=True)
    before = _v88_robust_luma(src[...,:3].astype(np.float32),alpha>=32)
    if p['enabled']:
        tuned = src[...,:3].astype(np.float32)
        tuned = (tuned-127.5)*contrast + 127.5
        tuned = tuned * gain * color_gain.reshape((1,1,3))
        src[...,:3] = np.uint8(np.clip(np.rint(tuned),0,255))
    output = Image.fromarray(src,'RGBA')
    if not np.array_equal(np.asarray(output.getchannel('A')),alpha):
        raise AssertionError('Ánh sáng không được thay đổi alpha của cutout API')
    after = _v88_robust_luma(src[...,:3].astype(np.float32),alpha>=32)
    diag = {'surface':placement.get('surface'),'sampling_mode':sampling_mode,
            'mask_source':mask_source,'local_sample_pixels':int(local_mask.sum()),
            'local_luma':round(float(luma_local),2),
            'surface_reference_luma':round(float(luma_ref),2),
            'product_luma_before':round(before['median'],2),
            'product_luma_after':round(after['median'],2),
            'brightness_gain':round(float(gain),4),
            'contrast_gain':round(float(contrast),4),
            'rgb_cast_gain':[round(float(v),4) for v in color_gain],
            'enabled':bool(p['enabled']),
            'alpha_preserved': True}
    return output,diag

# ===== CELL 165 =====
# F.4a.2b — V8.20: photometric target = 80% scene illumination, dynamic per input room.
# Works on API-returned RGBA only. Call AFTER V8.8 lighting and BEFORE compositing.
SCENE_RELATIVE_OBJECT_BRIGHTNESS_V820 = {
    'enabled': True,
    'relative_brightness': 0.80,      # muc tieu: 80% do sang anh phong tai noi dat
    'local_scene_weight': 0.55,       # 55% vung dat, 45% toan anh
    'local_padding_fraction': 0.25,
    'local_padding_min_px': 12,
    'min_local_sample_pixels': 64,
    'alpha_visible_threshold': 64,    # do sang chi do tren phan vat the co alpha ro
    'min_rgb_exposure_gain': 0.25,
    'max_rgb_exposure_gain': 2.00,
    'max_samples_for_exposure_search': 12000,
    'exposure_binary_steps': 15,
}


def v820_match_object_to_scene_minus_20(api_lit_cutout, scene_reference,
                                         placement, masks=None, group_name=None,
                                         policy=None):
    """Relight API cutout using the uploaded room's brightness, keep alpha exactly.

    Dynamic reference = mix of global room luminance and local placement surround.
    Target *visible-object average luminance* = 80% of dynamic reference.
    Exposure is bounded; when clipping/limits prevent 80%, diagnostics report it.
    This stage never creates a cutout or substitutes a product on API failure.
    """
    p = policy or SCENE_RELATIVE_OBJECT_BRIGHTNESS_V820
    if api_lit_cutout.mode != 'RGBA':
        raise ValueError('V8.20 chi nhan anh RGBA da duoc API tach nen xac thuc')
    alpha = np.asarray(api_lit_cutout.getchannel('A'))
    threshold = int(p['alpha_visible_threshold'])
    visible = alpha >= threshold
    if not visible.any() or not (alpha < 24).any():
        raise ValueError('Anh API khong co du pixel vat the va nen trong suot')

    bbox = (placement or {}).get('bbox_px')
    if not isinstance(bbox, dict):
        raise ValueError('V8.20 can bbox hop le de phan tich anh phong')
    W, H = scene_reference.size
    x1, y1, x2, y2 = (int(bbox[k]) for k in ('x1','y1','x2','y2'))
    if not (0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H):
        raise ValueError('BBox nam ngoai anh khi phan tich do sang')

    scene_rgb = np.asarray(scene_reference.convert('RGB'), dtype=np.float32)
    global_stats = _v88_robust_luma(scene_rgb)
    surface_mask, surface_name = _v88_surface_mask(
        masks, (placement or {}).get('surface'), (H, W))

    pad = max(int(p['local_padding_min_px']),
              int(round(max(x2-x1, y2-y1) * float(p['local_padding_fraction']))))
    rx1, ry1 = max(0, x1-pad), max(0, y1-pad)
    rx2, ry2 = min(W, x2+pad), min(H, y2+pad)
    local_mask = np.zeros((H,W), dtype=bool)
    local_mask[ry1:ry2, rx1:rx2] = True
    local_mask[y1:y2, x1:x2] = False
    local_mask &= surface_mask
    sample_source = 'bbox_ring_same_surface'
    if int(local_mask.sum()) < int(p['min_local_sample_pixels']):
        # When a side wall has a small mask, measure the uncovered location itself.
        local_mask = np.zeros((H,W), dtype=bool)
        local_mask[y1:y2, x1:x2] = True
        local_mask &= surface_mask
        sample_source = 'bbox_area_same_surface'
    if int(local_mask.sum()) < int(p['min_local_sample_pixels']):
        local_mask = surface_mask
        sample_source = 'surface_fallback'
    if int(local_mask.sum()) < int(p['min_local_sample_pixels']):
        # Last resort is global scene brightness, not a substitute image/cutout.
        local_luma = float(global_stats['mean_trimmed'])
        sample_source = 'global_luma_only'
    else:
        local_luma = float(_v88_robust_luma(scene_rgb, local_mask)['mean_trimmed'])

    global_luma = float(global_stats['mean_trimmed'])
    local_weight = float(np.clip(p['local_scene_weight'], 0, 1))
    ref_luma = global_luma*(1-local_weight) + local_luma*local_weight
    target_luma = float(ref_luma * float(p['relative_brightness']))
    pixels = np.asarray(api_lit_cutout, dtype=np.uint8).copy()
    original_alpha = pixels[...,3].copy()
    rgb_float = pixels[..., :3].astype(np.float32)
    sample_pixels = rgb_float[visible]
    max_samples = int(p['max_samples_for_exposure_search'])
    if len(sample_pixels) > max_samples:
        sample_pixels = sample_pixels[::max(1, len(sample_pixels)//max_samples)][:max_samples]

    def sample_luma_at_gain(gain):
        samples = np.minimum(sample_pixels * gain, 255.0)
        return float(_v88_luma(samples).mean())

    before_luma = sample_luma_at_gain(1.0)
    low = float(p['min_rgb_exposure_gain'])
    high = float(p['max_rgb_exposure_gain'])
    if low <= 0 or high < low:
        raise ValueError('Cau hinh do sang V8.20 khong hop le')
    if not p['enabled']:
        exposure_gain = 1.0
    elif target_luma <= sample_luma_at_gain(low):
        exposure_gain = low
    elif target_luma >= sample_luma_at_gain(high):
        exposure_gain = high
    else:
        left, right = low, high
        for _ in range(int(p['exposure_binary_steps'])):
            mid = (left + right)/2
            if sample_luma_at_gain(mid) < target_luma:
                left = mid
            else:
                right = mid
        exposure_gain = (left + right)/2

    if p['enabled']:
        # Color ratios & original alpha preserved. No local segmentation/fallback.
        corrected = np.uint8(np.clip(np.rint(rgb_float*exposure_gain), 0, 255))
        pixels[..., :3] = corrected
    if not np.array_equal(pixels[...,3], original_alpha):
        raise AssertionError('Lighting stage changed API alpha channel')
    output = Image.fromarray(pixels, mode='RGBA')
    after_luma = sample_luma_at_gain(exposure_gain)
    actual_ratio = after_luma/max(ref_luma, 1e-6)
    diagnostics = {
        'lighting_method': 'V8.20_dynamic_scene_minus_20_percent',
        'group_name': str(group_name or ''),
        'scene_global_luma': round(global_luma, 2),
        'scene_local_luma': round(local_luma, 2),
        'scene_reference_luma': round(ref_luma, 2),
        'local_scene_weight': round(local_weight, 3),
        'sample_source': sample_source,
        'surface_mask_source': surface_name,
        'target_brightness_fraction_of_scene': float(p['relative_brightness']),
        'target_product_luma': round(target_luma, 2),
        'product_luma_before_final_adjustment': round(before_luma, 2),
        'product_luma_after_final_adjustment': round(after_luma, 2),
        'actual_fraction_of_scene': round(actual_ratio, 4),
        'final_brightness_factor': round(float(exposure_gain), 4),
        'exposure_gain_limited': bool(p['enabled'] and
            ((exposure_gain <= low+1e-4) or (exposure_gain >= high-1e-4))),
        'alpha_preserved_after_final_brightness': True,
        'enabled': bool(p['enabled']),
    }
    return output, diagnostics

# ===== CELL 167 =====
# F.4a.3 — Photometric diagnostics.
def v88_save_lighting_placements_debug(base_scene, items, path):
    """Debug bbox + hệ số sáng; không sửa ảnh render thành phẩm."""
    canvas=base_scene.convert('RGB').copy()
    pen=ImageDraw.Draw(canvas)
    for row in items:
        b = row.get('bbox_px')
        if not b: continue
        coords=tuple(int(b[k]) for k in ('x1','y1','x2','y2'))
        ok=row.get('status')=='rendered'
        pen.rectangle(coords,outline=(34,198,85) if ok else (230,70,50),width=3)
        light=row.get('lighting') or {}
        label=f"{row.get('group_name','')} | gain {light.get('brightness_gain','?')}"
        pen.text((coords[0],max(0,coords[1]-14)),label,fill=(0,100,0) if ok else (170,30,30),stroke_width=1,stroke_fill=(255,255,255))
    canvas.save(path)
    return path

# ===== CELL 169 =====
# F.4b.1 - V8.10: Ti le MAY LANH / TV thay doi theo KICH THUOC TV.
# TV nho: AC co the rong gan bang TV. TV to: AC chiem ti le nho hon.
# Tham so chi dieu chinh ty le HIEN THI trong anh 2D, KHONG la kich thuoc lap dat thuc te.
# De kich thuoc vat ly dung, can SKU cm do duoc va ti le camera/tuong duoc hieu chuan.
TV_AC_PAIR_POLICY_V810 = {
    'enabled': True,
    # (duong cheo TV inch, chieu rong AC / chieu rong TV) - noi suy tuyen tinh.
    # TV 32-43 inch: AC gan bang TV. TV 65-85 inch: AC nho hon theo ty le.
    'tv_inches_to_ac_width_ratio': [
        (32, 1.02), (43, 0.96), (50, 0.87), (55, 0.80),
        (65, 0.68), (75, 0.59), (85, 0.52),
    ],
    'reference_aircon_width_cm': 95.0,
    # Neu co so do AC SKU, dieu chinh nhe duong cong neu dai hon/ngan hon 95cm.
    'aircon_sku_width_correction_min': 0.85,
    'aircon_sku_width_correction_max': 1.17,
    # Depth Anything V2 la DEPTH TUONG DOI, chi duoc dieu chinh nhe giua 2 diem tren cung tuong.
    'relative_depth_ratio_min': 0.90,
    'relative_depth_ratio_max': 1.10,
    'aircon_to_tv_ratio_multiplier': 1.00,
    'final_ratio_min': 0.43,
    'final_ratio_max': 1.08,
    # Cho phep sua bbox du kien theo ty le cap, NHUNG kiem tra lai va cham + mask + khoang ho.
    'min_adjustment_from_planned': 0.48,
    'max_adjustment_from_planned': 2.40,
    'max_rendered_ratio_error': 0.055,
    'min_valid_physical_ratio': 0.25,
    'max_valid_physical_ratio': 2.25,
    'debug': True,
}
# Ke thua ten dict cu de cac ham F.5 hien tai tiep tuc dung binh thuong.
TV_AC_PAIR_POLICY_V89 = TV_AC_PAIR_POLICY_V810


def v810_interpolated_ratio_for_tv(tv_inch):
    """Noi suy tuyen tinh, khong buoc anh TV/AC ve 1 ty le co dinh."""
    pts=TV_AC_PAIR_POLICY_V810['tv_inches_to_ac_width_ratio']
    x=float(tv_inch)
    if x<=pts[0][0]: return float(pts[0][1])
    if x>=pts[-1][0]: return float(pts[-1][1])
    for (x0,y0),(x1,y1) in zip(pts[:-1],pts[1:]):
        if x<=x1:
            return float(y0+(y1-y0)*(x-x0)/(x1-x0))
    raise RuntimeError('TV scale ratio points are invalid')


def v810_adaptive_base_ratio(tv_inch, aircon_width_cm):
    p=TV_AC_PAIR_POLICY_V810
    ref=v810_interpolated_ratio_for_tv(tv_inch)
    correction=float(aircon_width_cm)/float(p['reference_aircon_width_cm'])
    correction=max(p['aircon_sku_width_correction_min'],
                   min(p['aircon_sku_width_correction_max'],correction))
    value=ref*correction
    return max(p['final_ratio_min'],min(p['final_ratio_max'],value))


def v89_pair_selected_products(stage1_result):
    """Chi ghep cap tu dung hai SKU da duoc CHON o Stage 1.

    V8.10 khong doi san pham luc render, khong bypass API va khong can TV/AC
    phai render thanh cong cung luc. AC duoc xu ly sau TV neu ca hai co trong goi y.
    """
    if not TV_AC_PAIR_POLICY_V810['enabled']:
        return None
    profiles={}
    for entry in stage1_result.get('recommendations',[]):
        group=entry.get('group_name')
        if group in ('TV','Máy điều hòa') and group not in profiles:
            p=resolve_product_profile(group,entry.get('selected_product') or {})
            if p is not None and p.get('surface')=='wall':
                profiles[group]=p
    if len(profiles)!=2: return None
    tv=profiles['TV']; ac=profiles['Máy điều hòa']
    diagonal,diagonal_source=tv_diagonal_inches(tv.get('product_name',''))
    tv_w_cm=float(diagonal)*2.54*16.0/(16.0**2+9.0**2)**0.5
    ac_w_cm=float((ac.get('size_cm') or {}).get('width') or 95.0)
    if not (30 <= ac_w_cm <= 200 and 40 <= tv_w_cm <= 230): return None
    physical_ratio=ac_w_cm/tv_w_cm
    if not (TV_AC_PAIR_POLICY_V810['min_valid_physical_ratio'] <= physical_ratio <=
            TV_AC_PAIR_POLICY_V810['max_valid_physical_ratio']):
        return None
    adaptive=v810_adaptive_base_ratio(diagonal,ac_w_cm)
    return {
        'active':True,
        'tv_diagonal_inch':int(diagonal),
        'tv_size_source':diagonal_source,
        'tv_product_width_cm':round(tv_w_cm,2),
        'aircon_product_width_cm':round(ac_w_cm,2),
        'aircon_size_source':ac.get('size_source','category_prior_unverified'),
        'nominal_aircon_to_tv_ratio':round(physical_ratio,5),
        'adaptive_aircon_to_tv_ratio':round(adaptive,5),
        'adaptive_ratio_before_depth':round(adaptive,5),
        'measurement_verified':bool(
            ac.get('size_source')=='sku_verified_input' and
            diagonal_source=='sku_name_explicit_inch'),
        'basis':'TV diagonal adaptive ratio + AC SKU width + clamped relative depth',
        'is_metric_ground_truth':False,
    }


def v89_order_render_items(stage1_result, pair):
    """Xu ly TV truoc AC de co moc pixel, giu NGUYEN danh sach va rank Stage 1."""
    items=list(enumerate(stage1_result.get('recommendations',[]),1))
    if not pair:return items
    ti=next((i for i,(_,r) in enumerate(items) if r.get('group_name')=='TV'),None)
    ai=next((i for i,(_,r) in enumerate(items) if r.get('group_name')=='Máy điều hòa'),None)
    if ti is not None and ai is not None and ai<ti:
        items[ai],items[ti]=items[ti],items[ai]
    return items

# ===== CELL 170 =====
# F.4b.2 - V8.9: Khong de chieu CAO bbox ep vat the bi nho hon chieu RONG da tinh.
# Chi dieu chinh PNG do API tach nen tra ve; khong fallback hay tao alpha tu anh goc.
def _v89_expected_ac_tv_ratio(pair,a_meta,t_meta):
    """Adaptive width AC/TV; adjust only slightly for relative depth."""
    if not pair or not pair.get('active'):
        raise ValueError('No active TV/AC pair in Stage 1')
    p=TV_AC_PAIR_POLICY_V810
    base=float(pair['adaptive_aircon_to_tv_ratio'])
    a_factor=float((a_meta or {}).get('relative_depth_factor',1.0) or 1.0)
    t_factor=float((t_meta or {}).get('relative_depth_factor',1.0) or 1.0)
    if not (0.5<=a_factor<=2.0 and 0.5<=t_factor<=2.0):
        raise ValueError('Invalid relative depth correction for paired wall products')
    relative_depth=a_factor/t_factor
    relative_depth=max(p['relative_depth_ratio_min'],
                       min(p['relative_depth_ratio_max'],relative_depth))
    mul=float(p['aircon_to_tv_ratio_multiplier'])
    if not 0.80<=mul<=1.20:
        raise ValueError('aircon_to_tv_ratio_multiplier must be between 0.8 and 1.2')
    result=base*relative_depth*mul
    return max(p['final_ratio_min'],min(p['final_ratio_max'],result))


def v89_target_width_for_pair(group, placement, pair, committed):
    """Khoi tao kich thuoc tu bbox, sau do dong bo khi da co vat the con lai.

    Danh sach goc khong doi. Khong can ca 2 API phai thanh cong moi render 1 vat.
    """
    box=placement['bbox_px']
    planned=int(box['x2']-box['x1'])
    info={'paired_with':None,'planned_width_px':planned,
          'target_width_px':planned,'scale_method':'v8.10_adaptive_tv_inches_ac_sku_depth'}
    if pair is None or not pair.get('active') or group not in ('TV','Máy điều hòa'):
        return planned,info
    counterpart='TV' if group=='Máy điều hòa' else 'Máy điều hòa'
    other=committed.get(counterpart)
    if not other:
        info['paired_with']='awaiting_other_valid_cutout'
        return planned,info
    # Chua tinh pair neu mot vat the khong duoc ghep bang API hoac khong tren wall.
    if other.get('surface')!='wall' or placement.get('surface')!='wall':
        return planned,info
    ac_meta=(placement.get('scale_vs_scene') if group=='Máy điều hòa' else other.get('scale_vs_scene'))
    tv_meta=(placement.get('scale_vs_scene') if group=='TV' else other.get('scale_vs_scene'))
    ratio=_v89_expected_ac_tv_ratio(pair,ac_meta,tv_meta)
    other_width=int(other['rendered_bbox']['x2']-other['rendered_bbox']['x1'])
    target=other_width*ratio if group=='Máy điều hòa' else other_width/ratio
    low=float(TV_AC_PAIR_POLICY_V89['min_adjustment_from_planned'])
    high=float(TV_AC_PAIR_POLICY_V89['max_adjustment_from_planned'])
    correction=target/max(1,planned)
    if not low<=correction<=high:
        raise ValueError(f'Paired scale would need an unsafe {correction:.2f}x resize')
    target=max(1,int(round(target)))
    info.update(paired_with=counterpart,target_width_px=target,
                expected_ac_to_tv_ratio=round(ratio,4),
                tv_diagonal_inch=pair['tv_diagonal_inch'],
                adaptive_ratio_before_depth=pair['adaptive_aircon_to_tv_ratio'],
                actual_adjustment_factor=round(correction,4),
                other_rendered_width_px=other_width)
    return target,info


def v89_composite_width_first(room,cutout,placement,target_width_px):
    """Ghep theo WIDTH cua SKU; kiem tra *bbox thuc* sau do, khong fit 2 chieu.

    Alpha bat buoc phai den tu endpoint, do require_api_cutout xac nhan.
    Wall/ceiling/floor, chong chong lan duoc kiem tra lai truoc commit.
    """
    if cutout.mode!='RGBA':
        raise ValueError('Expected validated RGBA cutout from background-removal API')
    a=cutout.getchannel('A')
    content=a.point(lambda value:255 if value>8 else 0).getbbox()
    if not content:raise ValueError('Cutout from API is fully transparent')
    obj=cutout.crop(content)
    group=placement.get('group_name','')
    box=placement['bbox_px']
    x1,y1,x2,y2=[int(box[k]) for k in ('x1','y1','x2','y2')]
    bw=x2-x1;bh=y2-y1
    width=max(1,int(round(target_width_px)))
    height=max(1,int(round(width*obj.height/max(obj.width,1))))
    # Height is free to adjust because physical silhouette need not fit old prior.
    # Avoid pathological source PNG that would require huge vertical expansion.
    if height>max(bh*1.65, bh+14):
        raise ValueError('API cutout aspect requires excessive vertical expansion')
    item=obj.resize((width,height),Image.Resampling.LANCZOS)
    cx=(x1+x2)/2
    px=int(round(cx-width/2))
    if placement.get('surface')=='wall':
        # May lanh can giu canh tren de khong thay doi khoang ho tran.
        # TV giu nguyen tam bbox de bao toan cao do mat dat.
        py=y1 if group=='Máy điều hòa' else int(round((y1+y2-height)/2))
    else:
        py=y1 if placement.get('surface')=='ceiling' else y2-height
    W,H=room.size
    if px<0 or py<0 or px+width>W or py+height>H:
        raise ValueError('Width-first scaled PNG would exceed image boundary')
    out=room.convert('RGBA').copy()
    out.alpha_composite(item,(px,py))
    actual={'x1':px,'y1':py,'x2':px+width,'y2':py+height}
    return out,actual


def v89_validate_paired_render(group,rendered_bbox,placement,pair,committed):
    """Hard rule chi khi ca 2 co cutout API hop le va cung mat tuong."""
    if not pair or not pair.get('active') or group not in ('TV','Máy điều hòa'):
        return None
    counterpart='TV' if group=='Máy điều hòa' else 'Máy điều hòa'
    other=committed.get(counterpart)
    if not other or other.get('surface')!='wall' or placement.get('surface')!='wall':
        return {'status':'waiting_for_counterpart_valid_render'}
    ac_meta=(placement.get('scale_vs_scene') if group=='Máy điều hòa' else other.get('scale_vs_scene'))
    tv_meta=(placement.get('scale_vs_scene') if group=='TV' else other.get('scale_vs_scene'))
    ratio=_v89_expected_ac_tv_ratio(pair,ac_meta,tv_meta)
    cur_width=rendered_bbox['x2']-rendered_bbox['x1']
    other_width=other['rendered_bbox']['x2']-other['rendered_bbox']['x1']
    actual=(cur_width/other_width if group=='Máy điều hòa' else other_width/cur_width)
    err=abs(actual/ratio-1.0)
    if err>TV_AC_PAIR_POLICY_V89['max_rendered_ratio_error']:
        raise ValueError(f'Actual TV/AC ratio {actual:.3f} disagrees with expected {ratio:.3f} ({err:.1%})')
    return {'status':'valid_paired_scale','actual_aircon_to_tv_ratio':round(actual,4),
            'expected_aircon_to_tv_ratio':round(ratio,4),
            'relative_error':round(err,4),
            'physical_ratio_reference':pair['nominal_aircon_to_tv_ratio'],
            'adaptive_ratio_before_depth':pair['adaptive_aircon_to_tv_ratio'],
            'tv_diagonal_inch':pair['tv_diagonal_inch'],
            'depth_is_metric':False}

# ===== CELL 171 =====
# F.5 — Duyệt đúng danh sách Gợi ý của BƯỚC 1; không thử SKU khác và không cache.
# Chỉ đặt chỗ cho sản phẩm API xử lý THÀNH CÔNG (thất bại không giữ chỗ).
def stage2_render(stage1_result, output_base=OUTPUT_BASE, show_final=True):
    if 'recommendations' not in stage1_result or 'input_image' not in stage1_result:
        raise ValueError('Cần đầu ra stage1_recommend() từ BƯỚC 1')
    room=stage1_result['input_image'].convert('RGBA')
    job_dir=Path(output_base)/('render_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6])
    job_dir.mkdir(parents=True,exist_ok=False)
    # Không được suy diễn mặt sàn/tường nếu Mask2Former không chạy được.
    try:
        masks,segmentation_info=segment_room_surfaces(room.convert('RGB'))
    except Exception as exc:
        masks={}
        segmentation_info={'status':'unavailable','reason':f'{type(exc).__name__}: {exc}'}
    if segmentation_info.get('status')=='ok':
        # V8.11: TV wall candidates from semantic WALL + RGB junctions.
        # This does not change placement logic for AC or any other product.
        tv_meta=v811_prepare_tv_planes(room,masks)
        segmentation_info['tv_wall_planes_v811']=tv_meta
        print('[V8.12] TV wall planes:',tv_meta.get('planes',[]),
              '| corners:',tv_meta.get('corners_px',{}),
              '| confidence:',tv_meta.get('confidence'))
        if TV_WALL_FIRST_POLICY_V811['debug']:
            display(v811_save_tv_wall_debug(
                room,masks,job_dir/'tv_wall_planes_debug_v811.png'))
        ceiling_mode=('ceiling_and_floor' if _v86_has_visible_ceiling(masks,room.size)
                      else 'floor_only_no_ceiling')
        segmentation_info['height_reference_mode']=ceiling_mode
        if ceiling_mode=='floor_only_no_ceiling':
            print('[V8.6] Không thấy trần: TV/máy lạnh dựa trên mặt tường và ranh giới sàn; '
                  'quạt trần, đèn chùm vẫn yêu cầu trần thật.')
            try:
                debug_img=display_floor_wall_reference_v86(room,masks,
                    job_dir/'floor_wall_reference_debug.png')
                print('Đường xanh lá: mép trên tường nhìn thấy; cam: giao tường–sàn.')
                display(debug_img)
            except ValueError as exc:
                print('Chưa xác định được đường tường–sàn đủ tin cậy:',exc)
        wall_meta=segmentation_info.get('wall_planes',{})
        print('MẶT TƯỜNG NHẬN DIỆN:',wall_meta.get('method','unavailable'),
              '| giao tường:',list(wall_meta.get('corners',{})))
        if WALL_CORNER_VISUAL_POLICY_V87['debug']:
            try:
                corner_preview = display_wall_corner_debug_v87(
                    room, masks, wall_meta, job_dir/'wall_corner_debug_v87.png')
                print('V8.7 - Góc tường từ ảnh (tọa độ theo ảnh đầu vào):',
                      wall_meta.get('visual_corners', {}))
                display(corner_preview)
            except Exception as exc:
                print('Không thể hiển thị đường giao tường V8.7:', exc)
        if WALL_PLANE_POLICY_V83['debug_wall_planes']:
            try:
                plane_preview=display_wall_planes_debug(room,masks,job_dir/'wall_plane_debug.png')
                print('Màu xanh lá: tường chính; xanh dương/cam: tường bên; đỏ: vùng cấm góc tường.')
                display(plane_preview)
            except Exception as exc:
                print('Không tạo được ảnh debug mặt tường:',exc)
    if segmentation_info.get('status')!='ok':
        # BƯỚC 1 vẫn có danh sách gợi ý; BƯỚC 2 sẽ bỏ qua tất cả sản phẩm.
        print('Không có segmentation hợp lệ: chỉ xuất báo cáo, không render ảnh.')
    try:
        depth,depth_info=estimate_relative_depth(room.convert('RGB'))
    except Exception as exc:
        depth=None
        depth_info={'status':'unavailable','reason':f'{type(exc).__name__}: {exc}'}
    # V8.8: đo độ sáng ảnh phòng TRƯỚC ghép, chỉ chỉnh nhẹ ảnh render.
    # Mask2Former, Depth và Placement vẫn xử lý ảnh gốc để giữ kết quả ổn định.
    scene_render,scene_lighting = v88_analyze_room_and_prepare_render(room, job_dir)
    print('[V8.8] Độ sáng trung vị phòng:',round(scene_lighting['before']['median'],1),
          '| hệ số phơi sáng nền:',scene_lighting['room_exposure_gain'])
    detected=list(stage1_result.get('spatial_detections',[]))
    reserved=[];canvas=scene_render.copy();results=[]
    # V8.9: only pair the EXACT TV/AC SKUs chosen in Stage 1, if both mount on wall.
    tv_ac_pair=v89_pair_selected_products(stage1_result)
    committed_wall_pair={}
    if tv_ac_pair and TV_AC_PAIR_POLICY_V89['debug']:
        print('[V8.10] Adaptive TV/AC ratio (before depth):',tv_ac_pair['adaptive_aircon_to_tv_ratio'],
              '| dimensions cm:',tv_ac_pair['aircon_product_width_cm'],
              '/',tv_ac_pair['tv_product_width_cm'])
    for rank,reco in v89_order_render_items(stage1_result,tv_ac_pair):
        group=reco.get('placement_group') or reco['group_name'];product=reco['selected_product']
        rec={'rank':rank,'group_name':group,'api_group_name':reco['group_name'],
             'category_id':reco.get('category_id',''),'product_code':product.get('product_code',''),
             'product_id':product['product_id'],
             'product_name':product['product_name'],
             'image_url':product['image_url'],
             'product_type':resolve_remove_bg_product_type(reco,product),
             'status':'not_rendered','reason':'',
             'category_mapping':reco.get('category_mapping',{})}
        results.append(rec)
        # BẮT BUỘC xác minh category_id + group_name trước khi download/call API.
        verified_mapping=reco.get('category_mapping') or {}
        if not verified_mapping.get('renderable'):
            rec.update(status='unmapped_category',
                       reason=verified_mapping.get('reason','Thiếu category mapping an toàn'))
            continue
        restriction=api_product_placement_restriction(group,product)
        if restriction:
            rec.update(status='unsupported_product_surface',reason=restriction)
            continue
        if not rec['product_type']:
            rec['product_type'] = 'auto'
        if segmentation_info.get('status')!='ok':
            rec.update(status='no_placement',reason='Mask2Former không cung cấp surface masks')
            continue
        # Trước tiên tính placement từ các vật thể đã phát hiện + những sản phẩm thực sự ghép.
        try:
            profile=resolve_product_profile(group,product)
            if profile is None:
                rec.update(status='no_placement',reason='Không có placement profile')
                continue
            if (profile.get('surface')=='wall' and group!='TV' and
                    segmentation_info.get('wall_planes',{}).get('status')!='ok'):
                rec.update(status='no_wall_plane',
                    reason='Không xác định được mặt tường riêng biệt để treo sản phẩm')
                continue
            occupancy=detected+reserved
            placement=choose_placement_region(profile,masks,occupancy,room.size,depth_map=depth)
            if not placement.get('bbox_px'):
                rec.update(status='no_placement',
                           reason=str(placement.get('status','no_safe_region')) + ': ' +
                                  str(placement.get('reason','')),
                           failed_candidate_counts=(placement.get('failed_candidates',[]) or
                                                    placement.get('failed_candidate_counts',[])))
                if group=='TV' and not placement.get('bbox_px'):
                    rec['tv_wall_planes_evaluated']=placement.get('tv_wall_planes_evaluated',[])
                    print('[V8.11] TV không có vị trí:',rec['reason'],
                          '| reasons:',rec.get('failed_candidate_counts'))
                continue
            bbox=placement['bbox_px']
            if group=='TV' and TV_WALL_GEOMETRY_V812['enabled'] and TV_WALL_GEOMETRY_V812['debug']:
                display(v812_debug_tv_geometry(
                    room,masks,placement,job_dir/'tv_scale_perspective_debug_v812.png'))
            elif group=='TV' and TV_WALL_FIRST_POLICY_V811['debug']:
                display(v811_save_tv_wall_debug(
                    room,masks,job_dir/'tv_wall_selected_debug_v811.png',placement))
            if group in ('TV','Máy điều hòa'):
                rec['scale_diagnostics']=placement.get('scale_vs_scene',{})
                rec['planned_bbox_width_px']=bbox['x2']-bbox['x1']
                rec['reference_wall_span_px']=rec['scale_diagnostics'].get('front_wall_span_px')
                rec['sku_width_cm_estimate']=rec['scale_diagnostics'].get('product_width_cm_estimate')
            if any(boxes_intersect(bbox,d['bbox']) for d in occupancy if d.get('bbox') is not None):
                rec.update(status='placement_collision',reason='Bbox đè lên vật thể/ảnh đã ghép')
                continue
            if not product['image_url']:
                rec.update(status='missing_image_url',reason='SKU không có image_url')
                continue
            source_bytes,source_name,mime=download_selected_product_image(product['image_url'])
            # Bắt buộc gọi ngrok endpoint. Nếu lỗi => bắt exception => KHÔNG dán ảnh.
            cutout=require_api_cutout(source_bytes,source_name,mime,rec['product_type'])
            # V8.8: chỉ relight PNG đã xóa nền bằng API; giữ nguyên kênh alpha.
            # So ánh sáng quanh bbox với độ sáng của cùng bề mặt trong ảnh phòng.
            lit_cutout,lighting=v88_relight_api_cutout(
                cutout,scene_render,placement,masks=masks,group_name=group)
            # V8.20: do do sang anh phong + vung dat, muc tieu cutout thap hon 20%.
            # Ap dung cho MOI SKU da tach nen thanh cong, alpha tu API duoc giu nguyen.
            lit_cutout,final_brightness=v820_match_object_to_scene_minus_20(
                lit_cutout, scene_render, placement, masks=masks, group_name=group)
            lighting.update(final_brightness)
            if (group=='TV' and placement.get('surface')=='wall'
                    and TV_WALL_GEOMETRY_V812['enabled']):
                # TV only: API-validated cutout warped to the SINGLE chosen wall.
                # No SKU swap, no local segmentation/fallback.
                next_canvas,rendered_bbox,projection_diag=v812_composite_tv_cutout(
                    canvas,lit_cutout,placement,masks)
                rec['tv_projective_geometry']=projection_diag
            elif group in ('TV','Máy điều hòa') and placement.get('surface')=='wall':
                target_width,pair_info=v89_target_width_for_pair(
                    group,placement,tv_ac_pair,committed_wall_pair)
                placement['group_name']=group
                next_canvas,rendered_bbox=v89_composite_width_first(
                    canvas,lit_cutout,placement,target_width)
                rec['paired_scale_plan']=pair_info
            else:
                next_canvas,rendered_bbox=composite_api_cutout(canvas,lit_cutout,placement)
            # The actual rendered bbox can be wider than the planned bbox.
            # A collision is forbidden even when the original planned bbox was safe.
            if any(boxes_intersect(rendered_bbox,d['bbox']) for d in occupancy
                   if d.get('bbox') is not None):
                raise ValueError('Actual API cutout intersects detected/committed object')
            # Chỉ commit khi VẬT THỂ THỰC TẾ trả về từ API cũng nằm trên 1 mặt.
            # Bounding box sau alpha trim có thể khác bbox placement ban đầu.
            if placement.get('surface')=='wall':
                if group=='TV' and TV_WALL_GEOMETRY_V812['enabled']:
                    wall_ok,wall_diag=v812_tv_quad_check(
                        masks,room.size,placement['tv_projective_quad'],
                        placement['tv_wall_selection']['chosen_plane'])
                elif group=='TV':
                    wall_ok,wall_diag=v811_tv_plane_box_check(
                        masks,room.size,
                        [rendered_bbox[k] for k in ('x1','y1','x2','y2')],
                        plane_id=placement.get('tv_wall_selection',{}).get('chosen_plane'))
                else:
                    wall_ok,wall_diag=_wall_plane_box_check(masks,room.size,
                        [rendered_bbox[k] for k in ('x1','y1','x2','y2')],
                        threshold=(AIRCON_WALL_CLEARANCE_V84['wall_plane_coverage_min']
                                   if group=='Máy điều hòa' else None))
                if not wall_ok:
                    raise ValueError('Ảnh sau API vượt qua góc tường: '+wall_diag['reason'])
                rec['wall_plane']=wall_diag
            if group == 'TV':
                # Validate the *actual alpha silhouette* from the API, not just
                # the proposed geometric bbox. A tiny or portrait product photo
                # is rejected, never replaced with an unprocessed image.
                tv_geometry=(v812_validate_projected_tv_scale(placement,rendered_bbox,room.size)
                    if TV_WALL_GEOMETRY_V812['enabled'] else
                    validate_tv_cutout_scale(bbox,rendered_bbox,room.size))
                rec['tv_scale']={
                    **tv_geometry,
                    'estimated_diagonal_inch':placement.get('scale_vs_scene',{}).get('diagonal_inch_estimate'),
                    'scale_source':placement.get('scale_vs_scene',{}).get('size_source'),
                    'scale_method':placement.get('scale_vs_scene',{}).get('method'),
                }
            if group == 'TV' and placement.get('surface')=='wall':
                chosen_plane=placement.get('tv_wall_selection',{}).get('chosen_plane')
                if TV_WALL_GEOMETRY_V812['enabled']:
                    # V8.12: eye-level reference supersedes the old 8/15/20%
                    # bottom-gap rule, which caused unnatural mounting heights.
                    rec['tv_wall_height']=placement.get('tv_wall_height',{})
                    rec['tv_wall_height']['validation']='perspective_quad_and_eye_level_verified_2d'
                else:
                    visible_ok,visible=v811_tv_height_check(
                        masks,room.size,
                        [rendered_bbox[k] for k in ('x1','y1','x2','y2')],chosen_plane)
                    if not visible_ok:
                        raise ValueError('TV sau API không đạt khoảng cách đến sàn/mép tường: '
                                         +str(visible.get('reason')))
                    rec['tv_wall_height']=visible
                rec['tv_wall_selection']=placement.get('tv_wall_selection',{})
            if group == 'Máy điều hòa':
                ac_validation_bbox=dict(bbox)
                if rec.get('paired_scale_plan',{}).get('paired_with') == 'TV':
                    ac_validation_bbox['x2'] = (ac_validation_bbox['x1'] +
                        int(rec['paired_scale_plan']['target_width_px']))
                rec['aircon_rendered_scale']=validate_aircon_api_silhouette(
                    ac_validation_bbox,rendered_bbox,room.size)
                ac_good,ac_clearance=_v84_aircon_wall_clearance(masks,room.size,
                    [rendered_bbox[k] for k in ('x1','y1','x2','y2')])
                if not ac_good:
                    raise ValueError('Máy lạnh sau API không đạt khoảng hở theo ranh giới bề mặt: '
                                     +str(ac_clearance))
                rec['aircon_wall_clearance']=ac_clearance
                rec['aircon_depth_scale']=placement.get('scale_vs_scene',{})
            if group == 'Máy điều hòa' and placement.get('surface')=='wall':
                # Keep the legacy AC behavior; TV is now calibrated independently.
                rec['tv_aircon_pair_check']=v89_validate_paired_render(
                    group,rendered_bbox,placement,tv_ac_pair,committed_wall_pair)
            # Chỉ COMMIT canvas và reserve sau khi POST + validate + compose thành công.
            canvas=next_canvas
            reserved.append({'label':'Đã ghép: '+group,
                             'bbox':[rendered_bbox[k] for k in ('x1','y1','x2','y2')],
                             'confidence':1.0})
            if group in ('TV','Máy điều hòa') and placement.get('surface')=='wall':
                committed_wall_pair[group]={
                    'rendered_bbox':dict(rendered_bbox),
                    'scale_vs_scene':placement.get('scale_vs_scene',{}),
                    'surface':'wall'}
            rec.update(status='rendered',reason='',bbox_px=bbox,
                       rendered_bbox_px=rendered_bbox,surface=placement.get('surface'),
                       source='live_background_removal_api',lighting=lighting)
            # Xuất ảnh của riêng sản phẩm này chỉ khi API thành công.
            preview=job_dir/f'preview_{rank:02d}_{re.sub(r"[^A-Za-z0-9_-]","_",product["product_id"])}.png'
            next_canvas.convert('RGB').save(preview)
            rec['preview_path']=str(preview)
        except Exception as exc:
            # Không dùng rembg, threshold mask hay bất cứ fallback hình ảnh nào.
            rec.update(status='api_or_render_failed',reason=f'{type(exc).__name__}: {exc}')
            print(f'Không ghép {group}: {rec["reason"]}')

    results.sort(key=lambda item:item['rank'])  # Stage 1 order in output
    success=[x for x in results if x['status']=='rendered']
    combined_path=None
    if success:
        combined_path=job_dir/'final_render.png'
        canvas.convert('RGB').save(combined_path)
    if LIGHTING_POLICY_V88.get('debug'):
        try:
            v88_save_lighting_placements_debug(scene_render,results,
                job_dir/'lighting_placement_debug.png')
        except Exception as exc:
            print('Không tạo được debug ánh sáng vùng đặt:',exc)
    mapping_audit=pd.DataFrame([{
        'category_id':r.get('category_id'), 'api_group_name':r.get('api_group_name'),
        'product_id':r.get('product_id'), 'product_name':r.get('product_name'),
        'mapping_status':r.get('category_mapping',{}).get('status'),
        'official_category_name':r.get('category_mapping',{}).get('official_category_name'),
        'placement_group':r.get('category_mapping',{}).get('placement_group'),
        'product_type':r.get('product_type'),
        'render_status':r.get('status'), 'reason':r.get('reason'),
    } for r in results])
    mapping_audit.to_csv(job_dir/'category_mapping_audit.csv',index=False,encoding='utf-8-sig')
    report={'category_registry_version':'V8.15 (434 rows from categories(1).json)',
            'category_mapping_audit':str(job_dir/'category_mapping_audit.csv'),
            'source':'external_recommendation_api','recommend_endpoint':stage1_result.get('api_endpoint'),
            'recommend_response_source':stage1_result.get('api_response_source'),
            'selection_seed':stage1_result.get('selection_seed'),
            'endpoint':REMOVE_BG_API_URL,
            'scene_lighting':scene_lighting,
            'lighting_policy_version':'V8.8',
            'tv_scale_policy_version':'V8.12',
            'tv_wall_geometry_policy':TV_WALL_GEOMETRY_V812,
            'tv_aircon_pair_policy':'V8.10', 'tv_aircon_pair':tv_ac_pair,
            'rendered_count':len(success),'recommended_count':len(results),
            'final_image':str(combined_path) if combined_path else None,
            'segmentation_info':segmentation_info,
            'wall_plane_info':segmentation_info.get('wall_planes',{}),
            'depth_info':depth_info,
            'items':results}
    (job_dir/'render_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
    # Lưu nguyên JSON backend và SKU đã chọn để kiểm tra nguồn gốc dữ liệu.
    raw_payload=stage1_result.get('raw_recommendation_response')
    if raw_payload is not None:
        (job_dir/'recommend_api_response.json').write_text(
            json.dumps(raw_payload,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
    selected_table=stage1_result.get('recommendation_table')
    if isinstance(selected_table,pd.DataFrame):
        selected_table.to_csv(job_dir/'recommend_selected_products.csv',index=False,encoding='utf-8-sig')
    df=pd.DataFrame(results)
    df.to_csv(job_dir/'render_results.csv',index=False,encoding='utf-8-sig')
    print(f'BƯỚC 2: Đã ghép {len(success)}/{len(results)} sản phẩm bằng API ngrok.')
    if tv_ac_pair and all(g in committed_wall_pair for g in ('TV','Máy điều hòa')):
        aw=committed_wall_pair['Máy điều hòa']['rendered_bbox']
        tw=committed_wall_pair['TV']['rendered_bbox']
        actual=(aw['x2']-aw['x1'])/max(1,tw['x2']-tw['x1'])
        print(f'[V8.10] Tỷ lệ thực trên ảnh máy lạnh/TV = {actual:.3f} '
              f'(tỷ lệ adaptive theo inch = {tv_ac_pair["adaptive_aircon_to_tv_ratio"]:.3f}; '
              'đã xét depth tương đối).')
    if len(df):
        display(df[['rank','group_name','product_name','product_type','status','reason']])
    else:
        print('Danh sách gợi ý rỗng; không gọi API và không tạo ảnh kết quả.')
    if show_final and combined_path is not None:
        print('ẢNH CUỐI CÙNG — chỉ gồm sản phẩm trả cutout hợp lệ từ API:')
        display(canvas.convert('RGB'))
        print('Đã lưu:',combined_path)
    elif show_final:
        print('Không có sản phẩm nào ghép thành công. Không xuất ảnh giả / không dùng ảnh gốc làm kết quả.')
    return report


# ===== ORCHESTRATION FUNCTION =====
def render_room_recommendations(
    room_image: Image.Image,
    recommendations_result: dict,
    remove_bg_api_url: str | None = None,
    output_dir: Path | None = None,
) -> dict:
    """Run spatial placement and render for recommended products.

    - Any product that fails background removal or placement is captured with status and reason.
    - Successfully composited products are composited onto the room image.
    - Returns structured result including rendered image (base64 PNG) and item details.
    """
    api_url = remove_bg_api_url or REMOVE_BG_API_URL
    if not api_url and REMOVE_BG_BASE_URL:
        _b = REMOVE_BG_BASE_URL.rstrip("/")
        api_url = _b if "/api/v1/products/remove-background" in _b else f"{_b}/api/v1/products/remove-background"
    room = room_image.convert("RGBA")
    W, H = room.size

    groups = recommendations_result.get("recommendations", [])
    selected_items = []
    seen = set()

    for grp in groups:
        gname = str(grp.get("group_name") or "").strip()
        gcode = str(grp.get("category_id") or grp.get("group_code") or "").strip()
        key = gcode or _fold_vi(gname)
        if not key or not gname or key in seen:
            continue
        seen.add(key)
        prods = grp.get("products") or []
        if not prods:
            continue
        prod = dict(prods[0])  # Top 1 sản phẩm của category (không random)
        for k in ("product_id", "product_code", "product_name", "image_url", "product_url", "category_id", "brand"):
            prod[k] = str(prod.get(k) or "").strip()
        prod["brand"] = prod["brand"] or str(prod.get("brand_name") or "").strip()
        prod["category_id"] = prod["category_id"] or gcode
        try:
            prod["price"] = float(prod.get("price") or 0.0)
        except (ValueError, TypeError):
            prod["price"] = 0.0
        prod["currency"] = str(prod.get("currency") or "VND")
        prod["subcategory"] = str(prod.get("subcategory") or prod.get("product_type") or "").strip()

        # Category resolution
        resolved = v815_resolve_category(gcode, gname, prod)
        canonical = resolved.get("placement_group") or canonical_placement_group(gname, prod)
        product_type = resolved.get("product_type") or resolve_remove_bg_product_type({"category_id": gcode, "placement_group": canonical}, prod)

        item = {k: v for k, v in grp.items() if k != "products"}
        item.update(
            group_name=gname,
            group_code=gcode,
            category_id=gcode,
            placement_group=canonical,
            product_type=product_type,
            products=prods,
            selected_product=prod,
            category_mapping=resolved,
            final_score=float(grp.get("final_score") or 0.0),
        )
        selected_items.append(item)

    detected_raw = recommendations_result.get("detected_objects", [])
    spatial_detections = extract_api_spatial_detections(detected_raw, room.size)
    non_obstacles = {
        "rug", "carpet", "wall art", "decor", "curtain", "pillow",
        "light", "books", "vase", "clock", "mirror", "candle",
        "picture frame", "mugs teacups", "planter", "basket/bin"
    }
    spatial_detections = [d for d in spatial_detections if d.get("label", "").lower() not in non_obstacles]

    # Sync metadata
    API_CATEGORY_PRODUCTS.clear()
    for entry in selected_items:
        API_CATEGORY_PRODUCTS.setdefault(entry["placement_group"], []).extend(entry["products"])
    for entry in selected_items:
        apply_api_sku_surface_metadata(entry["placement_group"], entry["selected_product"])
    sync_placement_profiles_with_catalog(API_CATEGORY_PRODUCTS, display_audit=False)
    attach_group_name_to_profiles()

    # Segmentation and Depth
    try:
        masks, segmentation_info = segment_room_surfaces(room.convert("RGB"))
    except Exception as exc:
        masks = {}
        segmentation_info = {"status": "unavailable", "reason": f"{type(exc).__name__}: {exc}"}

    if segmentation_info.get("status") == "ok":
        tv_meta = v811_prepare_tv_planes(room, masks)
        segmentation_info["tv_wall_planes_v811"] = tv_meta

    try:
        depth, depth_info = estimate_relative_depth(room.convert("RGB"))
    except Exception as exc:
        depth = None
        depth_info = {"status": "unavailable", "reason": f"{type(exc).__name__}: {exc}"}

    # Lighting
    try:
        scene_render, scene_lighting = v88_analyze_room_and_prepare_render(room, job_dir=None)
    except Exception as exc:
        scene_render = room.copy()
        scene_lighting = {"status": "unavailable", "room_exposure_gain": 1.0}

    # TV/AC pairing
    stage1_wrap = {"recommendations": selected_items}
    tv_ac_pair = v89_pair_selected_products(stage1_wrap)
    ordered_items = v89_order_render_items(stage1_wrap, tv_ac_pair)

    reserved = []
    committed_wall_pair = {}
    canvas = scene_render.copy()
    results = []

    for rank, entry in ordered_items:
        prods = entry.get("products") or [entry.get("selected_product")]
        prod = prods[0] if prods else entry.get("selected_product")
        group = entry["placement_group"]

        prod_dict = dict(prod or {})
        for k in ("product_id", "product_code", "product_name", "image_url", "product_url", "category_id", "brand"):
            prod_dict[k] = str(prod_dict.get(k) or "").strip()
        prod_dict["brand"] = prod_dict["brand"] or str(prod.get("brand_name") or "").strip()
        prod_dict["category_id"] = prod_dict["category_id"] or entry.get("category_id") or ""
        try:
            prod_dict["price"] = float(prod_dict.get("price") or 0.0)
        except (ValueError, TypeError):
            prod_dict["price"] = 0.0
        prod_dict["currency"] = str(prod_dict.get("currency") or "VND")
        prod_dict["subcategory"] = str(prod_dict.get("subcategory") or prod_dict.get("product_type") or "").strip()

        current_rec = {
            "rank": rank,
            "category_id": entry.get("category_id"),
            "group_name": entry.get("group_name"),
            "placement_group": group,
            "product_type": entry.get("product_type") or "auto",
            "product_id": prod_dict.get("product_id"),
            "product_name": prod_dict.get("product_name"),
            "price": prod_dict.get("price"),
            "currency": prod_dict.get("currency"),
            "image_url": prod_dict.get("image_url"),
            "status": "pending",
            "reason": "",
            "surface": None,
            "bbox_px": None,
            "rendered_bbox_px": None,
        }

        verified_mapping = entry.get("category_mapping") or {}
        if not verified_mapping.get("renderable"):
            current_rec.update(status="unmapped_category", reason=verified_mapping.get("reason", "Thiếu category mapping an toàn"))
            results.append(current_rec)
            continue

        restriction = api_product_placement_restriction(group, prod_dict)
        if restriction:
            current_rec.update(status="unsupported_product_surface", reason=restriction)
            results.append(current_rec)
            continue

        if segmentation_info.get("status") != "ok":
            current_rec.update(status="no_placement", reason="Mask2Former không cung cấp surface masks")
            results.append(current_rec)
            continue

        try:
            profile = resolve_product_profile(group, prod_dict)
            if profile is None:
                current_rec.update(status="no_placement", reason="Không có placement profile")
                results.append(current_rec)
                continue

            occupancy = spatial_detections + reserved
            placement = choose_placement_region(profile, masks, occupancy, room.size, depth_map=depth)
            if not placement.get("bbox_px"):
                current_rec.update(
                    status="no_placement",
                    reason=f"{placement.get('status', 'no_safe_region')}: {placement.get('reason', '')}",
                )
                results.append(current_rec)
                continue

            bbox = placement["bbox_px"]
            if any(boxes_intersect(bbox, d["bbox"]) for d in reserved if d.get("bbox") is not None):
                current_rec.update(status="placement_collision", reason="Bbox đè lên ảnh đã ghép")
                results.append(current_rec)
                continue

            if not prod_dict.get("image_url"):
                current_rec.update(status="missing_image_url", reason="SKU không có image_url")
                results.append(current_rec)
                continue

            # Download & Cutout (Không fallback: nếu không gọi được API tách nền thì log lỗi ra)
            try:
                source_bytes, source_name, mime = download_selected_product_image(prod_dict["image_url"])
            except Exception as dl_exc:
                logger.error(
                    "Lỗi tải ảnh sản phẩm [%s - %s] từ URL '%s': %s",
                    prod_dict.get("product_id"), prod_dict.get("product_name"),
                    prod_dict.get("image_url"), dl_exc, exc_info=True,
                )
                current_rec.update(status="download_failed", reason=f"Lỗi tải ảnh: {dl_exc}")
                results.append(current_rec)
                continue

            try:
                cutout = require_api_cutout(
                    source_bytes,
                    source_name,
                    mime,
                    current_rec["product_type"],
                    api_url=api_url,
                    product_id=prod_dict.get("product_id"),
                    product_name=prod_dict.get("product_name"),
                    group_name=group,
                )
                logger.info(
                    "Tách nền THÀNH CÔNG cho sản phẩm [%s - %s] (nhóm: %s, product_type: %s) — tiếp tục xử lý relighting và compositing",
                    prod_dict.get("product_id"),
                    prod_dict.get("product_name"),
                    group,
                    current_rec["product_type"],
                )
            except Exception as api_exc:
                logger.error(
                    "Tách nền THẤT BẠI cho sản phẩm [%s - %s] (nhóm: %s, product_type: %s): %s",
                    prod_dict.get("product_id"),
                    prod_dict.get("product_name"),
                    group,
                    current_rec["product_type"],
                    api_exc,
                )
                current_rec.update(status="cutout_error", reason=f"Không gọi được API tách nền: {api_exc}")
                results.append(current_rec)
                continue

            # Photometric Relighting (V8.8 + V8.20 minus 20%)
            lit_cutout, lighting = v88_relight_api_cutout(cutout, scene_render, placement, masks=masks, group_name=group)
            lit_cutout, final_brightness = v820_match_object_to_scene_minus_20(lit_cutout, scene_render, placement, masks=masks, group_name=group)

            # Compositing
            if group == "TV" and TV_WALL_GEOMETRY_V812.get("enabled"):
                next_canvas, rendered_bbox, projection_diag = v812_composite_tv_cutout(canvas, lit_cutout, placement, masks)
                current_rec["tv_projective_geometry"] = projection_diag
            elif group == "Máy điều hòa" and current_rec.get("paired_scale_plan", {}).get("target_width_px"):
                next_canvas, rendered_bbox = v89_composite_width_first(canvas, lit_cutout, placement, current_rec["paired_scale_plan"]["target_width_px"])
            else:
                next_canvas, rendered_bbox = composite_api_cutout(canvas, lit_cutout, placement)

            canvas = next_canvas
            reserved.append({
                "label": f"Đã ghép: {group}",
                "bbox": [rendered_bbox[k] for k in ("x1", "y1", "x2", "y2")],
                "confidence": 1.0,
            })
            current_rec.update(
                status="rendered",
                reason="",
                bbox_px=bbox,
                rendered_bbox_px=rendered_bbox,
                surface=placement.get("surface"),
            )
            entry["selected_product"] = prod_dict
            results.append(current_rec)
        except Exception as exc:
            logger.error(
                "Lỗi xử lý ghép ảnh cho sản phẩm [%s - %s] (nhóm: %s): %s",
                prod_dict.get("product_id"), prod_dict.get("product_name"), group, exc, exc_info=True,
            )
            current_rec.update(status="render_failed", reason=f"{type(exc).__name__}: {exc}")
            results.append(current_rec)

    results.sort(key=lambda x: x["rank"])
    success = [x for x in results if x["status"] == "rendered"]

    buf = BytesIO()
    canvas.convert("RGB").save(buf, format="PNG")
    png_bytes = buf.getvalue()
    rendered_image_b64 = f"data:image/png;base64,{base64.b64encode(png_bytes).decode('ascii')}"

    output_recs = []
    for grp in recommendations_result.get("recommendations", []):
        grp_copy = dict(grp)
        # Giữ nguyên toàn bộ danh sách sản phẩm nhận được từ recommend (không cắt bớt)
        grp_copy["products"] = list(grp.get("products") or [])
        output_recs.append(grp_copy)

    return {
        "room_type": recommendations_result.get("room_type"),
        "detected_objects": recommendations_result.get("detected_objects", []),
        "recommendations": output_recs,
        "items": results,
        "image": rendered_image_b64,
        "rendered_image": rendered_image_b64,
        "image_bytes": png_bytes,
        "rendered_count": len(success),
        "recommended_count": len(results),
        "diagnostics": {
            "segmentation_info": segmentation_info,
            "depth_info": depth_info,
            "scene_lighting": scene_lighting,
        },
    }
