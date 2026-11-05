#!/usr/bin/env bash
# Kiểm tra toàn vẹn catalog sau khi nạp api_crawl_infor_mwgshop.json.
#
#   ./scripts/verify.sh
#
# Thoát mã 1 nếu có bất kỳ kiểm tra nào sai.
set -euo pipefail
cd "$(dirname "$0")/.."

COMPOSE="docker compose --env-file .env"
PSQL="$COMPOSE exec -T postgres psql -U mwg -d mwgshop -v ON_ERROR_STOP=1 -tA"

run() {
  # In giá trị scalar; thoát nếu psql lỗi.
  $PSQL -c "$1"
}

expect() {
  # expect <mô tả> <sql trả về số> <giá trị mong đợi>
  local label=$1 sql=$2 want=$3 got
  got=$(run "$sql" | tr -d '[:space:]')
  if [[ "$got" == "$want" ]]; then
    printf '  ✅ %-46s %s\n' "$label" "$got"
  else
    printf '  ❌ %-46s %s (mong đợi %s)\n' "$label" "$got" "$want"
    failures=$((failures + 1))
  fi
}

failures=0

echo "📦 catalog — kiểm tra toàn vẹn (bảng public.products & public.categories)"
expect "số sản phẩm (public.products)" \
  "SELECT count(*) FROM public.products" 17347
expect "số category (public.categories)" \
  "SELECT count(*) FROM public.categories" 191
expect "số sản phẩm qua view mwg.products" \
  "SELECT count(*) FROM mwg.products" 17347
expect "số category qua view mwg.categories" \
  "SELECT count(*) FROM mwg.categories" 191

echo
echo "🔗 khoá ngoại — tất cả phải 0"
expect "sản phẩm mồ côi khỏi category" \
  "SELECT count(*) FROM public.products p
   LEFT JOIN public.categories c ON c.id = p.category_id
   WHERE c.id IS NULL" 0

echo
echo "🧮 ràng buộc dữ liệu"
expect "giá gốc âm" "SELECT count(*) FROM public.products WHERE original_price < 0" 0
expect "giá khuyến mãi âm" "SELECT count(*) FROM public.products WHERE sale_price < 0" 0
expect "tên sản phẩm rỗng" \
  "SELECT count(*) FROM public.products WHERE length(btrim(name)) = 0" 0
expect "category_name rỗng" \
  "SELECT count(*) FROM public.categories WHERE length(btrim(name)) = 0" 0
expect "product_code duy nhất" \
  "SELECT count(DISTINCT product_code) FROM public.products" 17347
expect "category id duy nhất" \
  "SELECT count(DISTINCT id) FROM public.categories" 191

echo
echo "🧩 toàn vẹn khoá ngoại theo nhóm"
expect "nhóm có sản phẩm đều có trong categories" \
  "SELECT count(*) FROM (
     SELECT DISTINCT category_id FROM public.products
   ) t WHERE t.category_id NOT IN (SELECT id FROM public.categories)" 0

echo
echo "🎯 artifact (bắt buộc, không dự phòng)"
for f in api/artifacts/matrix/cooccurrence_matrix.json \
         api/artifacts/matrix/category_room_scores.json \
         api/artifacts/rtdetr_room_detection/weights/best.pt; do
  if [[ -f "$f" ]]; then
    printf '  ✅ %-52s %s\n' "$f" "$(du -h "$f" | cut -f1)"
  else
    printf '  ❌ %-52s thiếu\n' "$f"
    failures=$((failures + 1))
  fi
done

echo
if (( failures > 0 )); then
  echo "❌ $failures kiểm tra thất bại"
  exit 1
fi
echo "✅ catalog + artifact toàn vẹn"
