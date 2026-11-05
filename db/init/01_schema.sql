-- MWGShop schema (PostgreSQL)
-- Source: api_crawl_infor_mwgshop.json
-- Two tables: categories, products

BEGIN;

-- Drop old tables/views if they exist to allow clean migration
DROP SCHEMA IF EXISTS mwg CASCADE;
CREATE SCHEMA mwg;

DROP TABLE IF EXISTS public.products CASCADE;
DROP TABLE IF EXISTS public.categories CASCADE;

-- Categories (243 categories from crawled data)
CREATE TABLE IF NOT EXISTS public.categories (
    id VARCHAR(20) PRIMARY KEY,
    name VARCHAR(255) NOT NULL
);

-- Products (25,034 items, keyed by unique product_code)
CREATE TABLE IF NOT EXISTS public.products (
    product_code VARCHAR(50) PRIMARY KEY,
    product_id VARCHAR(50) NOT NULL,
    category_id VARCHAR(20) NOT NULL,

    name TEXT NOT NULL,
    product_type VARCHAR(20),
    brand VARCHAR(255),
    brand_code VARCHAR(50),

    image_url TEXT,
    original_price NUMERIC(15, 2),
    sale_price NUMERIC(15, 2),
    rating_vote NUMERIC(3, 1),
    quantity_sold BIGINT,

    color TEXT,
    accessories TEXT,
    warranty_policy TEXT,
    promotion TEXT,
    outstanding TEXT,

    spec_product JSONB,
    online_sale_only BOOLEAN,

    product_url TEXT,
    model_code VARCHAR(50),
    sale_program_id INTEGER,
    program_prices JSONB,
    total_gift_value NUMERIC(15, 2),

    seller_id INTEGER,
    seller_name TEXT,
    total_rating INTEGER,
    online_status_id INTEGER,
    time_crawler TIMESTAMP,

    CONSTRAINT fk_products_category
        FOREIGN KEY (category_id) REFERENCES public.categories(id) ON UPDATE CASCADE ON DELETE RESTRICT,
    CONSTRAINT chk_original_price CHECK (original_price >= 0),
    CONSTRAINT chk_sale_price CHECK (sale_price >= 0)
);

CREATE INDEX IF NOT EXISTS idx_products_product_id ON public.products(product_id);
CREATE INDEX IF NOT EXISTS idx_products_category_id ON public.products(category_id);
CREATE INDEX IF NOT EXISTS idx_products_brand_code ON public.products(brand_code);
CREATE INDEX IF NOT EXISTS idx_products_sale_price ON public.products(sale_price);
CREATE INDEX IF NOT EXISTS idx_products_spec_product ON public.products USING GIN(spec_product);

-- Ensure schema mwg compatibility: views pointing to public tables
CREATE OR REPLACE VIEW mwg.categories AS SELECT * FROM public.categories;
CREATE OR REPLACE VIEW mwg.products AS SELECT * FROM public.products;

COMMIT;
