"""FastAPI app factory."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.openapi.docs import get_redoc_html, get_swagger_ui_html
from fastapi.responses import HTMLResponse

from app.api.openapi import TAGS_METADATA, API_DESCRIPTION, configure_openapi
from app.api.routes import health, recommendations
from app.core.config import Settings, get_settings
from app.core.exceptions import install_exception_handlers
from app.services.bootstrap import AppState, build_state

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger("app.main")


def create_app(settings: Settings | None = None, *, state: AppState | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Nạp artifact đúng một lần. Không train; checkpoint chỉ được đọc từ file .pt.
        app.state.app_state = state or build_state(settings)
        logger.info("ready=%s", app.state.app_state.ready)
        try:
            yield
        finally:
            app.state.app_state.executor.shutdown(wait=False, cancel_futures=True)

    docs = settings.docs_enabled
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        summary="Gợi ý sản phẩm điện máy từ ảnh phòng — RT-DETR + đồng xuất hiện (Cell 1 → 10.14)",
        description=API_DESCRIPTION,
        lifespan=lifespan,
        openapi_tags=TAGS_METADATA,
        contact={"name": "recommend-by-image-v5"},
        license_info={"name": "MIT"},
        swagger_ui_parameters={
            # Giữ API key và tham số sau khi tải lại trang.
            "persistAuthorization": True,
            "tryItOutEnabled": True,
            "displayRequestDuration": True,
            "docExpansion": "list",
            "defaultModelsExpandDepth": 1,
            "defaultModelExpandDepth": 2,
            "filter": True,
            "syntaxHighlight.theme": "monokai",
        },
        # Tài liệu được khai báo thủ công bên dưới để chèn CDN tiếng Việt/nhãn riêng.
        docs_url=None,
        redoc_url=None,
        openapi_url="/openapi.json" if docs else None,
    )

    install_exception_handlers(app)
    configure_openapi(app)

    app.include_router(health.router)
    app.include_router(recommendations.router)

    if docs:

        @app.get("/docs", include_in_schema=False)
        async def swagger_ui() -> HTMLResponse:
            """Swagger UI — thử gọi API trực tiếp từ trình duyệt."""
            return get_swagger_ui_html(
                openapi_url=app.openapi_url,
                title=f"{app.title} — Swagger UI",
                swagger_favicon_url="https://fastapi.tiangolo.com/img/favicon.png",
                swagger_ui_parameters=app.swagger_ui_parameters,
            )

        @app.get("/redoc", include_in_schema=False)
        async def redoc_ui() -> HTMLResponse:
            """ReDoc — đọc hợp đồng API dạng tài liệu."""
            return get_redoc_html(
                openapi_url=app.openapi_url,
                title=f"{app.title} — ReDoc",
                redoc_favicon_url="https://fastapi.tiangolo.com/img/favicon.png",
                with_google_fonts=True,
            )

    @app.get("/", include_in_schema=False)
    async def root() -> dict:
        return {
            "name": settings.app_name,
            "version": settings.app_version,
            "docs": "/docs" if docs else None,
            "redoc": "/redoc" if docs else None,
            "openapi": app.openapi_url,
            "endpoints": {
                "health_ready": "/health/ready",
                "recommend_image": "POST /api/v1/recommendations/image",
            },
        }

    return app


def get_app() -> FastAPI:
    return create_app()


if __name__ == "__main__":
    import uvicorn

    s = get_settings()
    uvicorn.run("app.main:get_app", factory=True, host="127.0.0.1", port=8000, reload=False)
