"""Ngoại lệ nghiệp vụ + handler trả lỗi JSON thống nhất."""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger("app.errors")


class AppError(Exception):
    """Lỗi có mã HTTP đi kèm."""

    status_code = 500
    code = "internal_error"

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class ArtifactError(AppError):
    """Thiếu/sai artifact (.pt, ma trận đồng xuất hiện, catalog)."""

    status_code = 503
    code = "artifact_error"


class DataSourceError(AppError):
    """Không đọc được dữ liệu COCO/DB/CSV."""

    status_code = 503
    code = "data_source_error"


class NotFoundError(AppError):
    status_code = 404
    code = "not_found"


class ImageError(AppError):
    status_code = 400
    code = "invalid_image"


class BusyError(AppError):
    status_code = 429
    code = "too_many_requests"


class ModelNotReadyError(AppError):
    status_code = 503
    code = "model_not_ready"


def install_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def _app_error(_: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": exc.code, "message": exc.message}},
        )
