"""Dependency dùng chung: lấy `AppState` từ `app.state`."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from app.services.bootstrap import AppState


def get_app_state(request: Request) -> AppState:
    state = getattr(request.app.state, "app_state", None)
    if state is None:  # pragma: no cover - chỉ xảy ra khi lifespan chưa chạy
        from app.core.exceptions import ModelNotReadyError

        raise ModelNotReadyError("Ứng dụng chưa khởi động xong.")
    return state


AppStateDep = Annotated[AppState, Depends(get_app_state)]
