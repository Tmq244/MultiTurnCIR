from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from .config import get_config
from .model_service import ModelService
from .retrieval_service import RetrievalService
from .reference_text_service import ReferenceTextService
from .reference_tag_service import ReferenceTagService
from .schemas import (
    GalleryItem,
    GalleryResponse,
    HealthResponse,
    ReferenceResolveResponse,
    RetrieveRequest,
    RetrieveResponse,
    RetrieveResult,
    SessionCreateRequest,
    SessionCreateResponse,
    SessionResetResponse,
)
from .session_service import SessionService

cfg = get_config()
model_service = ModelService(cfg)
retrieval_service = RetrievalService(model_service, cfg.cache_dir, cfg.index_limit)
session_service = SessionService()
reference_text_service = ReferenceTextService(cfg.data_dir)
reference_tag_service = ReferenceTagService(cfg.attr_dir)

app = FastAPI(title="Multiturn Fashion Retrieval Demo")
app.mount("/assets", StaticFiles(directory=cfg.app_root / "app/static"), name="assets")
templates = Jinja2Templates(directory=str(cfg.app_root / "app/templates"))


@app.on_event("startup")
def _startup() -> None:
    retrieval_service.ensure_index()


@app.get("/", response_class=HTMLResponse)
def index_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", model_loaded=model_service.loaded, index_size=retrieval_service.size)


@app.get("/api/gallery", response_model=GalleryResponse)
def gallery() -> GalleryResponse:
    ids = retrieval_service.gallery(limit=20, offset=0, random_sample=True)
    items = [GalleryItem(image_id=image_id, image_url=f"/images/{image_id}.jpg") for image_id in ids]
    return GalleryResponse(items=items)


@app.get("/api/reference/{image_id}", response_model=ReferenceResolveResponse)
def resolve_reference(image_id: str) -> ReferenceResolveResponse:
    exists = model_service.image_exists(image_id)
    tags = reference_tag_service.get_tags(image_id=image_id, limit=12)
    suggestions = reference_text_service.get_suggestions(image_id=image_id, count=2)
    return ReferenceResolveResponse(
        exists=exists,
        image_id=image_id,
        image_url=f"/images/{image_id}.jpg" if exists else None,
        tags=tags,
        suggested_texts=suggestions,
    )


@app.post("/api/session/new", response_model=SessionCreateResponse)
def create_session(payload: SessionCreateRequest) -> SessionCreateResponse:
    ids = retrieval_service.gallery(limit=1, offset=0)
    if not ids:
        raise HTTPException(status_code=500, detail="No gallery images available.")

    reference_id = payload.reference_id or ids[0]
    if not model_service.image_exists(reference_id):
        raise HTTPException(status_code=400, detail=f"Invalid reference_id: {reference_id}")

    session_id, state = session_service.create(reference_id)
    return SessionCreateResponse(session_id=session_id, reference_id=state.reference_id, turns=state.turns)


@app.post("/api/session/{session_id}/retrieve", response_model=RetrieveResponse)
def retrieve(session_id: str, payload: RetrieveRequest) -> RetrieveResponse:
    try:
        state = session_service.get(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found") from None

    if payload.reference_id:
        if not model_service.image_exists(payload.reference_id):
            raise HTTPException(status_code=400, detail=f"Invalid reference_id: {payload.reference_id}")
        state = session_service.update_reference(session_id, payload.reference_id)

    state = session_service.append_turn(session_id, payload.modified_text, model_service.max_turn_len)

    query_vec = model_service.embed_query(state.reference_id, state.turns)
    ranked = retrieval_service.search(query_vec, top_k=payload.top_k)

    results = [
        RetrieveResult(image_id=image_id, score=score, image_url=f"/images/{image_id}.jpg")
        for image_id, score in ranked
    ]

    return RetrieveResponse(
        session_id=session_id,
        reference_id=state.reference_id,
        turns=state.turns,
        results=results,
    )


@app.post("/api/session/{session_id}/reset", response_model=SessionResetResponse)
def reset_session(session_id: str) -> SessionResetResponse:
    try:
        state = session_service.reset(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found") from None
    return SessionResetResponse(session_id=session_id, reference_id=state.reference_id, turns=state.turns)


@app.get("/images/{filename}")
def serve_image(filename: str) -> FileResponse:
    if "/" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    image_path = cfg.images_dir / filename
    if not image_path.exists() or not image_path.is_file():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(image_path)
