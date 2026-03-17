from __future__ import annotations

from pydantic import BaseModel, Field


class SessionCreateRequest(BaseModel):
    reference_id: str | None = None


class SessionCreateResponse(BaseModel):
    session_id: str
    reference_id: str
    turns: list[str]


class RetrieveRequest(BaseModel):
    modified_text: str = Field(..., min_length=1)
    top_k: int = Field(default=10, ge=1, le=200)
    reference_id: str | None = None


class RetrieveResult(BaseModel):
    image_id: str
    score: float
    image_url: str


class RetrieveResponse(BaseModel):
    session_id: str
    reference_id: str
    turns: list[str]
    results: list[RetrieveResult]


class SessionResetResponse(BaseModel):
    session_id: str
    reference_id: str
    turns: list[str]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    index_size: int


class GalleryItem(BaseModel):
    image_id: str
    image_url: str


class GalleryResponse(BaseModel):
    items: list[GalleryItem]


class ReferenceResolveResponse(BaseModel):
    exists: bool
    image_id: str
    image_url: str | None = None
    suggested_texts: list[str] = Field(default_factory=list)
