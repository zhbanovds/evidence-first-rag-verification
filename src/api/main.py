from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from src.api.schemas import (
    AskRequest,
    ErrorResponse,
    HealthResponse,
    IndexRequest,
    VerifyRequest,
)
from src.api.service import api_service


app = FastAPI(
    title="RAG VKR Demonstration API",
    description="HTTP wrapper over Final Evidence-First Verified RAG.",
    version="0.1.0",
)


@app.get("/health", response_model=HealthResponse)
def health() -> dict:
    return api_service.health()


@app.post("/ask")
def ask(request: AskRequest) -> dict:
    return api_service.ask(
        question=request.question,
        mode=request.mode,
        top_k=request.top_k,
    )


@app.post("/verify")
def verify(request: VerifyRequest) -> dict:
    return api_service.verify(
        question=request.question,
        answer=request.answer,
        evidence=request.evidence,
    )


@app.post(
    "/index",
    response_model=ErrorResponse,
    status_code=501,
)
def index(request: IndexRequest) -> JSONResponse:
    return JSONResponse(
        status_code=501,
        content=api_service.index_message(request.mode),
    )
