"""FastAPI application for RAG knowledge assistant."""

import logging
import time
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator

from app.answer import answer_question
from app.contracts import Answer, USERS
from app.llm import LLMError, LLMTimeout
from app.log import log_unanswered
from app.stubs import stub_retrieve

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG Knowledge Assistant")


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    errors = exc.errors()
    msg = "Invalid request"
    if errors:
        msg = errors[0].get("msg", "Invalid request")
        if msg.startswith("Value error, "):
            msg = msg[len("Value error, "):]
    return JSONResponse(
        status_code=400,
        content={"error": msg},
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    if exc.status_code == 400:
        return JSONResponse(
            status_code=400,
            content={"error": exc.detail if isinstance(exc.detail, str) else "Bad Request"},
        )
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )


class ChatSearchRequest(BaseModel):
    question: str
    user: str
    force_low_score: bool = False

    @field_validator("question")
    @classmethod
    def validate_question(cls, v: str) -> str:
        if not isinstance(v, str):
            raise ValueError("question must be a string")
        stripped = v.strip()
        if not stripped:
            raise ValueError("question must be non-empty")
        if len(stripped) > 2000 or len(v) > 2000:
            raise ValueError("question must not exceed 2000 characters")
        return stripped

    @field_validator("user")
    @classmethod
    def validate_user(cls, v: str) -> str:
        if v not in USERS:
            raise ValueError(f"Unknown user: {v}")
        return v


class DocumentStubRequest(BaseModel):
    filename: str
    access_group: str


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/chat")
def chat(payload: ChatSearchRequest) -> dict[str, Any]:
    start_time = time.perf_counter()
    request_id = str(uuid.uuid4())

    # Identity resolution: map user to user_groups via USERS
    user_groups = USERS[payload.user]

    passages = []
    try:
        # Retrieve relevant passages without passing user identity into prompts
        passages = stub_retrieve(
            payload.question,
            user_groups=user_groups,
            force_low_score=payload.force_low_score,
        )
        answer = answer_question(payload.question, passages, request_id=request_id)
    except HTTPException:
        raise
    except LLMTimeout as exc:
        logger.warning("LLM timeout for request_id=%s: %s", request_id, exc)
        log_unanswered(
            question_len=len(payload.question),
            reason="llm_timeout",
            request_id=request_id,
        )
        answer = Answer(
            status="error",
            answer="",
            citations=[],
            conflict=None,
            error_message="The answering service timed out. Please try again.",
            request_id=request_id,
        )
    except LLMError as exc:
        logger.warning("LLM error for request_id=%s: %s", request_id, exc)
        log_unanswered(
            question_len=len(payload.question),
            reason="llm_error",
            request_id=request_id,
        )
        answer = Answer(
            status="error",
            answer="",
            citations=[],
            conflict=None,
            error_message="The answering service failed. Please try again.",
            request_id=request_id,
        )
    except Exception as exc:
        logger.exception("Unexpected error for request_id=%s: %s", request_id, exc)
        log_unanswered(
            question_len=len(payload.question),
            reason="unexpected_error",
            request_id=request_id,
        )
        answer = Answer(
            status="error",
            answer="",
            citations=[],
            conflict=None,
            error_message="The answering service failed. Please try again.",
            request_id=request_id,
        )

    latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

    # Safe logging: only log question length and request_id, never raw question or prompt
    logger.info(
        "Chat request processed: question_len=%d, request_id=%s",
        len(payload.question),
        answer.request_id,
    )

    passages_payload = [
        {
            "title": p.chunk.title,
            "page": p.chunk.page,
            "score": p.score,
            "access_group": p.chunk.access_group,
        }
        for p in passages
    ]

    response_data = answer.model_dump()
    response_data["latency_ms"] = latency_ms
    response_data["passages"] = passages_payload
    return response_data


@app.post("/api/search")
def search(payload: ChatSearchRequest) -> dict[str, Any]:
    user_groups = USERS[payload.user]
    passages = stub_retrieve(payload.question, user_groups=user_groups)

    logger.info("Search request processed: question_len=%d", len(payload.question))

    passages_payload = [
        {
            "title": p.chunk.title,
            "page": p.chunk.page,
            "score": p.score,
            "access_group": p.chunk.access_group,
        }
        for p in passages
    ]

    return {"passages": passages_payload}


@app.post("/api/documents")
def documents(payload: DocumentStubRequest) -> dict[str, str]:
    return {"status": "not_implemented"}
