"""
main.py — FastAPI service for the SHL Conversational Assessment Recommender.

Endpoints:
  GET  /health  → {"status": "ok"}
  POST /chat    → AgentResponse
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv()   
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from agent import run_agent
from catalog import catalog_index

# ─── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ─── Startup / shutdown ──────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Building catalog index …")
    catalog_index.build()
    logger.info("Catalog index ready. Service is up.")
    yield
    logger.info("Shutting down.")


# ─── App ─────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="SHL Assessment Recommender",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Schemas ─────────────────────────────────────────────────────────────────
class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1)


class ChatRequest(BaseModel):
    messages: list[Message] = Field(..., min_length=1, max_length=16)

    @field_validator("messages")
    @classmethod
    def last_message_must_be_user(cls, v: list[Message]) -> list[Message]:
        if v[-1].role != "user":
            raise ValueError("The last message must be from the user.")
        return v


class Recommendation(BaseModel):
    name:      str
    url:       str
    test_type: str


class ChatResponse(BaseModel):
    reply:               str
    # null = still clarifying; [] = nothing to recommend; 1-10 = shortlist
    recommendations:     list[Recommendation] | None = None
    end_of_conversation: bool = False


# ─── Routes ──────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    messages = [m.model_dump() for m in request.messages]
    try:
        result = run_agent(messages, catalog_index)
    except Exception as exc:
        logger.exception("Unhandled error in agent: %s", exc)
        raise HTTPException(status_code=500, detail="Agent error. Please try again.")

    recs = result.get("recommendations") or []
    return ChatResponse(
        reply=result["reply"],
        recommendations=[Recommendation(**r) for r in recs],
        end_of_conversation=result.get("end_of_conversation", False),
    )