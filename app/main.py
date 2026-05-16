"""
main.py
FastAPI service exposing GET /health and POST /chat.
Stateless: every /chat call carries the full conversation history.
"""

import time
from typing import Any

from dotenv import load_dotenv
load_dotenv()  # Load .env file automatically

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

from app.agent import run_agent

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="SHL Assessment Recommender",
    description="Conversational agent for SHL Individual Test Solutions",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_START_TIME = time.time()


# ── Request / Response schemas ────────────────────────────────────────────────

class Message(BaseModel):
    role: str       # "user" or "assistant"
    content: str

    @field_validator("role")
    @classmethod
    def role_must_be_valid(cls, v: str) -> str:
        if v not in ("user", "assistant"):
            raise ValueError("role must be 'user' or 'assistant'")
        return v


class ChatRequest(BaseModel):
    messages: list[Message]

    @field_validator("messages")
    @classmethod
    def messages_not_empty(cls, v: list) -> list:
        if not v:
            raise ValueError("messages list cannot be empty")
        return v


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str


class ChatResponse(BaseModel):
    reply: str
    recommendations: list[Recommendation]
    end_of_conversation: bool


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health() -> dict[str, str]:
    """Readiness check. Returns 200 OK immediately."""
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> Any:
    """
    Main conversational endpoint.
    Accepts full conversation history, returns next agent reply.
    """
    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    # Validate: last message must be from user
    if messages[-1]["role"] != "user":
        raise HTTPException(
            status_code=400,
            detail="Last message in history must be from 'user'."
        )

    # Enforce 8-turn cap (evaluator rule)
    if len(messages) > 8:
        messages = messages[-8:]

    result = run_agent(messages)

    # Ensure schema compliance — never let bad data through
    recs = [
        Recommendation(
            name=r.get("name", ""),
            url=r.get("url", ""),
            test_type=r.get("test_type", ""),
        )
        for r in result.get("recommendations", [])
        if r.get("name") and r.get("url")
    ]

    return ChatResponse(
        reply=result.get("reply", ""),
        recommendations=recs,
        end_of_conversation=bool(result.get("end_of_conversation", False)),
    )