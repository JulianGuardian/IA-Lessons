"""Week 14 — a hand-written FastAPI wrapper around the Week 6 campus-helper LangGraph agent, with streaming.

Run from this directory with: python main.py  (serves on http://localhost:8080)
Non-streaming: POST /chat         body: {"question": "What does week 9 cover?"}
Streaming:     POST /chat/stream  same body; the answer's tokens arrive as plain text (use curl -N)
"""

import logging
import os
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain.agents import create_agent
from langchain_core.messages import AIMessageChunk
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

load_dotenv(Path(__file__).resolve().parent / ".env")
logger = logging.getLogger("campus_helper_api")

CHAT_MODEL = "qwen2.5:14b"
# Comma-separated list of the frontends allowed to call the API. Never "*" in a real deployment.
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",") if o.strip()]

COURSE_TOPICS = {
    6: "Agent fundamentals", 7: "Google ADK", 8: "Multi-agent systems",
    9: "LangGraph I", 10: "LangGraph II + advanced RAG", 14: "Production deployment",
}
STUDY_HOURS_TABLE = {"easy": 2, "medium": 4, "hard": 7}


def get_course_topic(week: int) -> dict:
    """Look up which topic is covered in a given week of the AI Agentic Engineering course."""
    return {"week": week, "topic": COURSE_TOPICS.get(week, "unknown")}


def estimate_study_hours(topic: str, difficulty: str) -> dict:
    """Estimate independent study hours for a topic given a difficulty: 'easy', 'medium', or 'hard'."""
    hours = STUDY_HOURS_TABLE.get(difficulty.lower().strip())
    if hours is None:
        return {"error": f"Unknown difficulty '{difficulty}'. Use easy, medium, or hard."}
    return {"topic": topic, "difficulty": difficulty, "hours": hours}


agent = create_agent(
    model=ChatOllama(model=CHAT_MODEL),
    tools=[get_course_topic, estimate_study_hours],
    system_prompt="Use get_course_topic and estimate_study_hours as needed.",
)

app = FastAPI(title="Campus Helper Agent API")
app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_methods=["GET", "POST"],
                   allow_headers=["Content-Type"])


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
async def chat(req: ChatRequest):
    try:
        result = await agent.ainvoke({"messages": [{"role": "user", "content": req.question}]})
    except Exception:
        # Log the traceback for us; return a clean error to the caller instead of a raw 500 traceback.
        logger.exception("Agent failed on /chat")
        raise HTTPException(status_code=502, detail="The agent could not answer right now. Try again later.")
    return {"answer": result["messages"][-1].content}


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    async def token_generator():
        try:
            async for message_chunk, metadata in agent.astream(
                {"messages": [{"role": "user", "content": req.question}]}, stream_mode="messages"
            ):
                # Only the model's own tokens: skip tool results, which also arrive as messages.
                if isinstance(message_chunk, AIMessageChunk) and message_chunk.content:
                    yield message_chunk.content
        except Exception:
            # The 200 status was already sent with the first token, so the error goes in the stream itself.
            logger.exception("Agent failed on /chat/stream")
            yield "\n[error] The agent could not finish the answer. Try again later.\n"
    return StreamingResponse(token_generator(), media_type="text/plain")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)  # no --reload: this is not a dev server
