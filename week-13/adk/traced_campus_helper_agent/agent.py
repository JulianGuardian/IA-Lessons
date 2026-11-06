"""Week 13 — the campus helper agent, instrumented for Langfuse tracing.

get_client() starts the Langfuse client, which registers its OpenTelemetry exporter; then
GoogleADKInstrumentor().instrument() is a one-time, blanket call — every agent run from this point
on is automatically traced (agent, LLM call and tool spans). Needs LANGFUSE_PUBLIC_KEY /
LANGFUSE_SECRET_KEY / LANGFUSE_BASE_URL in .env next to this file (adk web loads it automatically).
Run with `adk web` from the `adk/` directory.
"""

from langfuse import get_client
from openinference.instrumentation.google_adk import GoogleADKInstrumentor

langfuse = get_client()  # exports the spans below to Langfuse; without it nothing is sent
GoogleADKInstrumentor().instrument()  # must run before the agent is used

from google.adk.agents import Agent  # noqa: E402  (import after instrumentation on purpose)
from google.adk.models.lite_llm import LiteLlm  # noqa: E402

CHAT_MODEL = "ollama_chat/qwen2.5:14b"

COURSE_TOPICS = {
    6: "Agent fundamentals", 7: "Google ADK", 8: "Multi-agent systems",
    9: "LangGraph I", 10: "LangGraph II + advanced RAG", 13: "Observability",
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


root_agent = Agent(
    model=LiteLlm(model=CHAT_MODEL),
    name="traced_campus_helper_agent",
    description="Answers questions about the course schedule and study planning, fully traced to Langfuse.",
    instruction=(
        "Use get_course_topic for schedule questions and estimate_study_hours for study-time questions. "
        "Only call a tool when actually needed."
    ),
    tools=[get_course_topic, estimate_study_hours],
)
