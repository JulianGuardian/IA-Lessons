"""Week 12 — the Week 6 campus_helper_agent, copied here so this week's eval pipeline is self-contained."""

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm

CHAT_MODEL = "ollama_chat/qwen2.5:14b"

COURSE_TOPICS = {
    6: "Agent fundamentals",
    7: "Google ADK",
    8: "Multi-agent systems",
    9: "LangGraph I",
    10: "LangGraph II + advanced RAG",
    12: "Evaluation",
}
STUDY_HOURS_TABLE = {"easy": 2, "medium": 4, "hard": 7}


def get_course_topic(week: int) -> dict:
    """Look up which topic is covered in a given week of the AI Agentic Engineering course."""
    return {"week": week, "topic": COURSE_TOPICS.get(week, "unknown")}


def estimate_study_hours(topic: str, difficulty: str) -> dict:
    """Estimate independent study hours for a topic given a difficulty: 'easy', 'medium', or 'hard'."""
    hours = STUDY_HOURS_TABLE.get(difficulty.lower().strip())
    if hours is None:
        return {
            "error": f"Unknown difficulty '{difficulty}'. Use easy, medium, or hard."
        }
    return {"topic": topic, "difficulty": difficulty, "hours": hours}


root_agent = Agent(
    model=LiteLlm(model=CHAT_MODEL),
    name="campus_helper_agent",
    description="Answers questions about the AI Agentic Engineering course schedule and study planning.",
    instruction=(
        "Use get_course_topic to find out what a given week covers. Use estimate_study_hours to estimate "
        "how long a topic takes to study. Only call a tool when the question actually needs one — for "
        "ambiguous questions with no week number or topic, ask a clarifying question instead."
    ),
    tools=[get_course_topic, estimate_study_hours],
)
