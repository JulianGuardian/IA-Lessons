"""Week 8 — orchestrator + specialists using ADK's sub_agents (LLM-driven handoff).

Whichever specialist the coordinator routes to takes over the conversation and replies to the
user directly. Compare with ../orchestrator_agenttool_agent/agent.py, which keeps the parent in
control. Run with `adk web` from the `adk/` directory (both agents show up in the dropdown).
"""

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm

COURSE_TOPICS = {
    1: "LLM fundamentals", 2: "Context engineering I", 3: "Context engineering II", 4: "RAG",
    6: "Agent fundamentals", 7: "Google ADK", 8: "Multi-agent systems",
    9: "LangGraph I", 10: "LangGraph II + advanced RAG",
}
STUDY_HOURS_TABLE = {"easy": 2, "medium": 4, "hard": 7}
PREREQUISITES = {1: [], 2: [1], 3: [2], 4: [1, 3], 6: [1], 7: [6], 8: [6, 7], 9: [8], 10: [4, 9]}


def get_course_topic(week: int) -> dict:
    """Look up which topic is covered in a given week of the AI Agentic Engineering course."""
    return {"week": week, "topic": COURSE_TOPICS.get(week, "unknown")}


def estimate_study_hours(topic: str, difficulty: str) -> dict:
    """Estimate independent study hours for a topic given a difficulty: 'easy', 'medium', or 'hard'."""
    hours = STUDY_HOURS_TABLE.get(difficulty.lower().strip())
    if hours is None:
        return {"error": f"Unknown difficulty '{difficulty}'. Use easy, medium, or hard."}
    return {"topic": topic, "difficulty": difficulty, "hours": hours}


def get_week_prerequisites(week: int) -> dict:
    """Look up which earlier weeks of the course should be studied before a given week."""
    prereq_weeks = PREREQUISITES.get(week)
    if prereq_weeks is None:
        return {"week": week, "error": f"No prerequisite data for week {week}."}
    return {
        "week": week,
        "prerequisite_weeks": prereq_weeks,
        "prerequisite_topics": [COURSE_TOPICS.get(w, f"week {w}") for w in prereq_weeks],
    }


research_agent = Agent(
    model=LiteLlm(model="ollama_chat/qwen2.5:14b"),
    name="research_agent",
    description="Looks up which topic is covered in a given week of the AI Agentic Engineering course.",
    instruction="Answer schedule/topic questions using get_course_topic. Be brief.",
    tools=[get_course_topic],
)

planner_agent = Agent(
    model=LiteLlm(model="ollama_chat/qwen2.5:14b"),
    name="planner_agent",
    description="Estimates how many hours a student should budget to study a topic at a given difficulty.",
    instruction="Answer study-time questions using estimate_study_hours. Be brief.",
    tools=[estimate_study_hours],
)

prereq_agent = Agent(
    model=LiteLlm(model="ollama_chat/qwen2.5:14b"),
    name="prereq_agent",
    description=(
        "Says which earlier weeks of the AI Agentic Engineering course a student must review or "
        "master before starting a given week (prerequisites, what to study first, what a week depends on)."
    ),
    instruction=(
        "Answer prerequisite questions using get_week_prerequisites. List the prerequisite weeks "
        "and their topics. Never guess prerequisites yourself. Be brief."
    ),
    tools=[get_week_prerequisites],
)

root_agent = Agent(
    model=LiteLlm(model="ollama_chat/qwen2.5:14b"),
    name="orchestrator_subagents_agent",
    description="Coordinator that delegates schedule, study-time, and prerequisite questions to specialists.",
    instruction=(
        "You are a coordinator. Delegate questions about what a week covers to research_agent. "
        "Delegate questions about how long studying a topic will take to planner_agent. "
        "Delegate questions about which earlier weeks must be reviewed before a given week to "
        "prereq_agent. Once you delegate, let that specialist answer directly."
    ),
    sub_agents=[research_agent, planner_agent, prereq_agent],
)
