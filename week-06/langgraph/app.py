"""Week 6 — the same campus-helper agent, built with LangGraph's prebuilt ReAct agent.

Same two tools as the ADK version in ../adk/campus_helper_agent/agent.py, so you can compare the
Perception-Reasoning-Action loop across frameworks. Run with: python app.py
"""

from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent

COURSE_TOPICS = {
    1: "LLM fundamentals",
    2: "Context engineering I",
    3: "Context engineering II",
    4: "Retrieval-Augmented Generation (RAG)",
    6: "Agent fundamentals",
    7: "Google ADK",
    8: "Multi-agent systems",
    9: "LangGraph I",
    10: "LangGraph II + advanced RAG",
    12: "Evaluation",
    13: "Observability",
    14: "Production deployment",
    15: "Security and ethics",
}

STUDY_HOURS_TABLE = {"easy": 2, "medium": 4, "hard": 7}

PREREQUISITES = {
    1: [],
    2: [1],
    3: [2],
    4: [1, 3],
    6: [1],
    7: [6],
    8: [6, 7],
    9: [8],
    10: [4, 9],
    12: [6],
    13: [12],
    14: [13],
    15: [14],
}


def get_course_topic(week: int) -> dict:
    """Look up which topic is covered in a given week of the AI Agentic Engineering course."""
    topic = COURSE_TOPICS.get(week)
    if topic is None:
        return {"week": week, "topic": None, "note": "No lecture that week (exam/project delivery)."}
    return {"week": week, "topic": topic}


def estimate_study_hours(topic: str, difficulty: str) -> dict:
    """Estimate independent study hours for a topic given a difficulty: 'easy', 'medium', or 'hard'."""
    difficulty = difficulty.lower().strip()
    hours = STUDY_HOURS_TABLE.get(difficulty)
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
        "prerequisite_topics": [
            COURSE_TOPICS.get(w, f"week {w}") for w in prereq_weeks
        ],
    }


def build_agent():
    llm = ChatOllama(model="qwen2.5:14b")
    return create_react_agent(
        model=llm,
        tools=[get_course_topic, estimate_study_hours, get_week_prerequisites],
        prompt=(
            "You help students of the AI Agentic Engineering course plan their study time. "
            "Use get_course_topic to find out what a given week covers. Use get_week_prerequisites "
            "to find out which earlier weeks should be studied first. Use estimate_study_hours to "
            "estimate how long a topic takes to study, guessing a reasonable difficulty "
            "(easy/medium/hard) if the user does not specify one, and stating your assumption."
        ),
    )


def main():
    agent = build_agent()
    print("Campus helper agent (LangGraph). Type a question, or 'quit' to exit.\n")
    while True:
        user_input = input("you> ").strip()
        if user_input.lower() in {"quit", "exit"}:
            break
        result = agent.invoke({"messages": [{"role": "user", "content": user_input}]})
        for msg in result["messages"]:
            if getattr(msg, "tool_calls", None):
                for call in msg.tool_calls:
                    print(f"  [Action] {call['name']}({call['args']})")
            if msg.__class__.__name__ == "ToolMessage":
                print(f"  [Observation] {msg.content}")
        print("agent>", result["messages"][-1].content, "\n")


if __name__ == "__main__":
    main()
