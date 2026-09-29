"""Week 8 — orchestrator + specialists in LangChain, mirroring ADK's AgentTool pattern.

Each specialist is its own create_react_agent; we wrap each one as a plain callable tool for the
coordinator, so calling a specialist is isolated (it runs, returns a string, the coordinator stays
in control) — the LangChain analog of ADK's AgentTool. A true sub_agents-style handoff needs an
explicit graph with conditional routing, which is Week 9's job (LangGraph I).
Run with: python app.py
"""

from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent

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


def build_agent():
    llm = ChatOllama(model="qwen2.5:14b")

    research_agent = create_react_agent(
        model=llm, tools=[get_course_topic],
        prompt="Answer schedule/topic questions using get_course_topic. Be brief.",
    )
    planner_agent = create_react_agent(
        model=llm, tools=[estimate_study_hours],
        prompt="Answer study-time questions using estimate_study_hours. Be brief.",
    )
    prereq_agent = create_react_agent(
        model=llm, tools=[get_week_prerequisites],
        prompt=(
            "Answer prerequisite questions using get_week_prerequisites. List the prerequisite weeks "
            "and their topics. Never guess prerequisites yourself. Be brief."
        ),
    )

    def ask_research_agent(question: str) -> str:
        """Ask the research specialist about the course schedule (e.g. what a given week covers)."""
        result = research_agent.invoke({"messages": [{"role": "user", "content": question}]})
        return result["messages"][-1].content

    def ask_planner_agent(question: str) -> str:
        """Ask the planner specialist for a study-time estimate for a topic and difficulty."""
        result = planner_agent.invoke({"messages": [{"role": "user", "content": question}]})
        return result["messages"][-1].content

    def ask_prereq_agent(question: str) -> str:
        """Ask the prerequisite specialist which earlier weeks must be reviewed before a given week."""
        result = prereq_agent.invoke({"messages": [{"role": "user", "content": question}]})
        return result["messages"][-1].content

    return create_react_agent(
        model=llm,
        tools=[ask_research_agent, ask_planner_agent, ask_prereq_agent],
        prompt=(
            "Use ask_research_agent for what-does-week-N-cover questions, ask_planner_agent for "
            "study-time estimates, and ask_prereq_agent for which earlier weeks to review before a "
            "given week. When a question needs several, call each one and combine the results "
            "yourself into one final answer."
        ),
    )


def main():
    agent = build_agent()
    print("Orchestrator (LangChain, AgentTool-style). Type a question, or 'quit' to exit.\n")
    while True:
        user_input = input("you> ").strip()
        if user_input.lower() in {"quit", "exit"}:
            break
        result = agent.invoke({"messages": [{"role": "user", "content": user_input}]})
        # Trace: which specialists the coordinator called, with what question, and what came back.
        for message in result["messages"][1:-1]:
            for call in getattr(message, "tool_calls", None) or []:
                print(f"[tool call] {call['name']}({call['args']})")
            if message.type == "tool":
                print(f"[tool result] {message.name} -> {message.content}")
        print("agent>", result["messages"][-1].content, "\n")


if __name__ == "__main__":
    main()
