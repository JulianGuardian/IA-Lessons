"""Week 9 — the routing graph from the content page, LLM-backed, with a persistent checkpointer.

classify (LLM, structured output) -> research_node | planner_node | both_node (deterministic) -> respond (LLM) -> END
                                  +-> clarify_node (deterministic, when unsure) -> END

Run with: python app.py
Type /state to print the currently persisted state for this thread.
"""

import re
from typing import Literal, TypedDict

from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

COURSE_TOPICS = {
    6: "Agent fundamentals",
    7: "Google ADK",
    8: "Multi-agent systems",
    9: "LangGraph I",
    10: "LangGraph II + advanced RAG",
}
STUDY_HOURS_TABLE = {"easy": 2, "medium": 4, "hard": 7}


class PlannerState(TypedDict):
    question: str
    route: str
    result: str


class RouteDecision(BaseModel):
    route: Literal["research_node", "planner_node", "both_node", "clarify_node"]


WORKER_ROUTES = {"research_node", "planner_node", "both_node"}
CLARIFY_MESSAGE = (
    "I'm not sure what you're asking. Ask what a given week covers (e.g. 'What does week 9 "
    "cover?') or how many hours to study (e.g. 'How many hours if it's hard?')."
)


def _extract_week(question: str) -> int | None:
    m = re.search(r"week\s*(\d+)", question, re.IGNORECASE)
    return int(m.group(1)) if m else None


def _extract_difficulty(question: str) -> str:
    for level in STUDY_HOURS_TABLE:
        if level in question.lower():
            return level
    return "medium"  # default assumption, stated explicitly in the final answer


def route_after_classify(state: PlannerState) -> str:
    """Pick the next node. Deterministic (no LLM), so it can be unit-tested with plain asserts."""
    route = state["route"]
    if route not in WORKER_ROUTES:
        return "clarify_node"  # classify was unsure, or returned something unexpected
    if route in {"research_node", "both_node"} and _extract_week(state["question"]) is None:
        return "clarify_node"  # a topic lookup needs a week number
    return route


def build_graph():
    llm = ChatOllama(model="qwen2.5:14b")
    router_llm = llm.with_structured_output(RouteDecision)

    def classify_node(state: PlannerState) -> dict:
        decision = router_llm.invoke(
            "Classify the user's question about a course. Return route='research_node' if it only "
            "asks what a given week covers, 'planner_node' if it only asks about study hours, "
            "'both_node' if it asks about both, or 'clarify_node' if it is unclear or about "
            "something else.\n\nQuestion: " + state["question"]
        )
        return {"route": decision.route}

    def clarify_node(state: PlannerState) -> dict:
        return {"result": CLARIFY_MESSAGE}

    def research_node(state: PlannerState) -> dict:
        week = _extract_week(state["question"])
        topic = COURSE_TOPICS.get(week, "unknown") if week else "unknown"
        return {"result": f"Week {week}: {topic}"}

    def planner_node(state: PlannerState) -> dict:
        difficulty = _extract_difficulty(state["question"])
        hours = STUDY_HOURS_TABLE[difficulty]
        return {"result": f"Estimated {hours}h (assumed difficulty: {difficulty})"}

    def both_node(state: PlannerState) -> dict:
        research = research_node(state)["result"]
        plan = planner_node(state)["result"]
        return {"result": f"{research}; {plan}"}

    def respond_node(state: PlannerState) -> dict:
        reply = llm.invoke(
            f"Phrase a short, friendly final answer to '{state['question']}' using this raw data: "
            f"{state['result']}"
        )
        return {"result": reply.content}

    builder = StateGraph(PlannerState)
    for name, fn in [
        ("classify", classify_node),
        ("research_node", research_node),
        ("planner_node", planner_node),
        ("both_node", both_node),
        ("clarify_node", clarify_node),
        ("respond", respond_node),
    ]:
        builder.add_node(name, fn)

    builder.add_edge(START, "classify")
    builder.add_conditional_edges("classify", route_after_classify)
    for worker in ["research_node", "planner_node", "both_node"]:
        builder.add_edge(worker, "respond")
    builder.add_edge("respond", END)
    builder.add_edge("clarify_node", END)

    return builder.compile(checkpointer=InMemorySaver())


def main():
    graph = build_graph()
    config = {"configurable": {"thread_id": "student-session-1"}}
    print(
        "Routing graph (LangGraph). Type a question, '/state' to inspect persisted state, or 'quit'.\n"
    )
    while True:
        user_input = input("you> ").strip()
        if user_input.lower() in {"quit", "exit"}:
            break
        if user_input == "/state":
            print(graph.get_state(config).values, "\n")
            continue
        result = graph.invoke({"question": user_input}, config=config)
        print("agent>", result["result"], "\n")


if __name__ == "__main__":
    main()
