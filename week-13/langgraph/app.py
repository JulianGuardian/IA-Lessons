"""Week 13 — the Week 6 campus-helper agent, traced to Langfuse and structured-logged.

Unlike ADK's blanket instrumentation, the Langfuse callback handler must be passed explicitly on
every invoke() call — wrapped here in ask() so it can't be forgotten on a new code path.
Each turn runs inside a root Langfuse observation, so the trace_id written to the logs is the real
Langfuse trace ID: a log line can be pasted into the dashboard search to open its trace.
Needs LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_BASE_URL in a .env next to this file.
Run with: python app.py
"""

import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")  # before the Langfuse client reads its keys

from langchain.agents import create_agent  # noqa: E402  (LangGraph's create_react_agent moved here in v1)
from langchain_core.tools import tool  # noqa: E402
from langchain_ollama import ChatOllama  # noqa: E402
from langfuse import get_client  # noqa: E402
from langfuse.langchain import CallbackHandler  # noqa: E402

CHAT_MODEL = "qwen2.5:14b"

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("agent")
for noisy in ("httpx", "httpcore"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

COURSE_TOPICS = {
    6: "Agent fundamentals", 7: "Google ADK", 8: "Multi-agent systems",
    9: "LangGraph I", 10: "LangGraph II + advanced RAG", 13: "Observability",
}
STUDY_HOURS_TABLE = {"easy": 2, "medium": 4, "hard": 7}

langfuse = None  # set in main(), once the keys are known to be present


def log_event(event: str, **fields) -> None:
    """Structured (JSON) log line, correlated with the Langfuse trace of the current turn."""
    trace_id = langfuse.get_current_trace_id() if langfuse else None
    logger.info(json.dumps({"event": event, "trace_id": trace_id, **fields}, ensure_ascii=False))


@tool
def get_course_topic(week: int) -> dict:
    """Look up which topic is covered in a given week of the AI Agentic Engineering course."""
    result = {"week": week, "topic": COURSE_TOPICS.get(week, "unknown")}
    log_event("tool_called", tool="get_course_topic", args={"week": week}, result=result)
    return result


@tool
def estimate_study_hours(topic: str, difficulty: str) -> dict:
    """Estimate independent study hours for a topic given a difficulty: 'easy', 'medium', or 'hard'."""
    hours = STUDY_HOURS_TABLE.get(difficulty.lower().strip())
    if hours is None:
        result = {"error": f"Unknown difficulty '{difficulty}'. Use easy, medium, or hard."}
    else:
        result = {"topic": topic, "difficulty": difficulty, "hours": hours}
    log_event("tool_called", tool="estimate_study_hours",
              args={"topic": topic, "difficulty": difficulty}, result=result)
    return result


def build_agent():
    llm = ChatOllama(model=CHAT_MODEL)
    return create_agent(
        model=llm,
        tools=[get_course_topic, estimate_study_hours],
        system_prompt="Use get_course_topic for schedule questions and estimate_study_hours for study-time questions.",
    )


def ask(agent, langfuse_handler, question: str) -> str:
    # The root observation creates the trace; the callback handler nests the LangGraph spans under it.
    with langfuse.start_as_current_observation(name="campus_helper_turn", as_type="agent", input=question) as turn:
        log_event("turn_started", question=question)
        try:
            result = agent.invoke(
                {"messages": [{"role": "user", "content": question}]},
                config={"callbacks": [langfuse_handler]},   # easy to forget on a new call site -- always route through ask()
            )
        except Exception as exc:
            log_event("turn_failed", error=str(exc))
            raise
        answer = result["messages"][-1].content
        turn.update(output=answer)
        log_event("turn_completed", answer_length=len(answer), trace_url=trace_url())
    return answer


def trace_url() -> str | None:
    """Dashboard link for the current trace. It asks the Langfuse API for the project ID, so an
    unreachable Langfuse must not break the user's turn -- the trace_id in the logs is enough."""
    try:
        return langfuse.get_trace_url(trace_id=langfuse.get_current_trace_id())
    except Exception:  # noqa: BLE001
        return None


def main():
    global langfuse
    if not os.environ.get("LANGFUSE_PUBLIC_KEY") or not os.environ.get("LANGFUSE_SECRET_KEY"):
        raise SystemExit("Set LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_BASE_URL in langgraph/.env.")

    langfuse = get_client()
    agent = build_agent()
    langfuse_handler = CallbackHandler()
    print("Traced campus helper agent (LangGraph + Langfuse). Type a question, or 'quit'.\n")
    try:
        while True:
            user_input = input("you> ").strip()
            if user_input.lower() in {"quit", "exit"}:
                break
            if user_input:
                print("agent>", ask(agent, langfuse_handler, user_input), "\n")
    finally:
        langfuse.flush()  # send buffered spans before the process exits


if __name__ == "__main__":
    main()
