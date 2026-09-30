"""Week 9 — confirm the checkpointer persists state per thread_id and starts fresh on a new one.

Uses the real graph (and the local LLM), so run it with Ollama up: python check_threads.py
"""

from app import build_graph


def show(graph, config: dict, label: str) -> None:
    snapshot = graph.get_state(config)
    checkpoints = len(list(graph.get_state_history(config)))
    print(f"{label}\n  state: {snapshot.values}\n  checkpoints saved: {checkpoints}\n")


def main():
    graph = build_graph()
    thread_a = {"configurable": {"thread_id": "thread-a"}}
    thread_b = {"configurable": {"thread_id": "thread-b"}}

    graph.invoke({"question": "What does week 9 cover?"}, config=thread_a)
    show(graph, thread_a, "thread-a after invoke #1")

    graph.invoke({"question": "How many hours if it's hard?"}, config=thread_a)
    show(graph, thread_a, "thread-a after invoke #2 (same thread_id)")

    show(graph, thread_b, "thread-b before any invoke (new thread_id)")

    graph.invoke({"question": "What does week 7 cover?"}, config=thread_b)
    show(graph, thread_b, "thread-b after invoke #1")
    show(graph, thread_a, "thread-a again (untouched by thread-b)")


if __name__ == "__main__":
    main()
