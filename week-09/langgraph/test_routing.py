"""Week 9 — unit tests for route_after_classify. Plain asserts, no LLM call.

Run with: python test_routing.py
"""

from app import route_after_classify


def state(question: str, route: str) -> dict:
    return {"question": question, "route": route, "result": ""}


def test_research_branch():
    assert route_after_classify(state("What does week 9 cover?", "research_node")) == "research_node"


def test_planner_branch():
    assert route_after_classify(state("How many hours if it's hard?", "planner_node")) == "planner_node"


def test_both_branch():
    question = "What does week 8 cover and how many hours if it's hard?"
    assert route_after_classify(state(question, "both_node")) == "both_node"


def test_clarify_when_classifier_unsure():
    assert route_after_classify(state("Tell me something", "clarify_node")) == "clarify_node"


def test_clarify_when_route_unknown():
    assert route_after_classify(state("What does week 9 cover?", "weather_node")) == "clarify_node"


def test_clarify_when_topic_question_has_no_week():
    assert route_after_classify(state("What does the course cover?", "research_node")) == "clarify_node"
    assert route_after_classify(state("What is covered and how long?", "both_node")) == "clarify_node"


def test_planner_does_not_need_a_week():
    assert route_after_classify(state("How many hours for an easy topic?", "planner_node")) == "planner_node"


if __name__ == "__main__":
    tests = [fn for name, fn in dict(globals()).items() if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"\n{len(tests)} tests passed")
