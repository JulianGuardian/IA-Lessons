"""Week 12 — a small, self-contained RAG pipeline scored by an LLM-as-judge for faithfulness and relevance.

Reuses the sample course corpus from the Week 4/10 RAG apps. Run with: python llm_judge.py
"""

from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

CHAT_MODEL = "qwen2.5:14b"

COURSE_DOCS = """
AI Agentic Engineering is a 16-week elective course for Systems Engineering students at Universidad de
Santander, organized into three graded cuts called cortes. Corte 1 (weeks 1-5) covers LLM fundamentals,
context engineering, and RAG. Corte 2 (weeks 6-11) covers agents, multi-agent systems, Google ADK, and
LangGraph. Corte 3 (weeks 12-16) covers evaluation, observability, and deployment to production.

Corte 3 is worth 40% of the final grade: 30% for a final integrator project and 10% for in-class activities.
The final project must combine agents, RAG, an automatic evaluation pipeline, tracing with Langfuse, and a
REST API exposed with FastAPI, delivered with a demo video and a technical presentation.
"""

# (question, expected-to-be-answerable-from-context) — a tiny evaluation dataset.
EVAL_DATASET = [
    "How much is Corte 3 worth, and what are its two graded components?",
    "What must the Corte 3 final project combine?",
    "How many total weeks does the course run?",           # answerable
    "What programming language is used for all assignments?",  # NOT in the context -> should decline
]


class JudgeScore(BaseModel):
    faithfulness: int = Field(ge=1, le=5, description="1=unsupported claims, 5=fully grounded in context")
    relevance: int = Field(ge=1, le=5, description="1=off-topic, 5=directly answers the question")
    rationale: str


def answer(llm, question: str) -> str:
    prompt = (
        f"Answer using ONLY this context. If the answer isn't in the context, say so explicitly.\n\n"
        f"Context:\n{COURSE_DOCS}\n\nQuestion: {question}"
    )
    return llm.invoke(prompt).content


def judge(judge_llm, question: str, context: str, generated_answer: str) -> JudgeScore:
    return judge_llm.invoke(
        f"Question: {question}\nContext: {context}\nAnswer: {generated_answer}\n\n"
        "Score the answer's faithfulness to the context (1-5) and relevance to the question (1-5), "
        "and give a one-sentence rationale."
    )


def main():
    llm = ChatOllama(model=CHAT_MODEL)
    judge_llm = llm.with_structured_output(JudgeScore)

    total_faithfulness = total_relevance = 0
    for question in EVAL_DATASET:
        generated = answer(llm, question)
        score = judge(judge_llm, question, COURSE_DOCS, generated)
        total_faithfulness += score.faithfulness
        total_relevance += score.relevance
        print(f"Q: {question}")
        print(f"A: {generated}")
        print(f"  faithfulness={score.faithfulness} relevance={score.relevance} :: {score.rationale}\n")

    n = len(EVAL_DATASET)
    print(f"Average faithfulness: {total_faithfulness / n:.2f} / 5")
    print(f"Average relevance:    {total_relevance / n:.2f} / 5")


if __name__ == "__main__":
    main()
