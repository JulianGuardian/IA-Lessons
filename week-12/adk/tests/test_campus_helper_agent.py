"""Programmatic (pytest) version of the same eval — useful wired into CI.

Run from the `adk/` directory: pytest tests/
"""

import pytest
from google.adk.evaluation.agent_evaluator import AgentEvaluator


@pytest.mark.asyncio
async def test_campus_helper_agent():
    await AgentEvaluator.evaluate(
        agent_module="campus_helper_agent",
        eval_dataset_file_path_or_dir="evalsets/campus_helper.evalset.json",
    )
