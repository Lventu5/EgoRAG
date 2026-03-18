import importlib.util
from pathlib import Path

from data.query import Query


MODULE_PATH = Path(__file__).resolve().parents[1] / "retrieval" / "query_orchestrator.py"
SPEC = importlib.util.spec_from_file_location("query_orchestrator_module", MODULE_PATH)
QUERY_ORCHESTRATOR_MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(QUERY_ORCHESTRATOR_MODULE)
RetrievalOrchestratorLLM = QUERY_ORCHESTRATOR_MODULE.RetrievalOrchestratorLLM


def _build_query(query_text: str, query_date: str = "DAY4", query_time_sec: float = 15 * 3600) -> Query:
    query = Query(qid="q1", query_text=query_text, video_uid="A1_JAKE_DAY4")
    query.decomposed = {
        "text": query_text,
        "audio": query_text,
        "video": query_text,
        "metadata": {
            "query_date": query_date,
            "query_time_sec": query_time_sec,
            "need_audio": False,
        },
    }
    return query


def test_planner_resolves_yesterday_morning():
    planner = RetrievalOrchestratorLLM(
        model_name="Qwen/Qwen2.5-3B-Instruct",
        use_llm=False,
        device="cpu",
    )
    query = _build_query("Who was I talking to yesterday morning?")

    plan = planner.plan_query(
        query=query,
        modalities=["text", "video", "audio"],
        default_use_windows=False,
        rewrite_queries=True,
    )

    assert plan["planner"] == "heuristic"
    assert plan["temporal"]["allowed_days"] == ["DAY3"]
    assert plan["temporal"]["time_of_day"] == "morning"
    assert plan["temporal"]["relation_to_query_time"] == "unrestricted"
    assert plan["use_windows"] is True
    assert "day_filter" in plan["workflow"]
    assert "time_range_filter" in plan["workflow"]


def test_planner_detects_future_same_day_period():
    planner = RetrievalOrchestratorLLM(
        model_name="Qwen/Qwen2.5-3B-Instruct",
        use_llm=False,
        device="cpu",
    )
    query = _build_query("What will I do this afternoon?", query_date="DAY2", query_time_sec=9 * 3600)

    plan = planner.plan_query(
        query=query,
        modalities=["text", "video"],
        default_use_windows=False,
        rewrite_queries=False,
    )

    assert plan["temporal"]["allowed_days"] == ["DAY2"]
    assert plan["temporal"]["time_of_day"] == "afternoon"
    assert plan["temporal"]["relation_to_query_time"] == "after_query_time"
    assert plan["use_windows"] is True
    assert "after_query_time_filter" in plan["workflow"]


def test_planner_defaults_to_same_day_past_filter():
    planner = RetrievalOrchestratorLLM(
        model_name="Qwen/Qwen2.5-3B-Instruct",
        use_llm=False,
        device="cpu",
    )
    query = _build_query("Where did I leave my keys earlier today?", query_date="DAY5", query_time_sec=17 * 3600)

    plan = planner.plan_query(
        query=query,
        modalities=["text", "video"],
        default_use_windows=False,
        rewrite_queries=True,
    )

    assert plan["temporal"]["allowed_days"] == ["DAY5"]
    assert plan["temporal"]["relation_to_query_time"] == "before_query_time"
    assert "before_query_time_filter" in plan["workflow"]
