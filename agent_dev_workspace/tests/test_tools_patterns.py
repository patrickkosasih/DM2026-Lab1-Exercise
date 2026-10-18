import pytest
import numpy as np
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_patterns import make_tools


def setup_fixture_session():
    session = SessionState()
    premade, _ = load_workspace_tools("premade_tools", session)
    load_dataset = next(t for t in premade if t.name == "load_dataset_tool")
    load_dataset.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    return session


def test_mine_patterns_zero_survival():
    session = setup_fixture_session()
    tools = make_tools(session)
    mine_tool = next(t for t in tools if t.name == "mine_patterns_tool")

    # Known answer #17: variance and term_frequency keep zero terms for both categories
    res_var = mine_tool.invoke({
        "category_name": "catA",
        "filtering_method": "variance",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })
    assert res_var.get("patterns") == []
    assert res_var.get("note") == "no terms survived filtering"

    res_tf = mine_tool.invoke({
        "category_name": "catB",
        "filtering_method": "term_frequency",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })
    assert res_tf.get("patterns") == []
    assert res_tf.get("note") == "no terms survived filtering"


def test_mine_patterns_fpgrowth_tfidf():
    session = setup_fixture_session()
    tools = make_tools(session)
    mine_tool = next(t for t in tools if t.name == "mine_patterns_tool")

    # Known answer #17: tfidf filtering + FPGrowth
    # catA -> keeps alpha, support 3
    res_catA = mine_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })
    patterns_A = res_catA.get("patterns", [])
    assert len(patterns_A) == 1
    assert patterns_A[0]["pattern"] == ["alpha"]
    assert patterns_A[0]["support"] == pytest.approx(3)

    # catB -> keeps gamma, support 4
    res_catB = mine_tool.invoke({
        "category_name": "catB",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })
    patterns_B = res_catB.get("patterns", [])
    assert len(patterns_B) == 1
    assert patterns_B[0]["pattern"] == ["gamma"]
    assert patterns_B[0]["support"] == pytest.approx(4)


def test_mine_patterns_topk_and_maxfpgrowth():
    session = setup_fixture_session()
    tools = make_tools(session)
    mine_tool = next(t for t in tools if t.name == "mine_patterns_tool")

    # Known answer #20: topk -> returns 1 pattern for both categories at k=1
    res_topk = mine_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "topk",
        "k": 1
    })
    patterns_topk = res_topk.get("patterns", [])
    assert len(patterns_topk) == 1
    assert patterns_topk[0]["pattern"] == ["alpha"]
    assert patterns_topk[0]["support"] == pytest.approx(3)

    # Known answer #20: maxfpgrowth -> returns 0 patterns for both categories
    res_max = mine_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "maxfpgrowth",
        "min_sup": 1
    })
    assert res_max.get("patterns", []) == []
