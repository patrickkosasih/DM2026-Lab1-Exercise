import numpy as np
import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_exploration import make_tools as make_exploration_tools


def setup_session_with_dtm():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    return session


def test_cosine_similarity_tool_known_answers():
    session = setup_session_with_dtm()
    tools = make_exploration_tools(session)
    cosine_tool = next(t for t in tools if t.name == "cosine_similarity_tool")

    # Known answer #16: 0 vs 1 (~0.9847), 0 vs 3 (~0.0909), 0 vs 7 (0.0)
    res_0_1 = cosine_tool.invoke({"doc_id1": 0, "doc_id2": 1})
    assert res_0_1["cosine_similarity"] == pytest.approx(0.9847, abs=1e-3)

    res_0_3 = cosine_tool.invoke({"doc_id1": 0, "doc_id2": 3})
    assert res_0_3["cosine_similarity"] == pytest.approx(0.0909, abs=1e-3)

    res_0_7 = cosine_tool.invoke({"doc_id1": 0, "doc_id2": 7})
    assert res_0_7["cosine_similarity"] == pytest.approx(0.0, abs=1e-3)


def test_feature_correlation_matrix_tool_known_answers():
    session = setup_session_with_dtm()
    tools = make_exploration_tools(session)
    corr_tool = next(t for t in tools if t.name == "feature_correlation_matrix_tool")

    res = corr_tool.invoke({"top_k": 20})

    # Known answer #19
    expected_terms = ["alpha", "gamma", "delta", "beta", "always"]
    assert res["terms"] == expected_terms
    assert session.pending_figure is not None

    corr = np.array(res["correlation_matrix"])
    
    # Check diagonal is 1.0
    assert np.allclose(np.diag(corr), 1.0, atol=1e-3)

    # alpha vs beta (indices 0 and 3) = 0.8885
    assert corr[0, 3] == pytest.approx(0.8885, abs=1e-3)
    # gamma vs delta (indices 1 and 2) = 0.8307
    assert corr[1, 2] == pytest.approx(0.8307, abs=1e-3)
