import pytest

from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools


def test_build_dtm_tool_known_answer_4():
    """Test build_dtm_tool against known answer #4 in TEST_FIXTURE.md.

    sample_fixture.csv produces a 8x5 matrix with vocabulary:
      ['alpha', 'always', 'beta', 'delta', 'gamma']
    Non-zero entries: 21 / 40 total elements -> sparsity_pct = 47.5%.
    """
    session = SessionState()

    # Load premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    # Load fixture dataset
    load_dataset_tool.invoke(
        {
            "file_path": "agent_dev/sample_fixture.csv",
            "text_column": "text",
            "label_column": "label",
        }
    )

    # Instantiate student tools from tools_dtm.py
    dtm_tools = make_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")

    # Run build_dtm_tool
    res = build_dtm_tool.invoke({})

    # Assert results against known answer #4
    assert res["n_documents"] == 8
    assert res["vocabulary_size"] == 5
    assert res["non_zero"] == 21
    assert res["total_elements"] == 40
    assert res["sparsity_pct"] == pytest.approx(47.5, abs=1e-3)

    # Verify session state updates
    assert session.feature_matrix is not None
    assert session.feature_matrix.shape == (8, 5)
    assert session.feature_names == ["alpha", "always", "beta", "delta", "gamma"]
    assert "count_vectorizer" in session.artifacts
    assert session.categories == ["catA", "catB"]


def test_term_frequency_tool_known_answer_14():
    """Test term_frequency_tool against known answer #14 in TEST_FIXTURE.md.

    Expected frequencies across all documents:
      alpha: 6, always: 7, beta: 3, delta: 4, gamma: 7
    """
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke(
        {
            "file_path": "agent_dev/sample_fixture.csv",
            "text_column": "text",
            "label_column": "label",
        }
    )

    dtm_tools = make_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    term_frequency_tool = next(t for t in dtm_tools if t.name == "term_frequency_tool")

    # Build DTM first
    build_dtm_tool.invoke({})

    # Run term_frequency_tool
    res = term_frequency_tool.invoke({})

    assert res["total_terms"] == 5
    assert res["returned_terms"] == 5

    freqs = res["term_frequencies"]
    assert freqs["alpha"] == 6
    assert freqs["always"] == 7
    assert freqs["beta"] == 3
    assert freqs["delta"] == 4
    assert freqs["gamma"] == 7


def test_dtm_heatmap_tool_known_answer_21():
    """Test dtm_heatmap_tool against known answer #21 in TEST_FIXTURE.md.

    sample_fixture.csv matrix (8 docs x 5 terms: alpha, always, beta, delta, gamma):
      Doc 0: [3, 1, 1, 0, 0]
      Doc 1: [2, 1, 1, 0, 0]
      Doc 2: [1, 1, 1, 0, 0]
      Doc 3: [0, 1, 0, 1, 3]
      Doc 4: [0, 1, 0, 1, 2]
      Doc 5: [0, 1, 0, 1, 1]
      Doc 6: [0, 1, 0, 1, 1]
      Doc 7: [0, 0, 0, 0, 0]

    A slice with n_terms=2, n_documents=3 should return:
      [[3, 1], [2, 1], [1, 1]]
    """
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke(
        {
            "file_path": "agent_dev/sample_fixture.csv",
            "text_column": "text",
            "label_column": "label",
        }
    )

    dtm_tools = make_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    dtm_heatmap_tool = next(t for t in dtm_tools if t.name == "dtm_heatmap_tool")

    build_dtm_tool.invoke({})

    # Test full matrix default slice (20x20 capped to 8x5)
    res_full = dtm_heatmap_tool.invoke({})
    assert res_full["n_documents"] == 8
    assert res_full["n_terms"] == 5
    assert res_full["terms"] == ["alpha", "always", "beta", "delta", "gamma"]
    assert res_full["matrix"][0] == [3, 1, 1, 0, 0]
    assert res_full["matrix"][7] == [0, 0, 0, 0, 0]
    assert session.pending_figure is not None

    # Test sliced matrix (n_terms=2, n_documents=3)
    session.pending_figure = None
    res_sliced = dtm_heatmap_tool.invoke({"n_terms": 2, "n_documents": 3})
    assert res_sliced["n_documents"] == 3
    assert res_sliced["n_terms"] == 2
    assert res_sliced["terms"] == ["alpha", "always"]
    assert res_sliced["matrix"] == [[3, 1], [2, 1], [1, 1]]
    assert session.pending_figure is not None
