import pytest

from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_filtering import make_tools as make_filtering_tools


def test_variance_filter_tool_known_answer_5():
    """Test variance_filter_tool against known answer #5 in TEST_FIXTURE.md.

    sample_fixture.csv term variances:
      alpha: 1.1875
      always: 0.109375
      beta: 0.234375
      delta: 0.25
      gamma: 1.109375

    Threshold 0.15 keeps alpha, beta, delta, gamma and removes always.
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

    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    filtering_tools = make_filtering_tools(session)
    variance_filter_tool = next(t for t in filtering_tools if t.name == "variance_filter_tool")

    res = variance_filter_tool.invoke({"threshold": 0.15})

    # Check term variances
    vars_dict = res["term_variances"]
    assert vars_dict["alpha"] == pytest.approx(1.1875, abs=1e-3)
    assert vars_dict["always"] == pytest.approx(0.109375, abs=1e-3)
    assert vars_dict["beta"] == pytest.approx(0.234375, abs=1e-3)
    assert vars_dict["delta"] == pytest.approx(0.25, abs=1e-3)
    assert vars_dict["gamma"] == pytest.approx(1.109375, abs=1e-3)

    # Check filtering behavior at threshold=0.15
    assert set(res["kept_terms"]) == {"alpha", "beta", "delta", "gamma"}
    assert res["removed_terms"] == ["always"]


def test_pearson_spearman_filter_tools_known_answer_6():
    """Test pearson_filter_tool and spearman_filter_tool against known answer #6 in TEST_FIXTURE.md.

    Target class 'catB' expected correlations:
      alpha: Pearson r ~ -0.6882, Spearman r ~ -0.7500
      always: Pearson r ~ 0.3780, Spearman r ~ 0.3780
      beta: Pearson r ~ -0.7746, Spearman r ~ -0.7746
      delta: Pearson r ~ 1.0000, Spearman r ~ 1.0000
      gamma: Pearson r ~ 0.8307, Spearman r ~ 0.9363
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

    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    filtering_tools = make_filtering_tools(session)
    pearson_filter_tool = next(t for t in filtering_tools if t.name == "pearson_filter_tool")
    spearman_filter_tool = next(t for t in filtering_tools if t.name == "spearman_filter_tool")

    # Pearson filter test for catB
    res_p = pearson_filter_tool.invoke({"target_class": "catB", "threshold": 0.0})
    p_corr = res_p["correlations"]
    assert p_corr["alpha"] == pytest.approx(-0.6882, abs=1e-3)
    assert p_corr["always"] == pytest.approx(0.3780, abs=1e-3)
    assert p_corr["beta"] == pytest.approx(-0.7746, abs=1e-3)
    assert p_corr["delta"] == pytest.approx(1.0000, abs=1e-3)
    assert p_corr["gamma"] == pytest.approx(0.8307, abs=1e-3)

    # Spearman filter test for catB
    res_s = spearman_filter_tool.invoke({"target_class": "catB", "threshold": 0.0})
    s_corr = res_s["correlations"]
    assert s_corr["alpha"] == pytest.approx(-0.7500, abs=1e-3)
    assert s_corr["always"] == pytest.approx(0.3780, abs=1e-3)
    assert s_corr["beta"] == pytest.approx(-0.7746, abs=1e-3)
    assert s_corr["delta"] == pytest.approx(1.0000, abs=1e-3)
    assert s_corr["gamma"] == pytest.approx(0.9363, abs=1e-3)
