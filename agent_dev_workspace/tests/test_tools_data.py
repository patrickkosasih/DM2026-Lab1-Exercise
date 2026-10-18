import matplotlib.pyplot as plt
import numpy as np
import pytest

from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_data import make_tools


def test_list_files_tool_known_answer_11():
    """Test list_files_tool against known answer #11 in TEST_FIXTURE.md.

    list_files_tool(subdirectory="newdataset") should return:
      - directories: []
      - files: ["Reddit-stock-sentiment.csv"]
    """
    session = SessionState()
    tools = make_tools(session)
    list_files_tool = next(t for t in tools if t.name == "list_files_tool")

    res = list_files_tool.invoke({"subdirectory": "newdataset"})

    assert "result_id" in res
    assert res.get("subdirectory") == "newdataset"
    assert res.get("directories") == []
    assert res.get("files") == ["Reddit-stock-sentiment.csv"]


def test_inspect_data_tool():
    """Test inspect_data_tool after loading sample_fixture.csv."""
    session = SessionState()

    # Load fixture dataset using premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Instantiate student's tool using the exact same session instance
    data_tools = make_tools(session)
    inspect_data_tool = next(t for t in data_tools if t.name == "inspect_data_tool")

    # Call inspect_data_tool with n=3
    res = inspect_data_tool.invoke({"n": 3, "mode": "head"})

    assert "result_id" in res
    assert res["n_documents"] == 8
    assert res["counts_per_category"] == {"catA": 4, "catB": 4}
    assert len(res["preview"]) == 3
    assert "text" in res["columns"]


def test_check_missing_tool_known_answer_2():
    """Test check_missing_tool against known answer #2 in TEST_FIXTURE.md.

    sample_fixture.csv has 8 rows and exactly 1 empty/missing row at index 7.
    """
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    data_tools = make_tools(session)
    check_missing_tool = next(t for t in data_tools if t.name == "check_missing_tool")

    res = check_missing_tool.invoke({})

    assert "result_id" in res
    assert res["n_documents"] == 8
    assert res["missing_count"] == 1
    assert res["missing_indices"] == [7]


def test_check_duplicates_tool_known_answer_3():
    """Test check_duplicates_tool against known answer #3 in TEST_FIXTURE.md.

    sample_fixture.csv has 8 rows; row 6 is an exact copy of row 5 ("always gamma delta").
    Dropping with keep=False removes both row 5 and row 6, leaving exactly 6 rows.
    """
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    data_tools = make_tools(session)
    check_duplicates_tool = next(t for t in data_tools if t.name == "check_duplicates_tool")

    # Step 1: Check without dropping
    res_check = check_duplicates_tool.invoke({"drop": False})

    assert "result_id" in res_check
    assert res_check["n_documents_before"] == 8
    assert res_check["duplicate_count"] == 1
    assert res_check["duplicate_indices"] == [6]
    assert res_check["dropped"] is False
    assert res_check["n_documents_after"] == 8

    # Step 2: Drop duplicates (keep=False removes both copies)
    res_drop = check_duplicates_tool.invoke({"drop": True})

    assert "result_id" in res_drop
    assert res_drop["n_documents_before"] == 8
    assert res_drop["duplicate_count"] == 1
    assert res_drop["dropped"] is True
    assert res_drop["n_documents_after"] == 6
    assert len(session.dataframe) == 6


def test_sample_data_tool_known_answer_13():
    """Test sample_data_tool against known answer #13 in TEST_FIXTURE.md.

    Calling sample_data_tool(n=4, random_state=42) on sample_fixture.csv returns
    exactly original rows [1, 5, 0, 7].
    """
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    data_tools = make_tools(session)
    sample_data_tool = next(t for t in data_tools if t.name == "sample_data_tool")

    res = sample_data_tool.invoke({"n": 4, "random_state": 42})

    assert "result_id" in res
    assert res["n_documents_before"] == 8
    assert res["n_documents_after"] == 4
    assert res["sampled_indices"] == [1, 5, 0, 7]
    assert len(session.dataframe) == 4


def test_describe_data_tool_known_answer_18():
    """Test describe_data_tool against known answer #18 in TEST_FIXTURE.md."""
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

    # Instantiate student tools from tools_data.py
    data_tools = make_tools(session)
    describe_data_tool = next(t for t in data_tools if t.name == "describe_data_tool")

    # Run describe_data_tool
    res = describe_data_tool.invoke({})

    # Check summary structure
    assert res["n_documents"] == 8

    # Check overall stats against known answer #18
    overall = res["stats_overall"]
    assert overall["count"] == 8
    assert overall["mean"] == pytest.approx(19.875, abs=1e-3)
    assert overall["std"] == pytest.approx(9.4330, abs=1e-3)
    assert overall["min"] == 0
    assert overall["25%"] == pytest.approx(17.75, abs=1e-3)
    assert overall["50%"] == pytest.approx(20.5, abs=1e-3)
    assert overall["75%"] == pytest.approx(25.25, abs=1e-3)
    assert overall["max"] == 30

    # Check per-category stats against known answer #18
    stats_cat = res["stats_by_category"]
    assert "catA" in stats_cat
    assert "catB" in stats_cat

    catA = stats_cat["catA"]
    assert catA["count"] == 4
    assert catA["mean"] == pytest.approx(17.25, abs=1e-3)
    assert catA["std"] == pytest.approx(12.5, abs=1e-3)

    catB = stats_cat["catB"]
    assert catB["count"] == 4
    assert catB["mean"] == pytest.approx(22.5, abs=1e-3)
    assert catB["std"] == pytest.approx(5.7446, abs=1e-3)

    # Verify that session.pending_figure was set and is a matplotlib Figure
    assert session.pending_figure is not None
    assert isinstance(session.pending_figure, plt.Figure)
