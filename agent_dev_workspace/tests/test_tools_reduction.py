import numpy as np
import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_reduction import make_tools


def test_reduce_dimensions_pca_known_answer():
    session = SessionState()

    # Load premade tools and invoke load_dataset_tool on sample_fixture.csv
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Build DTM using build_dtm_tool from tools_dtm
    from tools_dtm import make_tools as make_dtm_tools
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    # Build reduction tools
    tools = make_tools(session)
    reduce_dimensions_tool = next(t for t in tools if t.name == "reduce_dimensions_tool")

    # Test PCA reduction (Known answer #7)
    res = reduce_dimensions_tool.invoke({"method": "pca", "n_components": 2})

    expected_coords = np.array([
        [2.3669, 0.8871],
        [1.7039, 0.2642],
        [1.0408, -0.3587],
        [-2.0759, 1.0129],
        [-1.4554, 0.3405],
        [-0.8349, -0.3319],
        [-0.8349, -0.3319],
        [0.0894, -1.4822],
    ])
    expected_explained_var = [0.7532, 0.1965]

    coords = np.array(res["coordinates"])
    explained_var = res["explained_variance_ratio"]

    assert np.allclose(coords, expected_coords, atol=1e-3)
    assert np.allclose(explained_var, expected_explained_var, atol=1e-3)


def test_binarize_labels_tool():
    session = SessionState()

    # Load premade tools and load sample_fixture.csv
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Test binarize_labels_tool
    tools = make_tools(session)
    binarize_labels_tool = next(t for t in tools if t.name == "binarize_labels_tool")

    res = binarize_labels_tool.invoke({})

    assert res["n_samples"] == 8
    assert res["n_classes"] == 2
    assert "one_hot_matrix" in res

    matrix = np.array(res["one_hot_matrix"])
    assert matrix.shape == (8, 2)
    # Check that each document has exactly one 1 in its row (one-hot property)
    assert np.all(matrix.sum(axis=1) == 1)
    # Check that stored artifacts contain the matrix
    assert "one_hot_labels" in session.artifacts
