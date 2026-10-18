from typing import Any, Dict, List, Literal, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from langchain_core.tools import tool
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import OneHotEncoder


def make_tools(session) -> List[Any]:
    @tool
    def reduce_dimensions_tool(
        method: Literal["pca", "tsne", "umap"] = "pca",
        n_components: int = 2,
        perplexity: float = 30.0,
        n_neighbors: int = 15,
        random_state: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Reduces feature matrix dimensionality using PCA, t-SNE, or UMAP and plots the result.

        Args:
            method: Reduction algorithm to use ('pca', 'tsne', or 'umap'). Default 'pca'.
            n_components: Number of target dimensions (typically 2). Default 2.
            perplexity: Perplexity parameter for t-SNE. Default 30.0.
            n_neighbors: Number of neighbors for UMAP. Default 15.
            random_state: Optional random seed for deterministic/reproducible reduction.

        Returns:
            Dict containing result_id, method, coordinates shape, and meexplained variance (PCA).
        """
        if session.feature_matrix is None:
            return {"status": "error", "message": "No feature matrix found in session. Run build_dtm_tool first."}

        # Convert sparse DTM matrix to dense array if necessary
        X = session.feature_matrix
        if hasattr(X, "toarray"):
            X = X.toarray()

        if method == "pca":
            reducer = PCA(n_components=n_components, random_state=random_state)
            coords = reducer.fit_transform(X)
            explained_variance = reducer.explained_variance_ratio_.tolist()
        elif method == "tsne":
            reducer = TSNE(
                n_components=n_components,
                perplexity=perplexity,
                random_state=random_state,
            )
            coords = reducer.fit_transform(X)
            explained_variance = None
        elif method == "umap":
            import umap
            reducer = umap.UMAP(
                n_components=n_components,
                n_neighbors=n_neighbors,
                random_state=random_state,
            )
            coords = reducer.fit_transform(X)
            explained_variance = None
        else:
            return {"status": "error", "message": f"Unsupported method: {method}"}

        # Save coordinates to session artifacts
        session.artifacts[f"reduced_{method}"] = coords

        # Generate plot: 2D scatter plot colored by category
        fig, ax = plt.subplots(figsize=(8, 6))

        # Get labels / category names for coloring
        if session.dataframe is not None and "category_name" in session.dataframe.columns:
            categories = session.dataframe["category_name"].values
        elif session.labels is not None:
            categories = session.labels
        else:
            categories = ["default"] * coords.shape[0]

        unique_cats = np.unique(categories)
        for cat in unique_cats:
            mask = np.array(categories) == cat
            ax.scatter(coords[mask, 0], coords[mask, 1], label=str(cat), alpha=0.7, edgecolors="k")

        ax.set_title(f"Dimensionality Reduction ({method.upper()})")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.legend(title="Category")
        ax.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()

        session.pending_figure = fig

        result_id = session.next_result_id("reduce_dimensions")
        summary = {
            "result_id": result_id,
            "method": method,
            "n_components": n_components,
            "n_samples": int(coords.shape[0]),
            "coordinates": coords.tolist(),
        }
        if explained_variance is not None:
            summary["explained_variance_ratio"] = explained_variance

        session.store_result("reduce_dimensions_tool", {"method": method, "n_components": n_components}, summary)

        return summary

    @tool
    def binarize_labels_tool() -> Dict[str, Any]:
        """Converts categorical dataset labels into a one-hot binary matrix representation.

        Returns:
            Dict containing result_id, n_samples, n_classes, category names/classes, and one-hot matrix.
        """
        if session.dataframe is None or "category" not in session.dataframe.columns:
            if session.labels is None:
                return {"status": "error", "message": "No categorical labels found in session. Load a dataset first."}

        if session.dataframe is not None and "category" in session.dataframe.columns:
            labels_raw = session.dataframe["category"].values.reshape(-1, 1)
        else:
            labels_raw = np.array(session.labels).reshape(-1, 1)

        encoder = OneHotEncoder(sparse_output=False)
        one_hot_matrix = encoder.fit_transform(labels_raw)

        categories = session.categories if session.categories else list(np.unique(labels_raw.flatten()))
        # Convert NumPy integers / floats to standard Python types for JSON serialization
        categories_serializable = [str(c) if not isinstance(c, (int, str)) else c for c in categories]

        session.artifacts["one_hot_labels"] = one_hot_matrix

        result_id = session.next_result_id("binarize_labels")
        summary = {
            "result_id": result_id,
            "n_samples": int(one_hot_matrix.shape[0]),
            "n_classes": int(one_hot_matrix.shape[1]),
            "categories": categories_serializable,
            "one_hot_matrix": one_hot_matrix.astype(int).tolist(),
        }

        session.store_result("binarize_labels_tool", {}, summary)

        return summary

    return [reduce_dimensions_tool, binarize_labels_tool]
