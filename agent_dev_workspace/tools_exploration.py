from typing import Any, Dict, List
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from langchain_core.tools import tool
from sklearn.metrics.pairwise import cosine_similarity


def make_tools(session) -> List[Any]:
    @tool
    def cosine_similarity_tool(
        doc_id1: int,
        doc_id2: int,
    ) -> Dict[str, Any]:
        """Computes cosine similarity between two documents in the feature matrix.

        Args:
            doc_id1: Zero-based row index of the first document.
            doc_id2: Zero-based row index of the second document.

        Returns:
            Dict containing result_id, doc_id1, doc_id2, and cosine_similarity score.
        """
        if session.feature_matrix is None:
            return {"status": "error", "message": "No feature matrix found in session. Run build_dtm_tool first."}

        n_docs = session.feature_matrix.shape[0]
        if doc_id1 < 0 or doc_id1 >= n_docs or doc_id2 < 0 or doc_id2 >= n_docs:
            return {"status": "error", "message": f"Document indices must be between 0 and {n_docs - 1}."}

        vec1 = session.feature_matrix[doc_id1]
        vec2 = session.feature_matrix[doc_id2]

        sim_matrix = cosine_similarity(vec1, vec2)
        sim_val = float(sim_matrix[0, 0])

        result_id = session.next_result_id("cosine_sim")
        summary = {
            "result_id": result_id,
            "doc_id1": doc_id1,
            "doc_id2": doc_id2,
            "cosine_similarity": sim_val,
        }

        session.store_result("cosine_similarity_tool", {"doc_id1": doc_id1, "doc_id2": doc_id2}, summary)

        return summary

    @tool
    def feature_correlation_matrix_tool(
        top_k: int = 20,
    ) -> Dict[str, Any]:
        """Computes Pearson correlation matrix over the top-k terms by variance and plots a heatmap.

        Args:
            top_k: Number of highest-variance terms to include (default 20).

        Returns:
            Dict containing result_id, top feature names, and correlation matrix values.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"status": "error", "message": "No feature matrix found in session. Run build_dtm_tool first."}

        X = session.feature_matrix
        if hasattr(X, "toarray"):
            X = X.toarray()

        variances = np.var(X, axis=0)
        k = min(top_k, X.shape[1])

        # Get indices of top-k terms sorted by variance descending
        top_indices = np.argsort(variances)[::-1][:k]
        top_terms = [session.feature_names[i] for i in top_indices]

        X_top = X[:, top_indices]
        # np.corrcoef expects variables as rows, so transpose X_top (samples x features -> features x samples)
        corr_matrix = np.corrcoef(X_top, rowvar=False)

        # Handle edge case where corrcoef returns a scalar if top_k == 1 or single term
        if np.ndim(corr_matrix) == 0:
            corr_matrix = np.array([[float(corr_matrix)]])
        # Replace NaN correlation values (e.g., constant features) with 0.0
        corr_matrix = np.nan_to_num(corr_matrix)

        # Build Seaborn heatmap figure
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            xticklabels=top_terms,
            yticklabels=top_terms,
            cmap="coolwarm",
            ax=ax,
            cbar=True,
        )
        ax.set_title(f"Feature Correlation Heatmap (Top {k} Terms by Variance)")
        plt.tight_layout()

        session.pending_figure = fig

        result_id = session.next_result_id("feature_corr")
        summary = {
            "result_id": result_id,
            "top_k": k,
            "terms": top_terms,
            "correlation_matrix": corr_matrix.tolist(),
        }

        session.store_result("feature_correlation_matrix_tool", {"top_k": top_k}, summary)

        return summary

    return [cosine_similarity_tool, feature_correlation_matrix_tool]
