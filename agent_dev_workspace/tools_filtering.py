from typing import Optional
from langchain_core.tools import tool
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def make_tools(session):
    @tool
    def variance_filter_tool(threshold: float = 0.1) -> dict:
        """Filter terms based on their variance across all documents in session.feature_matrix.

        Calculates per-term variance across the global DTM, sorts terms descending by variance,
        and identifies terms below the specified variance threshold.

        Args:
            threshold: Minimum variance required to keep a term (default: 0.1).

        Returns:
            dict: Summary containing result_id, threshold, total_terms, kept_terms, removed_terms, and term_variances.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {
                "error": "No feature_matrix found in session. Run build_dtm_tool first."
            }

        matrix_dense = session.feature_matrix.toarray()
        variances = np.var(matrix_dense, axis=0, ddof=0)

        df_report = pd.DataFrame(
            {"term": session.feature_names, "variance": variances}
        ).sort_values(by="variance", ascending=False)

        kept_df = df_report[df_report["variance"] >= threshold]
        removed_df = df_report[df_report["variance"] < threshold]

        kept_terms = kept_df["term"].tolist()
        removed_terms = removed_df["term"].tolist()

        result_id = session.next_result_id("variance_filter")
        summary = {
            "result_id": result_id,
            "threshold": threshold,
            "total_terms": len(df_report),
            "n_kept": len(kept_terms),
            "n_removed": len(removed_terms),
            "kept_terms": kept_terms,
            "removed_terms": removed_terms,
            "term_variances": dict(zip(df_report["term"], df_report["variance"].round(6))),
        }

        # Store full_report dataframe with term as column 1 and variance as column 2 for visualize_result_tool
        session.store_result(
            "variance_filter_tool",
            {"threshold": threshold},
            summary,
            full_report=df_report,
        )
        return summary

    @tool
    def pearson_filter_tool(target_class: str, threshold: float = 0.0) -> dict:
        """Filter terms by Pearson linear correlation with a target class label.

        Computes Pearson correlation coefficient (r) between each term's document frequency count
        and a binary target indicator for the specified class.

        Args:
            target_class: Target category label to correlate against (e.g. "catB").
            threshold: Minimum absolute correlation required to keep a term (default: 0.0).

        Returns:
            dict: Summary containing result_id, target_class, threshold, total_terms, kept_terms, and correlations.
        """
        if session.feature_matrix is None or session.labels is None:
            return {
                "error": "No feature_matrix or labels found in session. Run build_dtm_tool first."
            }

        labels = np.array(session.labels)
        if target_class not in labels:
            return {"error": f"Target class '{target_class}' not found in session labels."}

        # Binary indicator vector for target class (one-vs-rest)
        y = (labels == target_class).astype(float)
        matrix_dense = session.feature_matrix.toarray()

        correlations = []
        for col_idx in range(matrix_dense.shape[1]):
            x = matrix_dense[:, col_idx].astype(float)
            if np.std(x) == 0 or np.std(y) == 0:
                r = 0.0
            else:
                r = float(np.corrcoef(x, y)[0, 1])
                if np.isnan(r):
                    r = 0.0
            correlations.append(r)

        df_report = pd.DataFrame(
            {"term": session.feature_names, "pearson_r": correlations}
        )
        # Sort descending by absolute correlation magnitude
        df_report["abs_r"] = df_report["pearson_r"].abs()
        df_report = df_report.sort_values(by="abs_r", ascending=False).drop(columns=["abs_r"])

        kept_df = df_report[df_report["pearson_r"].abs() >= threshold]
        removed_df = df_report[df_report["pearson_r"].abs() < threshold]

        result_id = session.next_result_id("pearson_filter")
        summary = {
            "result_id": result_id,
            "target_class": target_class,
            "threshold": threshold,
            "total_terms": len(df_report),
            "n_kept": len(kept_df),
            "n_removed": len(removed_df),
            "kept_terms": kept_df["term"].tolist(),
            "removed_terms": removed_df["term"].tolist(),
            "correlations": dict(zip(df_report["term"], df_report["pearson_r"].round(6))),
        }

        session.store_result(
            "pearson_filter_tool",
            {"target_class": target_class, "threshold": threshold},
            summary,
            full_report=df_report,
        )
        return summary

    @tool
    def spearman_filter_tool(target_class: str, threshold: float = 0.0) -> dict:
        """Filter terms by Spearman rank correlation with a target class label.

        Computes Spearman correlation coefficient (rho) between each term's document count
        and a binary target indicator for the specified category.

        Args:
            target_class: Target category label to correlate against (e.g. "catB").
            threshold: Minimum absolute rank correlation required to keep a term (default: 0.0).

        Returns:
            dict: Summary containing result_id, target_class, threshold, total_terms, kept_terms, and correlations.
        """
        if session.feature_matrix is None or session.labels is None:
            return {
                "error": "No feature_matrix or labels found in session. Run build_dtm_tool first."
            }

        labels = np.array(session.labels)
        if target_class not in labels:
            return {"error": f"Target class '{target_class}' not found in session labels."}

        y = (labels == target_class).astype(float)
        matrix_dense = session.feature_matrix.toarray()

        correlations = []
        for col_idx in range(matrix_dense.shape[1]):
            x = matrix_dense[:, col_idx].astype(float)
            res = spearmanr(x, y)
            r = float(res.statistic) if not np.isnan(res.statistic) else 0.0
            correlations.append(r)

        df_report = pd.DataFrame(
            {"term": session.feature_names, "spearman_r": correlations}
        )
        df_report["abs_r"] = df_report["spearman_r"].abs()
        df_report = df_report.sort_values(by="abs_r", ascending=False).drop(columns=["abs_r"])

        kept_df = df_report[df_report["spearman_r"].abs() >= threshold]
        removed_df = df_report[df_report["spearman_r"].abs() < threshold]

        result_id = session.next_result_id("spearman_filter")
        summary = {
            "result_id": result_id,
            "target_class": target_class,
            "threshold": threshold,
            "total_terms": len(df_report),
            "n_kept": len(kept_df),
            "n_removed": len(removed_df),
            "kept_terms": kept_df["term"].tolist(),
            "removed_terms": removed_df["term"].tolist(),
            "correlations": dict(zip(df_report["term"], df_report["spearman_r"].round(6))),
        }

        session.store_result(
            "spearman_filter_tool",
            {"target_class": target_class, "threshold": threshold},
            summary,
            full_report=df_report,
        )
        return summary

    return [variance_filter_tool, pearson_filter_tool, spearman_filter_tool]
