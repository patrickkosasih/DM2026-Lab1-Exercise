from typing import List, Optional
from langchain_core.tools import tool
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer


def make_tools(session):
    @tool
    def build_dtm_tool(
        ngram_range: List[int] = [1, 1],
        max_df: float = 1.0,
        min_df: float = 1.0,
        max_features: Optional[int] = None,
    ) -> dict:
        """Build the Document-Term Matrix (DTM) from session.dataframe text and report its sparsity.

        Uses CountVectorizer to construct session.feature_matrix and session.feature_names.
        Stores the fitted vectorizer in session.artifacts["count_vectorizer"] for downstream tools
        (like cosine_similarity_tool). Also updates session.labels and session.categories.

        Args:
            ngram_range: Lower and upper boundary of the range of n-values for n-grams extracted (e.g. [1, 1]).
            max_df: Ignore terms with a document frequency strictly higher than the given threshold.
            min_df: Ignore terms with a document frequency strictly lower than the given threshold.
            max_features: If given, build a vocabulary that only considers the top max_features ordered by term frequency.

        Returns:
            dict: Summary containing result_id, n_documents, vocabulary_size, non_zero, total_elements, and sparsity_pct.
        """
        if session.dataframe is None:
            return {"error": "No dataframe loaded in session."}

        df = session.dataframe
        if "text" not in df.columns:
            return {"error": "Dataframe missing 'text' column."}

        # Convert List[int] parameter to tuple for CountVectorizer
        ng_range = tuple(ngram_range) if isinstance(ngram_range, list) else (1, 1)

        # Handle min_df/max_df integer conversions if needed
        min_df_val = int(min_df) if isinstance(min_df, float) and min_df >= 1.0 else min_df
        max_df_val = int(max_df) if isinstance(max_df, float) and max_df > 1.0 else max_df

        # Initialize and fit CountVectorizer
        vectorizer = CountVectorizer(
            ngram_range=ng_range,
            max_df=max_df_val,
            min_df=min_df_val,
            max_features=max_features,
        )

        dtm = vectorizer.fit_transform(df["text"].fillna(""))
        feature_names = vectorizer.get_feature_names_out().tolist()

        # Store results in session state
        session.feature_matrix = dtm
        session.feature_names = feature_names
        session.artifacts["count_vectorizer"] = vectorizer

        # Set labels and categories in session if available
        if "category_name" in df.columns:
            session.set_labels(df["category_name"].tolist())
        elif "category" in df.columns:
            session.set_labels(df["category"].tolist())

        # Calculate matrix dimensions and sparsity
        n_rows, n_cols = dtm.shape
        non_zero = int(dtm.nnz)
        total_elements = int(n_rows * n_cols)
        sparsity_pct = (
            round(100.0 * (1.0 - (non_zero / total_elements)), 4)
            if total_elements > 0
            else 0.0
        )

        result_id = session.next_result_id("build_dtm")
        summary = {
            "result_id": result_id,
            "n_documents": n_rows,
            "vocabulary_size": n_cols,
            "non_zero": non_zero,
            "total_elements": total_elements,
            "sparsity_pct": sparsity_pct,
        }

        session.store_result("build_dtm_tool", {}, summary)
        return summary

    @tool
    def term_frequency_tool(top_n: Optional[int] = None) -> dict:
        """Aggregate term frequencies across all documents in session.feature_matrix.

        Sums occurrences of each term across the global document-term matrix and sorts terms
        by total count descending.

        Args:
            top_n: Optional integer to limit the returned terms to the top N most frequent.

        Returns:
            dict: Summary containing result_id, total_terms, returned_terms, and term_frequencies.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {
                "error": "No feature_matrix found in session. Run build_dtm_tool first."
            }

        # Sum counts across all documents (columns)
        totals = np.asarray(session.feature_matrix.sum(axis=0)).flatten()

        # Build DataFrame and sort descending
        freq_df = pd.DataFrame(
            {"term": session.feature_names, "frequency": totals}
        ).sort_values(by="frequency", ascending=False)

        total_terms = len(freq_df)

        if top_n is not None and top_n > 0:
            freq_df_sub = freq_df.head(top_n)
        else:
            freq_df_sub = freq_df

        term_freqs = dict(zip(freq_df_sub["term"], freq_df_sub["frequency"].astype(int)))

        result_id = session.next_result_id("term_frequency")
        summary = {
            "result_id": result_id,
            "total_terms": total_terms,
            "returned_terms": len(term_freqs),
            "term_frequencies": term_freqs,
        }

        session.store_result("term_frequency_tool", {"top_n": top_n}, summary, full_report=freq_df_sub)
        return summary

    @tool
    def dtm_heatmap_tool(n_terms: int = 20, n_documents: int = 20) -> dict:
        """Render a heatmap of a positional slice of the global Document-Term Matrix (DTM).

        Slices the raw DTM for the specified number of terms and documents without re-ordering,
        generates an annotated heatmap figure, and attaches it to session.pending_figure.

        Args:
            n_terms: Number of terms (columns) from session.feature_names to slice (default: 20).
            n_documents: Number of documents (rows) from session.feature_matrix to slice (default: 20).

        Returns:
            dict: Summary containing result_id, n_documents, n_terms, document_indices, terms, and matrix.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {
                "error": "No feature_matrix found in session. Run build_dtm_tool first."
            }

        # Determine slice bounds
        total_docs, total_terms = session.feature_matrix.shape
        sliced_docs = min(n_documents, total_docs)
        sliced_terms = min(n_terms, total_terms)

        # Slice sub-matrix and term names
        sub_matrix = session.feature_matrix[0:sliced_docs, 0:sliced_terms].toarray()
        sub_terms = session.feature_names[0:sliced_terms]
        doc_indices = list(range(sliced_docs))

        # Build heatmap visualization
        fig, ax = plt.subplots(figsize=(max(6, sliced_terms * 0.5), max(4, sliced_docs * 0.4)))
        sns.heatmap(
            sub_matrix,
            annot=True,
            fmt="d",
            cmap="YlGnBu",
            xticklabels=sub_terms,
            yticklabels=[f"Doc {i}" for i in doc_indices],
            ax=ax,
            cbar=True,
        )
        ax.set_title("Document-Term Matrix (DTM) Heatmap")
        ax.set_xlabel("Terms")
        ax.set_ylabel("Documents")
        plt.tight_layout()

        # Attach figure to session state for rendering
        session.pending_figure = fig

        result_id = session.next_result_id("dtm_heatmap")
        summary = {
            "result_id": result_id,
            "n_documents": sliced_docs,
            "n_terms": sliced_terms,
            "document_indices": doc_indices,
            "terms": sub_terms,
            "matrix": sub_matrix.tolist(),
        }

        session.store_result(
            "dtm_heatmap_tool",
            {"n_terms": n_terms, "n_documents": n_documents},
            summary,
        )
        return summary

    return [build_dtm_tool, term_frequency_tool, dtm_heatmap_tool]
