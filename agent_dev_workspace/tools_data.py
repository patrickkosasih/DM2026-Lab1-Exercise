import os
from typing import Optional
import matplotlib.pyplot as plt
import nltk
import pandas as pd
import seaborn as sns
from langchain_core.tools import tool

# Download punkt tokenizer data quietly if needed
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)


def make_tools(session):
    @tool
    def tokenize_tool() -> dict:
        """Tokenize each document's text into unigrams using NLTK word_tokenize.

        Reads session.dataframe, lowercases the text column, and applies
        nltk.word_tokenize to split text into unigram tokens. Stores the resulting
        list of token lists in session.dataframe["unigrams"] and session.artifacts["unigrams"].

        Returns:
            dict: Summary containing result_id, n_documents, total_tokens, and avg_tokens_per_doc.
        """
        if session.dataframe is None:
            return {"error": "No dataframe loaded in session."}

        df = session.dataframe
        if "text" not in df.columns:
            return {"error": "Dataframe missing 'text' column."}

        # Lowercase and tokenize text column
        unigrams = [
            nltk.word_tokenize(str(text).lower()) if pd.notna(text) and str(text).strip() else []
            for text in df["text"]
        ]

        # Store in session state
        df["unigrams"] = unigrams
        session.artifacts["unigrams"] = unigrams

        total_tokens = sum(len(tokens) for tokens in unigrams)
        n_documents = len(df)
        avg_tokens = total_tokens / n_documents if n_documents > 0 else 0.0

        result_id = session.next_result_id("tokenize")
        summary = {
            "result_id": result_id,
            "n_documents": n_documents,
            "total_tokens": total_tokens,
            "avg_tokens_per_doc": round(avg_tokens, 4),
        }

        session.store_result("tokenize_tool", {}, summary)
        return summary

    @tool
    def list_files_tool(subdirectory: str = "") -> dict:
        """List files and subdirectories in a directory relative to the project root.

        Useful for discovering dataset files or workspace paths before attempting to load them.

        Args:
            subdirectory: Subdirectory path relative to the root directory (defaults to "" for root).

        Returns:
            dict: Summary containing result_id, subdirectory, directories, and files list.
        """
        target_dir = os.path.join(".", subdirectory) if subdirectory else "."

        if not os.path.exists(target_dir):
            return {"error": f"Directory '{subdirectory}' does not exist."}

        if not os.path.isdir(target_dir):
            return {"error": f"Path '{subdirectory}' is not a directory."}

        items = os.listdir(target_dir)
        directories = sorted([d for d in items if os.path.isdir(os.path.join(target_dir, d))])
        files = sorted([f for f in items if os.path.isfile(os.path.join(target_dir, f))])

        result_id = session.next_result_id("list_files")
        summary = {
            "result_id": result_id,
            "subdirectory": subdirectory,
            "directories": directories,
            "files": files,
        }

        session.store_result("list_files_tool", {"subdirectory": subdirectory}, summary)
        return summary

    @tool
    def inspect_data_tool(n: int = 5, mode: str = "head") -> dict:
        """Preview head or tail rows of the current dataframe and review general summary stats.

        Args:
            n: Number of sample rows to return (default 5).
            mode: Either 'head' or 'tail' to select preview rows (default 'head').

        Returns:
            dict: Summary containing result_id, n_documents, shape, columns, counts_per_category, and preview rows.
        """
        if session.dataframe is None:
            return {"error": "No dataframe loaded in session."}

        df = session.dataframe

        if mode == "tail":
            preview_df = df.tail(n)
        else:
            preview_df = df.head(n)

        # Convert NaNs/Nones to None or string for clean JSON serialization
        preview_rows = preview_df.where(pd.notnull(preview_df), None).to_dict(orient="records")

        # Category counts summary if label/category columns exist
        counts_per_category = {}
        if "category_name" in df.columns:
            counts_per_category = df["category_name"].value_counts().to_dict()
        elif "category" in df.columns:
            counts_per_category = df["category"].value_counts().to_dict()

        result_id = session.next_result_id("inspect_data")
        summary = {
            "result_id": result_id,
            "n_documents": len(df),
            "shape": list(df.shape),
            "columns": list(df.columns),
            "counts_per_category": counts_per_category,
            "preview": preview_rows,
        }

        session.store_result("inspect_data_tool", {"n": n, "mode": mode}, summary)
        return summary

    @tool
    def check_missing_tool() -> dict:
        """Check for missing or empty text values in the current dataframe.

        Scans session.dataframe['text'] for null/NaN values or empty/whitespace-only strings.

        Returns:
            dict: Summary containing result_id, n_documents, missing_count, and missing_indices.
        """
        if session.dataframe is None:
            return {"error": "No dataframe loaded in session."}

        df = session.dataframe
        if "text" not in df.columns:
            return {"error": "Dataframe missing 'text' column."}

        # Identify missing or empty/whitespace-only text values
        missing_mask = df["text"].apply(
            lambda x: pd.isna(x) or not str(x).strip()
        )
        missing_indices = df.index[missing_mask].tolist()

        result_id = session.next_result_id("check_missing")
        summary = {
            "result_id": result_id,
            "n_documents": len(df),
            "missing_count": len(missing_indices),
            "missing_indices": [int(idx) for idx in missing_indices],
        }

        session.store_result("check_missing_tool", {}, summary)
        return summary

    @tool
    def check_duplicates_tool(drop: bool = False) -> dict:
        """Check for (and optionally drop) duplicate documents based on exact text match.

        Scans session.dataframe['text'] for duplicates. When drop=True, removes ALL copies
        of duplicated text (using keep=False) to match Master notebook behavior.

        Args:
            drop: If True, drops all copies of duplicate rows from session.dataframe.

        Returns:
            dict: Summary with result_id, n_documents_before, duplicate_count,
                  duplicate_indices, dropped status, and n_documents_after.
        """
        if session.dataframe is None:
            return {"error": "No dataframe loaded in session."}

        df = session.dataframe
        if "text" not in df.columns:
            return {"error": "Dataframe missing 'text' column."}

        # Identify duplicate rows (using keep='first' for reporting count/indices)
        dup_mask = df["text"].duplicated(keep="first")
        duplicate_indices = df.index[dup_mask].tolist()
        duplicate_count = len(duplicate_indices)

        n_before = len(df)

        if drop and duplicate_count > 0:
            # Match Master notebook behavior: drop_duplicates(keep=False) removes ALL duplicate copies
            session.dataframe = df.drop_duplicates(subset=["text"], keep=False).reset_index(drop=True)

        n_after = len(session.dataframe)

        result_id = session.next_result_id("check_duplicates")
        summary = {
            "result_id": result_id,
            "n_documents_before": n_before,
            "duplicate_count": duplicate_count,
            "duplicate_indices": [int(idx) for idx in duplicate_indices],
            "dropped": drop,
            "n_documents_after": n_after,
        }

        session.store_result("check_duplicates_tool", {"drop": drop}, summary)
        return summary

    @tool
    def sample_data_tool(
        n: Optional[int] = None,
        frac: Optional[float] = None,
        random_state: Optional[int] = None,
    ) -> dict:
        """Subsample the current working dataframe in session.dataframe.

        Args:
            n: Number of items/documents to sample.
            frac: Fraction of items/documents to sample (e.g. 0.5 for 50%).
            random_state: Random seed for reproducible sampling.

        Returns:
            dict: Summary containing result_id, n_documents_before, n_documents_after,
                  and sampled_indices.
        """
        if session.dataframe is None:
            return {"error": "No dataframe loaded in session."}

        df = session.dataframe
        n_before = len(df)

        if n is None and frac is None:
            return {"error": "Must specify either 'n' or 'frac' for sampling."}

        sampled_df = df.sample(n=n, frac=frac, random_state=random_state)
        sampled_indices = sampled_df.index.tolist()

        session.dataframe = sampled_df.reset_index(drop=True)
        n_after = len(session.dataframe)

        result_id = session.next_result_id("sample_data")
        summary = {
            "result_id": result_id,
            "n_documents_before": n_before,
            "n_documents_after": n_after,
            "sampled_indices": [int(idx) for idx in sampled_indices],
        }

        session.store_result(
            "sample_data_tool",
            {"n": n, "frac": frac, "random_state": random_state},
            summary,
        )
        return summary

    @tool
    def describe_data_tool() -> dict:
        """Compute summary statistics for document text lengths and generate a boxplot by category.

        Calculates text length (character count) for each document, computes standard summary
        statistics overall and per category, and creates a boxplot assigned to session.pending_figure.

        Returns:
            dict: Summary containing result_id, n_documents, stats_overall, and stats_by_category.
        """
        if session.dataframe is None:
            return {"error": "No dataframe loaded in session."}

        df = session.dataframe
        if "text" not in df.columns:
            return {"error": "Dataframe missing 'text' column."}

        # Calculate character length for each document
        df_copy = df.copy()
        df_copy["text_length"] = df_copy["text"].astype(str).apply(len)

        # Overall descriptive stats
        overall_stats = df_copy["text_length"].describe().to_dict()
        overall_stats = {k: float(round(v, 4)) for k, v in overall_stats.items()}

        # Per-category descriptive stats
        cat_col = (
            "category_name"
            if "category_name" in df_copy.columns
            else ("category" if "category" in df_copy.columns else None)
        )
        stats_by_category = {}
        if cat_col:
            grouped = df_copy.groupby(cat_col)["text_length"].describe()
            for cat_name, row in grouped.iterrows():
                stats_by_category[str(cat_name)] = {
                    k: float(round(v, 4)) for k, v in row.to_dict().items()
                }

        # Create boxplot figure
        fig, ax = plt.subplots(figsize=(6, 4))
        if cat_col:
            sns.boxplot(data=df_copy, x=cat_col, y="text_length", ax=ax)
            ax.set_xlabel("Category")
        else:
            sns.boxplot(data=df_copy, y="text_length", ax=ax)

        ax.set_ylabel("Text Length (chars)")
        ax.set_title("Document Text Length Distribution")
        plt.tight_layout()

        session.pending_figure = fig

        result_id = session.next_result_id("describe_data")
        summary = {
            "result_id": result_id,
            "n_documents": len(df_copy),
            "stats_overall": overall_stats,
            "stats_by_category": stats_by_category,
        }

        session.store_result("describe_data_tool", {}, summary)
        return summary

    return [
        tokenize_tool,
        list_files_tool,
        inspect_data_tool,
        check_missing_tool,
        check_duplicates_tool,
        sample_data_tool,
        describe_data_tool,
    ]
