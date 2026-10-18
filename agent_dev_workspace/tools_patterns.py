import tempfile
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Literal, Optional
from langchain_core.tools import tool
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

import PAMI.extras.convert.DF2DB as db_converter
import PAMI.frequentPattern.basic.FPGrowth as fpgrowth
import PAMI.frequentPattern.topk.FAE as topk_fae
import PAMI.frequentPattern.maximal.MaxFPGrowth as max_fpgrowth


def make_tools(session):
    @tool
    def mine_patterns_tool(
        category_name: str,
        filtering_method: Literal["variance", "tfidf", "term_frequency"],
        algorithm: Literal["fpgrowth", "topk", "maxfpgrowth"] = "fpgrowth",
        min_sup: Optional[int] = None,
        k: Optional[int] = None
    ) -> Dict[str, Any]:
        """Mines frequent item patterns for a specific category using PAMI.

        First filters category documents, builds a fresh category-local DTM with English
        stopwords removed, applies vocabulary filtering (variance, tfidf, or term_frequency),
        converts surviving terms into a transactional database, and runs the requested PAMI algorithm.

        Args:
            category_name: Category to filter session.dataframe by.
            filtering_method: Vocabulary filtering method ('variance', 'tfidf', 'term_frequency').
            algorithm: Pattern mining algorithm ('fpgrowth', 'topk', 'maxfpgrowth').
            min_sup: Minimum support threshold (required for 'fpgrowth' and 'maxfpgrowth').
            k: Top-k patterns to keep (required for 'topk').

        Returns:
            Dict containing result_id, category_name, filtering_method, algorithm, and mined patterns.
        """
        if session.dataframe is None:
            return {"error": "No dataframe loaded in session."}

        df_cat = session.dataframe[session.dataframe["category_name"] == category_name]
        if df_cat.empty:
            return {"error": f"No documents found for category '{category_name}'."}

        # Step 1: Build category-local DTM with English stop words removed
        vectorizer = CountVectorizer(stop_words="english")
        dtm = vectorizer.fit_transform(df_cat["text"])
        feature_names = np.array(vectorizer.get_feature_names_out())

        if len(feature_names) == 0:
            result_id = session.next_result_id("patterns")
            summary = {
                "result_id": result_id,
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "patterns": [],
                "note": "no terms survived stopword removal"
            }
            session.store_result("mine_patterns_tool", {"category_name": category_name, "filtering_method": filtering_method, "algorithm": algorithm}, summary)
            return summary

        # Step 2: Apply vocabulary filtering
        dtm_dense = dtm.toarray()
        if filtering_method in ("variance", "term_frequency"):
            if filtering_method == "variance":
                scores = np.var(dtm_dense, axis=0)
            else:
                scores = np.sum(dtm_dense, axis=0)

            p5, p95 = np.percentile(scores, [5, 95])
            mask = (scores > p5) & (scores < p95)
        elif filtering_method == "tfidf":
            tfidf = TfidfTransformer().fit_transform(dtm)
            scores = np.mean(tfidf.toarray(), axis=0)
            p20 = np.percentile(scores, 20)
            mask = scores >= p20
        else:
            return {"error": f"Invalid filtering method: {filtering_method}"}

        surviving_features = feature_names[mask]
        surviving_dtm = dtm_dense[:, mask]

        # Step 3: Handle zero surviving terms
        if len(surviving_features) == 0 or surviving_dtm.shape[1] == 0:
            result_id = session.next_result_id("patterns")
            summary = {
                "result_id": result_id,
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "patterns": [],
                "note": "no terms survived filtering"
            }
            session.store_result("mine_patterns_tool", {"category_name": category_name, "filtering_method": filtering_method, "algorithm": algorithm}, summary)
            return summary

        # Step 4: Convert surviving terms to transactional database DataFrame (binary presence)
        trans_df = pd.DataFrame(
            (surviving_dtm >= 1).astype(int),
            columns=surviving_features
        )

        with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".txt") as tmp_db:
            db_path = tmp_db.name

        try:
            converter = db_converter.DF2DB(trans_df, 1)
            converter.convert2TransactionalDatabase(db_path)

            # Validate parameters and instantiate algorithm
            if algorithm in ("fpgrowth", "maxfpgrowth"):
                if min_sup is None:
                    return {"error": f"min_sup is required for algorithm '{algorithm}'."}
                if algorithm == "fpgrowth":
                    miner = fpgrowth.FPGrowth(iFile=db_path, minSup=min_sup)
                else:
                    miner = max_fpgrowth.MaxFPGrowth(iFile=db_path, minSup=min_sup)
            elif algorithm == "topk":
                if k is None:
                    return {"error": "k parameter is required for algorithm 'topk'."}
                miner = topk_fae.FAE(iFile=db_path, k=k)
            else:
                return {"error": f"Invalid algorithm: {algorithm}"}

            miner.mine()
            patterns_df = miner.getPatternsAsDataFrame()

            patterns_list = []
            if patterns_df is not None and not patterns_df.empty:
                for _, row in patterns_df.iterrows():
                    pat_str = str(row["Patterns"])
                    supp_val = float(row["Support"])
                    items = pat_str.strip().split("\t")
                    patterns_list.append({"pattern": items, "support": supp_val})

            result_id = session.next_result_id("patterns")
            artifact_key = f"patterns_{category_name}_{filtering_method}_{algorithm}"
            session.artifacts[artifact_key] = patterns_list

            summary = {
                "result_id": result_id,
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "n_patterns": len(patterns_list),
                "patterns": patterns_list
            }

            session.store_result("mine_patterns_tool", {
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "min_sup": min_sup,
                "k": k
            }, summary)

            return summary

        finally:
            import os
            if os.path.exists(db_path):
                os.remove(db_path)

    return [mine_patterns_tool]
