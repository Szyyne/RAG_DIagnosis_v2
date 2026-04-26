"""Task registry — three RAG pipeline failure scenarios."""
from __future__ import annotations

TASKS: dict = {
    "chunking_error_001": {
        "difficulty": "easy",
        "observable_state": {
            "retrieval_precision_at_3": 0.31,
            "avg_chunk_tokens": 4096,
            "context_window_utilisation": 0.98,
            "faithfulness_score": 0.44,
            "answer_relevance": 0.61,
            "logs": [
                "WARNING: chunk_size=4096 exceeds recommended max of 512",
                "WARNING: context overflow — truncating at 4096 tokens",
                "INFO: top_k=3 retrieved but only partial content used",
            ],
        },
        "root_cause": "chunk_size_too_large",
        "root_cause_explanation": (
            "chunk_size=4096 forces retrieval to return monolithic blocks. "
            "The relevant span is diluted inside a 4096-token blob, crashing "
            "precision. Faithfulness collapses because the LLM context is dominated "
            "by irrelevant surrounding text."
        ),
        "correct_patch": {"chunk_size": 512, "chunk_overlap": 64},
        "patch_alternatives": [
            {"top_k": 10},
            {"embedding_model": "text-embedding-3-large"},
        ],
        "expected_post_patch": {
            "retrieval_precision_at_3": 0.81,
            "faithfulness_score": 0.87,
        },
    },

    "embedding_mismatch_001": {
        "difficulty": "medium",
        "observable_state": {
            "retrieval_precision_at_3": 0.18,
            "cosine_similarity_avg": 0.21,
            "index_embedding_model": "text-embedding-ada-002",
            "query_embedding_model": "e5-large-v2",
            "index_dim": 1536,
            "query_dim": 1024,
            "faithfulness_score": 0.29,
            "answer_relevance": 0.34,
            "logs": [
                "ERROR: dimension mismatch — index=1536 vs query=1024",
                "WARNING: cosine similarities anomalously low (avg=0.21)",
                "INFO: fallback brute-force L2 search engaged",
            ],
        },
        "root_cause": "embedding_model_mismatch",
        "root_cause_explanation": (
            "The vector index was built with text-embedding-ada-002 (dim=1536) but "
            "the live query encoder is e5-large-v2 (dim=1024). The vectors live in "
            "different spaces; cosine similarity is meaningless and retrieval degrades "
            "to random selection."
        ),
        "correct_patch": {"query_encoder": "text-embedding-ada-002"},
        "patch_alternatives": [
            {"chunk_size": 256},
            {"reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2"},
        ],
        "expected_post_patch": {
            "retrieval_precision_at_3": 0.76,
            "faithfulness_score": 0.79,
        },
    },

    "hallucination_retrieval_001": {
        "difficulty": "hard",
        "observable_state": {
            "retrieval_precision_at_3": 0.52,
            "faithfulness_score": 0.11,
            "answer_relevance": 0.74,
            "query_expansion_enabled": True,
            "query_expansion_model": "gpt-3.5-turbo",
            "avg_expanded_query_tokens": 47,
            "reranker_enabled": False,
            "semantic_drift_score": 0.68,
            "logs": [
                "INFO: query expanded from 8 → 47 tokens",
                "WARNING: semantic drift detected (score=0.68)",
                "INFO: no reranker configured",
                "WARNING: retrieved doc mismatches expected source on 6/10 queries",
            ],
            "sample_failures": [
                {"query": "What is the refund policy?",
                 "retrieved": "cancellation_policy.txt",
                 "expected": "refund_policy.txt"},
            ],
        },
        "root_cause": "query_expansion_semantic_drift_no_reranker",
        "root_cause_explanation": (
            "Aggressive query expansion balloons the query into adjacent semantic "
            "territory. Without a cross-encoder reranker to re-score candidates, "
            "semantically-drifted documents rank first. The LLM then generates "
            "plausible-but-factually-wrong answers from those documents."
        ),
        "correct_patch": {
            "query_expansion": False,
            "reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "reranker_top_k": 3,
        },
        "patch_alternatives": [
            {"chunk_size": 128},
            {"embedding_model": "text-embedding-3-large"},
            {"query_expansion": True, "expansion_model": "gpt-4"},
        ],
        "expected_post_patch": {
            "retrieval_precision_at_3": 0.83,
            "faithfulness_score": 0.81,
        },
    },
}

DIFFICULTY_MULTIPLIER = {"easy": 1.0, "medium": 1.3, "hard": 1.6}
