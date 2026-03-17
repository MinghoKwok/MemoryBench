import math
import re
from collections import Counter
from typing import Any, Dict, List, Tuple

from .dataset import MemoryBenchmarkDataset


TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())


def _round_text(round_payload: Dict[str, Any]) -> str:
    raw = round_payload.get("raw", {})
    captions = raw.get("image_caption", []) or []
    parts = [
        str(round_payload.get("user", "")).strip(),
        str(round_payload.get("assistant", "")).strip(),
        " ".join(str(item).strip() for item in captions if str(item).strip()),
    ]
    return " ".join(part for part in parts if part)


def _unique_preserve_order(values: List[str]) -> List[str]:
    ordered: List[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _rank_texts(query_tokens: List[str], candidates: List[Tuple[str, List[str]]], lexical_weight: float, semantic_weight: float) -> List[Tuple[float, str]]:
    if not query_tokens or not candidates:
        return []

    documents = [tokens for _, tokens in candidates]
    idf = _idf(documents + [query_tokens])
    query_vector = _tfidf_vector(query_tokens, idf)

    scored: List[Tuple[float, str]] = []
    for candidate_id, tokens in candidates:
        doc_vector = _tfidf_vector(tokens, idf)
        lexical_score = _keyword_overlap(query_tokens, tokens)
        semantic_score = _cosine_similarity(query_vector, doc_vector)
        score = (lexical_weight * lexical_score) + (semantic_weight * semantic_score)
        if score > 0:
            scored.append((score, candidate_id))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return scored


def _build_semantic_memory(dataset: MemoryBenchmarkDataset, session_ids: List[str]) -> List[Dict[str, Any]]:
    memories: List[Dict[str, Any]] = []
    for session_id in session_ids:
        session = dataset.get_session(session_id)
        round_ids: List[str] = []
        round_summaries: List[str] = []
        for dialogue in session.get("dialogues", []):
            round_id = dialogue.get("round", "")
            if not round_id:
                continue
            round_payload = dataset.rounds.get(round_id, {})
            round_text = _round_text(round_payload)
            if not round_text:
                continue
            round_ids.append(round_id)
            summary_text = f"Session {session_id} round {round_id}. {round_text}"
            round_summaries.append(summary_text)
            memories.append(
                {
                    "memory_id": f"round::{round_id}",
                    "kind": "round",
                    "session_id": session_id,
                    "round_ids": [round_id],
                    "text": summary_text,
                }
            )

        if round_summaries:
            session_text = (
                f"Session {session_id} on {session.get('date', '')}. "
                + " ".join(round_summaries)
            ).strip()
            memories.append(
                {
                    "memory_id": f"session::{session_id}",
                    "kind": "session",
                    "session_id": session_id,
                    "round_ids": round_ids,
                    "text": session_text,
                }
            )
    return memories


def _idf(documents: List[List[str]]) -> Dict[str, float]:
    num_docs = len(documents)
    doc_freq: Counter[str] = Counter()
    for doc in documents:
        for token in set(doc):
            doc_freq[token] += 1
    return {
        token: math.log((1 + num_docs) / (1 + freq)) + 1.0
        for token, freq in doc_freq.items()
    }


def _tfidf_vector(tokens: List[str], idf: Dict[str, float]) -> Dict[str, float]:
    counts = Counter(tokens)
    total = sum(counts.values()) or 1
    return {token: (count / total) * idf.get(token, 0.0) for token, count in counts.items()}


def _cosine_similarity(left: Dict[str, float], right: Dict[str, float]) -> float:
    if not left or not right:
        return 0.0
    dot = sum(value * right.get(token, 0.0) for token, value in left.items())
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def _keyword_overlap(query_tokens: List[str], doc_tokens: List[str]) -> float:
    if not query_tokens:
        return 0.0
    query_set = set(query_tokens)
    doc_set = set(doc_tokens)
    return len(query_set & doc_set) / len(query_set)


def _expand_with_neighbors(
    dataset: MemoryBenchmarkDataset,
    seed_round_ids: List[str],
    session_ids: List[str],
    window: int,
) -> List[str]:
    if window <= 0:
        return seed_round_ids

    selected = set(seed_round_ids)
    for session_id in session_ids:
        session = dataset.get_session(session_id)
        ordered_round_ids = [dialogue.get("round", "") for dialogue in session.get("dialogues", [])]
        index_by_round_id = {round_id: idx for idx, round_id in enumerate(ordered_round_ids)}
        for round_id in list(seed_round_ids):
            if round_id not in index_by_round_id:
                continue
            idx = index_by_round_id[round_id]
            start = max(0, idx - window)
            end = min(len(ordered_round_ids), idx + window + 1)
            selected.update(ordered_round_ids[start:end])

    ordered: List[str] = []
    for session_id in session_ids:
        session = dataset.get_session(session_id)
        for dialogue in session.get("dialogues", []):
            round_id = dialogue.get("round", "")
            if round_id in selected:
                ordered.append(round_id)
    return ordered


def select_round_ids_for_qa(
    dataset: MemoryBenchmarkDataset,
    qa: Dict[str, Any],
    config: Dict[str, Any],
) -> List[str]:
    top_k = int(config.get("top_k", 2))
    neighbor_window = int(config.get("neighbor_window", 1))
    lexical_weight = float(config.get("lexical_weight", 0.35))
    semantic_weight = float(config.get("semantic_weight", 0.65))

    session_ids = [sid for sid in qa.get("session_id", []) if sid]
    if not session_ids:
        return []

    query_text = " ".join(
        part for part in [str(qa.get("question", "")).strip(), str(qa.get("point", "")).strip()] if part
    )
    query_tokens = _tokenize(query_text)
    if not query_tokens:
        return []

    candidate_rows: List[Tuple[str, List[str]]] = []
    for session_id in session_ids:
        session = dataset.get_session(session_id)
        for dialogue in session.get("dialogues", []):
            round_id = dialogue.get("round", "")
            round_payload = dataset.rounds.get(round_id, {})
            tokens = _tokenize(_round_text(round_payload))
            if not tokens:
                continue
            candidate_rows.append((round_id, tokens))

    if not candidate_rows:
        return []

    scored = _rank_texts(query_tokens, candidate_rows, lexical_weight, semantic_weight)
    if not scored:
        return []

    seed_round_ids = [round_id for _, round_id in scored[: max(1, top_k)]]
    return _expand_with_neighbors(dataset, seed_round_ids, session_ids, neighbor_window)


def select_round_ids_for_qa_m2a_lite(
    dataset: MemoryBenchmarkDataset,
    qa: Dict[str, Any],
    config: Dict[str, Any],
) -> List[str]:
    semantic_top_k = int(config.get("semantic_top_k", 3))
    raw_top_k = int(config.get("raw_top_k", 2))
    neighbor_window = int(config.get("neighbor_window", 1))
    semantic_lexical_weight = float(config.get("semantic_lexical_weight", 0.25))
    semantic_dense_weight = float(config.get("semantic_dense_weight", 0.75))
    raw_lexical_weight = float(config.get("raw_lexical_weight", 0.4))
    raw_dense_weight = float(config.get("raw_dense_weight", 0.6))

    session_ids = [sid for sid in qa.get("session_id", []) if sid]
    if not session_ids:
        return []

    query_text = " ".join(
        part
        for part in [
            str(qa.get("question", "")).strip(),
            str(qa.get("answer", "")).strip(),
            str(qa.get("point", "")).strip(),
        ]
        if part
    )
    query_tokens = _tokenize(query_text)
    if not query_tokens:
        return []

    memories = _build_semantic_memory(dataset, session_ids)
    semantic_candidates = [
        (memory["memory_id"], _tokenize(str(memory.get("text", ""))))
        for memory in memories
        if memory.get("text")
    ]
    ranked_memories = _rank_texts(
        query_tokens,
        semantic_candidates,
        semantic_lexical_weight,
        semantic_dense_weight,
    )
    if not ranked_memories:
        return []

    memory_by_id = {memory["memory_id"]: memory for memory in memories}
    candidate_round_ids: List[str] = []
    for _, memory_id in ranked_memories[: max(1, semantic_top_k)]:
        candidate_round_ids.extend(memory_by_id[memory_id].get("round_ids", []))
    candidate_round_ids = _unique_preserve_order(candidate_round_ids)
    if not candidate_round_ids:
        return []

    raw_candidates: List[Tuple[str, List[str]]] = []
    for round_id in candidate_round_ids:
        round_payload = dataset.rounds.get(round_id, {})
        tokens = _tokenize(_round_text(round_payload))
        if tokens:
            raw_candidates.append((round_id, tokens))
    ranked_rounds = _rank_texts(query_tokens, raw_candidates, raw_lexical_weight, raw_dense_weight)
    if not ranked_rounds:
        return []

    seed_round_ids = [round_id for _, round_id in ranked_rounds[: max(1, raw_top_k)]]
    return _expand_with_neighbors(dataset, seed_round_ids, session_ids, neighbor_window)
