import math
import re
from collections import Counter
from typing import Any, Dict, List, Tuple

from .dataset import MemoryBenchmarkDataset


TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "but",
    "by",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "his",
    "i",
    "in",
    "into",
    "is",
    "it",
    "its",
    "me",
    "my",
    "of",
    "on",
    "or",
    "our",
    "she",
    "that",
    "the",
    "their",
    "them",
    "they",
    "this",
    "to",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "you",
    "your",
}


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


def _rrf_fuse(rankings: List[List[str]], k: int = 60) -> List[str]:
    scores: Dict[str, float] = {}
    for ranked_ids in rankings:
        for rank, item_id in enumerate(ranked_ids, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + (1.0 / (k + rank))
    return [item_id for item_id, _ in sorted(scores.items(), key=lambda x: (-x[1], x[0]))]


def _extract_expansion_tokens(tokens: List[str], top_n: int = 6) -> List[str]:
    counts = Counter(t for t in tokens if t not in STOPWORDS and len(t) > 2)
    return [token for token, _ in counts.most_common(top_n)]


def _round_index_maps(dataset: MemoryBenchmarkDataset, session_ids: List[str]) -> Dict[str, int]:
    order: Dict[str, int] = {}
    idx = 0
    for session_id in session_ids:
        session = dataset.get_session(session_id)
        for dialogue in session.get("dialogues", []):
            round_id = dialogue.get("round", "")
            if not round_id:
                continue
            order[round_id] = idx
            idx += 1
    return order


def _build_m2a_full_memories(
    dataset: MemoryBenchmarkDataset,
    session_ids: List[str],
    min_round_words_for_memory: int,
) -> List[Dict[str, Any]]:
    memories: List[Dict[str, Any]] = []
    for session_id in session_ids:
        session = dataset.get_session(session_id)
        session_date = str(session.get("date", "")).strip()
        session_round_ids: List[str] = []
        session_snippets: List[str] = []

        for dialogue in session.get("dialogues", []):
            round_id = dialogue.get("round", "")
            if not round_id:
                continue
            round_payload = dataset.rounds.get(round_id, {})
            raw = round_payload.get("raw", {})
            user = str(round_payload.get("user", "")).strip()
            assistant = str(round_payload.get("assistant", "")).strip()
            captions = " ".join(str(c).strip() for c in (raw.get("image_caption", []) or []) if str(c).strip())
            round_text = " ".join(
                p for p in [f"On {session_date}.", f"Session {session_id}.", user, assistant, captions] if p
            ).strip()
            if len(_tokenize(round_text)) < min_round_words_for_memory:
                continue

            session_round_ids.append(round_id)
            session_snippets.append(round_text[:280])

            memories.append(
                {
                    "memory_id": f"turn::{round_id}::user",
                    "kind": "turn",
                    "session_id": session_id,
                    "round_ids": [round_id],
                    "text": f"Session {session_id} round {round_id} user: {user} {captions}".strip(),
                }
            )
            memories.append(
                {
                    "memory_id": f"turn::{round_id}::assistant",
                    "kind": "turn",
                    "session_id": session_id,
                    "round_ids": [round_id],
                    "text": f"Session {session_id} round {round_id} assistant: {assistant} {captions}".strip(),
                }
            )
            memories.append(
                {
                    "memory_id": f"round::{round_id}",
                    "kind": "round",
                    "session_id": session_id,
                    "round_ids": [round_id],
                    "text": f"Session {session_id} round {round_id}. {round_text}",
                }
            )

        if session_round_ids and session_snippets:
            session_summary = (
                f"Session {session_id} on {session_date}. "
                + " ".join(session_snippets[: min(10, len(session_snippets))])
            ).strip()
            memories.append(
                {
                    "memory_id": f"session::{session_id}",
                    "kind": "session",
                    "session_id": session_id,
                    "round_ids": session_round_ids,
                    "text": session_summary,
                }
            )

    return memories


def select_round_ids_for_qa_m2a_full(
    dataset: MemoryBenchmarkDataset,
    qa: Dict[str, Any],
    config: Dict[str, Any],
) -> List[str]:
    semantic_top_k = int(config.get("semantic_top_k", 8))
    raw_top_k = int(config.get("raw_top_k", 6))
    neighbor_window = int(config.get("neighbor_window", 1))
    max_iterations = int(config.get("max_iterations", 2))
    rrf_k = int(config.get("rrf_k", 60))
    min_round_words_for_memory = int(config.get("min_round_words_for_memory", 3))
    semantic_lexical_weight = float(config.get("semantic_lexical_weight", 0.35))
    semantic_dense_weight = float(config.get("semantic_dense_weight", 0.65))
    raw_lexical_weight = float(config.get("raw_lexical_weight", 0.35))
    raw_dense_weight = float(config.get("raw_dense_weight", 0.65))

    session_ids = [sid for sid in qa.get("session_id", []) if sid]
    if not session_ids:
        return []

    query_text = " ".join(
        part
        for part in [str(qa.get("question", "")).strip(), str(qa.get("point", "")).strip()]
        if part
    )
    query_tokens = _tokenize(query_text)
    if not query_tokens:
        return []

    round_order = _round_index_maps(dataset, session_ids)
    memories = _build_m2a_full_memories(dataset, session_ids, min_round_words_for_memory=min_round_words_for_memory)
    if not memories:
        return []

    memory_by_id = {m["memory_id"]: m for m in memories}
    semantic_candidates: List[Tuple[str, List[str]]] = []
    for memory in memories:
        tokens = _tokenize(str(memory.get("text", "")))
        if tokens:
            semantic_candidates.append((memory["memory_id"], tokens))
    if not semantic_candidates:
        return []

    selected_round_ids: List[str] = []
    query_state_tokens = list(query_tokens)

    for _ in range(max(1, max_iterations)):
        semantic_dense_rank = [
            mid
            for _, mid in _rank_texts(
                query_state_tokens,
                semantic_candidates,
                lexical_weight=0.0,
                semantic_weight=1.0,
            )
        ]
        semantic_lexical_rank = [
            mid
            for _, mid in _rank_texts(
                query_state_tokens,
                semantic_candidates,
                lexical_weight=1.0,
                semantic_weight=0.0,
            )
        ]
        semantic_mixed_rank = [
            mid
            for _, mid in _rank_texts(
                query_state_tokens,
                semantic_candidates,
                lexical_weight=semantic_lexical_weight,
                semantic_weight=semantic_dense_weight,
            )
        ]

        ranked_memory_ids = _rrf_fuse(
            [semantic_dense_rank, semantic_lexical_rank, semantic_mixed_rank],
            k=rrf_k,
        )
        if not ranked_memory_ids:
            break

        candidate_round_ids: List[str] = []
        for memory_id in ranked_memory_ids[: max(1, semantic_top_k)]:
            candidate_round_ids.extend(memory_by_id[memory_id].get("round_ids", []))
        candidate_round_ids = _unique_preserve_order(candidate_round_ids)
        if not candidate_round_ids:
            break

        raw_candidates: List[Tuple[str, List[str]]] = []
        for round_id in candidate_round_ids:
            round_payload = dataset.rounds.get(round_id, {})
            tokens = _tokenize(_round_text(round_payload))
            if tokens:
                raw_candidates.append((round_id, tokens))
        if not raw_candidates:
            break

        raw_dense_rank = [
            rid
            for _, rid in _rank_texts(
                query_state_tokens,
                raw_candidates,
                lexical_weight=0.0,
                semantic_weight=1.0,
            )
        ]
        raw_lexical_rank = [
            rid
            for _, rid in _rank_texts(
                query_state_tokens,
                raw_candidates,
                lexical_weight=1.0,
                semantic_weight=0.0,
            )
        ]
        raw_mixed_rank = [
            rid
            for _, rid in _rank_texts(
                query_state_tokens,
                raw_candidates,
                lexical_weight=raw_lexical_weight,
                semantic_weight=raw_dense_weight,
            )
        ]
        ranked_round_ids = _rrf_fuse([raw_dense_rank, raw_lexical_rank, raw_mixed_rank], k=rrf_k)
        if not ranked_round_ids:
            break

        picked_round_ids = ranked_round_ids[: max(1, raw_top_k)]
        selected_round_ids.extend(picked_round_ids)
        selected_round_ids = _unique_preserve_order(selected_round_ids)

        expansion_pool: List[str] = []
        for round_id in picked_round_ids:
            expansion_pool.extend(_tokenize(_round_text(dataset.rounds.get(round_id, {}))))
        query_state_tokens.extend(_extract_expansion_tokens(expansion_pool))
        query_state_tokens = _unique_preserve_order(query_state_tokens)

    if not selected_round_ids:
        return []

    selected_round_ids = sorted(
        selected_round_ids,
        key=lambda rid: round_order.get(rid, 10**9),
    )
    return _expand_with_neighbors(dataset, selected_round_ids, session_ids, neighbor_window)
