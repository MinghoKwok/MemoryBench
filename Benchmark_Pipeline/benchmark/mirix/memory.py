"""
MIRIX memory module for the MemEye benchmark.

Implements MIRIX-style memory retrieval and prompt construction.
Reference: github.com/Mirix-AI/MIRIX, branch public_evaluation.

Key design choices:
- Retrieval is embedding-based (not BM25) — ``search_method = 'embedding'``
- Episodic memory has dual retrieval: "Most Recent" + "Most Relevant"
- Each memory type is searched independently, not fused via RRF
- Episodic entries store separate summary and details fields
- Retrieved memories are assembled into XML-tagged system prompt sections

The active MIRIX path uses incremental agentic ingestion:
- meta-memory routing
- type-specific memory agents
- text-only retrieval/prompting at QA time
"""

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ...router.http_utils import encode_image_data_url, post_json, require_api_key
from ..dataset import MemoryBenchmarkDataset
from ..embeddings import ImageEmbedder, TextEmbedder
from ..retrieval import _dense_embedding_rank, _image_embedding_rank

logger = logging.getLogger(__name__)
MIRIX_CACHE_SCHEMA_VERSION = 6


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class MIRIXMemoryType(Enum):
    CORE = "core"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    RESOURCE = "resource"
    KNOWLEDGE_VAULT = "knowledge_vault"


@dataclass
class MIRIXMemoryEntry:
    """A single MIRIX memory entry.

    Aligned with official MIRIX schema: episodic entries use separate
    ``summary`` and ``details`` fields (searched on ``details``).
    The ``content`` field is the combined readable form.
    """
    memory_id: str
    memory_type: str  # MIRIXMemoryType.value
    content: str      # full readable text (for display / fallback search)
    summary: str      # short summary (episodic: event headline)
    details: str      # detailed description (episodic: searchable details field)
    source_round_ids: List[str] = field(default_factory=list)
    source_session_id: str = ""
    image_paths: List[str] = field(default_factory=list)
    timestamp: str = ""
    order_index: int = 0  # insertion order — used for "most recent" retrieval
    occurred_at: str = ""  # ISO date when event happened (for faithful recency)
    caption: str = ""      # for knowledge_vault entries (official uses this field)
    name: str = ""         # for semantic memories
    title: str = ""        # for resource memories
    resource_type: str = ""
    entry_type: str = ""   # for procedural memories
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RoundEvidence:
    session_id: str
    round_id: str
    date: str
    user_text: str
    assistant_text: str
    image_paths: List[str] = field(default_factory=list)
    image_captions: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Memory store
# ---------------------------------------------------------------------------

class MIRIXMemoryStore:
    """Holds all extracted MIRIX memories for a single scenario."""

    def __init__(
        self,
        memories: List[MIRIXMemoryEntry],
        extraction_mode: str = "heuristic",
        extraction_model: str = "",
        schema_version: int = MIRIX_CACHE_SCHEMA_VERSION,
    ) -> None:
        self.memories = memories
        self.extraction_mode = extraction_mode
        self.extraction_model = extraction_model
        self.schema_version = schema_version

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "extraction_mode": self.extraction_mode,
            "extraction_model": self.extraction_model,
            "schema_version": self.schema_version,
            "memories": [asdict(m) for m in self.memories],
        }
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        logger.info("Saved %d MIRIX memories to %s", len(self.memories), path)

    @classmethod
    def load(cls, path: Path) -> "MIRIXMemoryStore":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        memories = []
        for m in data.get("memories", []):
            # Backward compat: older caches may lack some fields
            m.setdefault("summary", m.get("content", ""))
            m.setdefault("details", m.get("content", ""))
            m.setdefault("order_index", 0)
            m.setdefault("occurred_at", m.get("timestamp", ""))
            m.setdefault("caption", m.get("summary", ""))
            m.setdefault("name", "")
            m.setdefault("title", "")
            m.setdefault("resource_type", "")
            m.setdefault("entry_type", "")
            memories.append(MIRIXMemoryEntry(**m))
        return cls(
            memories=memories,
            extraction_mode=data.get("extraction_mode", "unknown"),
            extraction_model=data.get("extraction_model", ""),
            schema_version=int(data.get("schema_version", 1)),
        )

    def all_entries(self) -> List[MIRIXMemoryEntry]:
        return list(self.memories)

    def get_by_type(self, memory_type: MIRIXMemoryType) -> List[MIRIXMemoryEntry]:
        return [m for m in self.memories if m.memory_type == memory_type.value]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _extract_core_from_profile(
    profile: Dict[str, Any], memory_id: str,
) -> Optional[MIRIXMemoryEntry]:
    """Build a Core memory entry from the character profile.

    Supports Mem-Gallery format (name/persona_summary/traits/conversation_style)
    and MemEye format (primary_user/role/setting/case_goal).
    """
    parts: List[str] = []
    # Mem-Gallery format
    if profile.get("name"):
        parts.append(f"Name: {profile['name']}")
    if profile.get("persona_summary"):
        parts.append(f"Summary: {profile['persona_summary']}")
    if profile.get("traits"):
        traits = profile["traits"]
        parts.append(f"Traits: {', '.join(traits) if isinstance(traits, list) else traits}")
    if profile.get("conversation_style"):
        parts.append(f"Style: {profile['conversation_style']}")
    # MemEye format
    if profile.get("primary_user"):
        parts.append(f"Name: {profile['primary_user']}")
    if profile.get("role"):
        parts.append(f"Role: {profile['role']}")
    if profile.get("setting"):
        parts.append(f"Setting: {profile['setting']}")
    if profile.get("case_goal"):
        parts.append(f"Goal: {profile['case_goal']}")
    if not parts:
        return None
    content = " | ".join(parts)
    return MIRIXMemoryEntry(
        memory_id=memory_id,
        memory_type=MIRIXMemoryType.CORE.value,
        content=content,
        summary=content,
        details=content,
        source_round_ids=[],
        source_session_id="",
        metadata={"source": "character_profile"},
    )


def _clean_text(text: Any) -> str:
    return str(text or "").strip()


def _summarize_sentence(text: str, limit: int = 160) -> str:
    text = " ".join(_clean_text(text).split())
    if len(text) <= limit:
        return text
    clipped = text[: limit - 3].rsplit(" ", 1)[0].strip()
    return (clipped or text[: limit - 3]).strip() + "..."


def _extract_json_captions(dialogue: Dict[str, Any]) -> List[str]:
    return [
        _clean_text(caption)
        for caption in (dialogue.get("image_caption", []) or [])
        if _clean_text(caption)
    ]


def build_round_evidence(
    dataset: MemoryBenchmarkDataset,
    llm_config: Optional[Dict[str, Any]] = None,
) -> List[RoundEvidence]:
    evidences: List[RoundEvidence] = []
    for session in dataset.data.get("multi_session_dialogues", []):
        sid = session.get("session_id", "")
        date = _clean_text(session.get("date", ""))
        for dialogue in session.get("dialogues", []):
            rid = dialogue.get("round", "")
            round_payload = dataset.rounds.get(rid, {})
            user_text = _clean_text(round_payload.get("user", ""))
            assistant_text = _clean_text(round_payload.get("assistant", ""))
            image_paths = list(round_payload.get("images", []))
            if not image_paths:
                image_captions = []
            elif llm_config and llm_config.get("direct_image_ingestion", True):
                image_captions = (
                    _extract_json_captions(dialogue)
                    if llm_config.get("use_dataset_image_captions", True)
                    else []
                )
            else:
                image_captions = _extract_json_captions(dialogue)
            evidences.append(
                RoundEvidence(
                    session_id=sid,
                    round_id=rid,
                    date=date,
                    user_text=user_text,
                    assistant_text=assistant_text,
                    image_paths=image_paths,
                    image_captions=image_captions,
                )
            )
    return evidences


def iter_incremental_windows(
    evidences: List[RoundEvidence],
    history_size: int = 2,
) -> List[List[RoundEvidence]]:
    windows: List[List[RoundEvidence]] = []
    session_history: Dict[str, List[RoundEvidence]] = {}
    for evidence in evidences:
        history = session_history.setdefault(evidence.session_id, [])
        window = history[-history_size:] + [evidence]
        windows.append(window)
        history.append(evidence)
    return windows


def _coerce_json_response(raw: str) -> Any:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned)
    return json.loads(cleaned)


def _normalize_memory_type(value: str) -> str:
    normalized = _clean_text(value).lower().replace(" ", "_")
    allowed = {t.value for t in MIRIXMemoryType}
    return normalized if normalized in allowed else MIRIXMemoryType.SEMANTIC.value


def _normalize_update_action(value: str) -> str:
    normalized = _clean_text(value).lower()
    return normalized if normalized in {"insert", "merge", "replace", "skip"} else "insert"


def _normalize_update_key(memory_type: str, update_key: str, fallback_summary: str) -> str:
    key = _clean_text(update_key)
    if key:
        return key
    summary = re.sub(r"[^a-z0-9]+", "_", fallback_summary.lower()).strip("_")
    summary = summary[:80] or "memory"
    return f"{memory_type}:{summary}"


def _normalize_confidence(value: Any) -> float:
    if isinstance(value, (int, float)):
        return max(0.0, min(1.0, float(value)))
    lowered = _clean_text(value).lower()
    if not lowered:
        return 0.5
    mapping = {
        "very high": 0.95,
        "high": 0.85,
        "medium": 0.6,
        "moderate": 0.6,
        "low": 0.35,
        "very low": 0.15,
    }
    if lowered in mapping:
        return mapping[lowered]
    try:
        return max(0.0, min(1.0, float(lowered)))
    except ValueError:
        return 0.5


def _build_memory_state_summary(
    store: MIRIXMemoryStore,
    max_items: int = 12,
) -> Dict[str, List[Dict[str, Any]]]:
    memories = sorted(store.memories, key=lambda m: m.order_index, reverse=True)
    summary: Dict[str, List[Dict[str, Any]]] = {
        "core": [],
        "episodic": [],
        "semantic": [],
        "procedural": [],
        "resource": [],
        "knowledge_vault": [],
    }
    per_type_limits = {
        "core": 3,
        "episodic": 4,
        "semantic": 3,
        "procedural": 1,
        "resource": 1,
        "knowledge_vault": 0,
    }
    counts = {k: 0 for k in summary}
    for memory in memories:
        key = memory.memory_type
        if key not in summary:
            continue
        if counts[key] >= per_type_limits.get(key, 0):
            continue
        summary[key].append(
            {
                "memory_id": memory.memory_id,
                "summary": memory.summary,
                "name": memory.name,
                "title": memory.title,
                "update_key": memory.metadata.get("update_key", ""),
                "occurred_at": memory.occurred_at,
            }
        )
        counts[key] += 1
        if sum(counts.values()) >= max_items:
            break
    return summary


def _build_type_memory_summary(
    store: MIRIXMemoryStore,
    memory_type: str,
    limit: int = 6,
) -> List[Dict[str, Any]]:
    selected = [
        memory for memory in sorted(store.memories, key=lambda m: m.order_index, reverse=True)
        if memory.memory_type == memory_type
    ][:limit]
    return [
        {
            "memory_id": memory.memory_id,
            "summary": memory.summary,
            "details": memory.details,
            "update_key": memory.metadata.get("update_key", ""),
            "occurred_at": memory.occurred_at,
            "confidence": memory.metadata.get("confidence", 0.5),
            "name": memory.name,
            "title": memory.title,
            "entry_type": memory.entry_type,
            "resource_type": memory.resource_type,
        }
        for memory in selected
    ]


def _window_payload(window: List[RoundEvidence]) -> List[Dict[str, Any]]:
    payload: List[Dict[str, Any]] = []
    for evidence in window:
        payload.append(
            {
                "session_id": evidence.session_id,
                "round_id": evidence.round_id,
                "date": evidence.date,
                "user_text": evidence.user_text,
                "assistant_text": evidence.assistant_text,
                "image_captions": evidence.image_captions,
                "has_image": bool(evidence.image_paths),
            }
        )
    return payload

_META_MEMORY_PROMPT = """\
You are the MIRIX Meta Memory Agent operating on a chronological replay of dialog turns.

Your job is to decide which memory systems should update because of the NEWEST round.

Rules:
- Episodic memory should usually be updated when the newest round contains a meaningful observation, reply, question, decision, state change, or image.
- Core memory is only for durable user profile/preferences/recurring constraints.
- Semantic memory is for generalized facts that should persist beyond one event.
- Procedural memory is for instructions, workflows, or how-to guidance.
- Resource memory is for links, files, documents, or named external references.
- Knowledge vault is only for clearly sensitive/private facts.
- Select at most 3 memory types.
- Use only information available in the window and current memory summary.

Return ONLY valid JSON with this shape:
{{
  "memory_types": ["episodic", "semantic"],
  "focus": {{
    "episodic": "most significant event to store",
    "semantic": "generalized fact to update"
  }}
}}

Newest round:
{newest_round}

Recent dialog window:
{window_json}

Current memory summary:
{memory_summary_json}
"""


_TYPE_AGENT_PROMPTS = {
    "episodic": """\
You are the MIRIX Episodic Memory Agent.

Update episodic memory for the newest round. Prefer the single most significant event.
When merging or replacing, target an existing memory only if it clearly refers to the same continuing event/object/option.
Images may be present and should be interpreted directly.

Return ONLY valid JSON with this shape:
{{
  "memory_type": "episodic",
  "update_action": "insert|merge|replace|skip",
  "target_memory_id": "existing id or empty",
  "target_update_key": "existing key or empty",
  "update_key": "stable canonical key",
  "summary": "self-contained event summary",
  "details": "detailed event description",
  "caption": "brief visual evidence if relevant",
  "occurred_at": "date or empty",
  "source_round_ids": ["round ids"],
  "confidence": 0.0,
  "resolved_referents": [],
  "entities": [],
  "evidence_type": ""
}}
""",
    "semantic": """\
You are the MIRIX Semantic Memory Agent.

Write or update one generalized fact that should persist beyond the current moment.
Do not restate the entire dialog turn as an event.

Return ONLY valid JSON with this shape:
{{
  "memory_type": "semantic",
  "update_action": "insert|merge|replace|skip",
  "target_memory_id": "existing id or empty",
  "target_update_key": "existing key or empty",
  "update_key": "stable canonical key",
  "name": "canonical entity/fact name",
  "summary": "generalized fact summary",
  "details": "detailed supporting fact",
  "caption": "brief visual evidence if relevant",
  "occurred_at": "date or empty",
  "source_round_ids": ["round ids"],
  "confidence": 0.0,
  "resolved_referents": [],
  "entities": [],
  "evidence_type": ""
}}
""",
    "core": """\
You are the MIRIX Core Memory Agent.

Update durable user profile, preferences, goals, or long-term constraints only.

Return ONLY valid JSON with this shape:
{{
  "memory_type": "core",
  "update_action": "insert|merge|replace|skip",
  "target_memory_id": "existing id or empty",
  "target_update_key": "existing key or empty",
  "update_key": "stable canonical key",
  "summary": "durable profile/preference summary",
  "details": "detailed durable fact",
  "occurred_at": "date or empty",
  "source_round_ids": ["round ids"],
  "confidence": 0.0,
  "resolved_referents": [],
  "entities": [],
  "evidence_type": ""
}}
""",
    "procedural": """\
You are the MIRIX Procedural Memory Agent.

Store one instruction, workflow, or reusable process only if the newest round introduces it.

Return ONLY valid JSON with this shape:
{{
  "memory_type": "procedural",
  "update_action": "insert|merge|replace|skip",
  "target_memory_id": "existing id or empty",
  "target_update_key": "existing key or empty",
  "update_key": "stable canonical key",
  "entry_type": "instruction|guide|procedure|workflow",
  "summary": "procedure summary",
  "details": "detailed procedure",
  "occurred_at": "date or empty",
  "source_round_ids": ["round ids"],
  "confidence": 0.0,
  "resolved_referents": [],
  "entities": [],
  "evidence_type": ""
}}
""",
    "resource": """\
You are the MIRIX Resource Memory Agent.

Store one resource, file, link, or external reference only if newly introduced or updated.

Return ONLY valid JSON with this shape:
{{
  "memory_type": "resource",
  "update_action": "insert|merge|replace|skip",
  "target_memory_id": "existing id or empty",
  "target_update_key": "existing key or empty",
  "update_key": "stable canonical key",
  "title": "resource title",
  "resource_type": "url|document|pdf|attachment|reference",
  "summary": "resource summary",
  "details": "resource details",
  "occurred_at": "date or empty",
  "source_round_ids": ["round ids"],
  "confidence": 0.0,
  "resolved_referents": [],
  "entities": [],
  "evidence_type": ""
}}
""",
    "knowledge_vault": """\
You are the MIRIX Knowledge Vault Agent.

Store one clearly sensitive/private fact only when it is explicitly present.

Return ONLY valid JSON with this shape:
{{
  "memory_type": "knowledge_vault",
  "update_action": "insert|merge|replace|skip",
  "target_memory_id": "existing id or empty",
  "target_update_key": "existing key or empty",
  "update_key": "stable canonical key",
  "summary": "private fact summary",
  "details": "private fact details",
  "caption": "",
  "occurred_at": "date or empty",
  "source_round_ids": ["round ids"],
  "confidence": 0.0,
  "resolved_referents": [],
  "entities": [],
  "evidence_type": ""
}}
""",
}


def _window_multimodal_content(
    prompt: str,
    window: List[RoundEvidence],
) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    for evidence in window:
        text_parts = [
            f"Session: {evidence.session_id}",
            f"Round: {evidence.round_id}",
            f"Date: {evidence.date or 'unknown'}",
            f"User: {evidence.user_text or '[empty]'}",
            f"Assistant: {evidence.assistant_text or '[empty]'}",
        ]
        if evidence.image_captions:
            text_parts.append(f"Dataset image captions: {' ; '.join(evidence.image_captions)}")
        if evidence.image_paths:
            text_parts.append("Images attached below for this round.")
        content.append({"type": "text", "text": "\n".join(text_parts)})
        for image_path in evidence.image_paths:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": encode_image_data_url(image_path)},
                }
            )
    return content


def _call_json_llm(
    content: List[Dict[str, Any]],
    llm_config: Dict[str, Any],
    model_key: str = "model",
    max_tokens_key: str = "max_tokens",
) -> Any:
    api_key = require_api_key(
        api_key=llm_config.get("api_key", ""),
        api_key_env=llm_config.get("api_key_env", "OPENAI_API_KEY"),
    )
    model = llm_config.get(model_key, llm_config.get("model", "gpt-4.1-mini"))
    base_url = llm_config.get("base_url", "https://api.openai.com/v1").rstrip("/")
    timeout = int(llm_config.get("timeout", 120))
    temperature = float(llm_config.get("temperature", 0.0))
    max_tokens = int(llm_config.get(max_tokens_key, llm_config.get("max_tokens", 1800)))
    response = post_json(
        url=f"{base_url}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        payload={
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
        timeout=timeout,
    )
    return _coerce_json_response(response["choices"][0]["message"]["content"])


def _call_meta_memory_agent(
    window: List[RoundEvidence],
    store: MIRIXMemoryStore,
    llm_config: Dict[str, Any],
) -> Dict[str, Any]:
    newest = window[-1]
    prompt = _META_MEMORY_PROMPT.format(
        newest_round=json.dumps(_window_payload([newest])[0], ensure_ascii=False, indent=2),
        window_json=json.dumps(_window_payload(window), ensure_ascii=False, indent=2),
        memory_summary_json=json.dumps(_build_memory_state_summary(store), ensure_ascii=False, indent=2),
    )
    parsed = _call_json_llm(
        _window_multimodal_content(prompt, window),
        llm_config,
        model_key="meta_model",
        max_tokens_key="meta_max_tokens",
    )
    if not isinstance(parsed, dict):
        return {"memory_types": ["episodic"], "focus": {"episodic": "most significant event in newest round"}}
    memory_types = [
        _normalize_memory_type(value)
        for value in (parsed.get("memory_types", []) or [])
        if _clean_text(value)
    ]
    if not memory_types and (newest.user_text or newest.assistant_text or newest.image_paths):
        memory_types = [MIRIXMemoryType.EPISODIC.value]
    focus = parsed.get("focus", {}) if isinstance(parsed.get("focus", {}), dict) else {}
    return {"memory_types": memory_types[:3], "focus": focus}


def _call_type_memory_agent(
    memory_type: str,
    window: List[RoundEvidence],
    store: MIRIXMemoryStore,
    llm_config: Dict[str, Any],
    focus: str = "",
) -> List[Dict[str, Any]]:
    prompt = (
        _TYPE_AGENT_PROMPTS[memory_type]
        + "\nFocus for this update:\n"
        + (focus or "Store only the most significant update caused by the newest round.")
        + "\n\nNewest round:\n"
        + json.dumps(_window_payload([window[-1]])[0], ensure_ascii=False, indent=2)
        + "\n\nRecent dialog window:\n"
        + json.dumps(_window_payload(window), ensure_ascii=False, indent=2)
        + "\n\nExisting memories of this type:\n"
        + json.dumps(_build_type_memory_summary(store, memory_type), ensure_ascii=False, indent=2)
    )
    parsed = _call_json_llm(
        _window_multimodal_content(prompt, window),
        llm_config,
        model_key="agent_model",
        max_tokens_key="agent_max_tokens",
    )
    updates = [parsed] if isinstance(parsed, dict) else parsed
    if not isinstance(updates, list):
        return []
    normalized: List[Dict[str, Any]] = []
    for update in updates:
        if not isinstance(update, dict):
            continue
        update["memory_type"] = _normalize_memory_type(update.get("memory_type", memory_type) or memory_type)
        normalized.append(update)
    return normalized[:1]


def apply_memory_updates(
    store: MIRIXMemoryStore,
    updates: List[Dict[str, Any]],
    newest: RoundEvidence,
    counters: Dict[str, int],
) -> None:
    existing_by_key: Dict[Tuple[str, str], MIRIXMemoryEntry] = {}
    existing_by_id: Dict[str, MIRIXMemoryEntry] = {}
    for memory in store.memories:
        existing_by_id[memory.memory_id] = memory
        key = _clean_text(memory.metadata.get("update_key", ""))
        if key:
            existing_by_key[(memory.memory_type, key)] = memory

    for update in updates:
        memory_type = _normalize_memory_type(update.get("memory_type", "semantic"))
        action = _normalize_update_action(update.get("update_action", "insert"))
        if action == "skip":
            continue

        summary = _clean_text(update.get("summary", "")) or _summarize_sentence(
            newest.user_text or newest.assistant_text or "Memory update",
        )
        details = _clean_text(update.get("details", "")) or summary
        update_key = _normalize_update_key(memory_type, update.get("update_key", ""), summary)
        source_round_ids = update.get("source_round_ids", []) or [newest.round_id]
        caption = _clean_text(update.get("caption", "")) or (
            " ".join(newest.image_captions) if newest.image_captions else ""
        )
        metadata = {
            "confidence": _normalize_confidence(update.get("confidence", 0.5)),
            "update_action": action,
            "resolved_referents": update.get("resolved_referents", []) or [],
            "entities": update.get("entities", []) or [],
            "evidence_type": _clean_text(update.get("evidence_type", "")) or (
                "single_round_image" if newest.image_paths else "single_round_text"
            ),
            "update_key": update_key,
        }

        target_memory_id = _clean_text(update.get("target_memory_id", ""))
        target_update_key = _clean_text(update.get("target_update_key", ""))
        current = None
        if target_memory_id:
            current = existing_by_id.get(target_memory_id)
        if current is None and target_update_key:
            current = existing_by_key.get((memory_type, target_update_key))
        if current is None:
            current = existing_by_key.get((memory_type, update_key))
        if action == "insert":
            current = None

        if current is None:
            counters["id"] += 1
            counters["order"] += 1
            memory_id = f"{memory_type}_{counters['id']:04d}"
            entry = MIRIXMemoryEntry(
                memory_id=memory_id,
                memory_type=memory_type,
                content=f"{summary} {details}".strip(),
                summary=summary,
                details=details,
                source_round_ids=list(dict.fromkeys(source_round_ids)),
                source_session_id=newest.session_id,
                image_paths=list(newest.image_paths),
                timestamp=newest.date,
                order_index=counters["order"],
                occurred_at=_clean_text(update.get("occurred_at", "")) or newest.date,
                caption=caption,
                name=_clean_text(update.get("name", "")),
                title=_clean_text(update.get("title", "")),
                resource_type=_clean_text(update.get("resource_type", "")),
                entry_type=_clean_text(update.get("entry_type", "")),
                metadata=metadata,
            )
            store.memories.append(entry)
            existing_by_id[entry.memory_id] = entry
            existing_by_key[(memory_type, update_key)] = entry
            continue

        current_confidence = _normalize_confidence(current.metadata.get("confidence", 0.5))
        new_confidence = metadata["confidence"]
        summary_is_better = (
            action == "replace"
            or new_confidence >= current_confidence
            or len(summary) > len(current.summary)
        )
        if summary_is_better:
            current.summary = summary
        if action == "replace":
            current.details = details
        else:
            if details and details not in current.details:
                current.details = f"{current.details} {details}".strip()
        current.content = f"{current.summary} {current.details}".strip()
        current.source_round_ids = list(dict.fromkeys(current.source_round_ids + source_round_ids))
        current.image_paths = list(dict.fromkeys(current.image_paths + list(newest.image_paths)))
        if caption:
            current.caption = caption
        if update.get("name"):
            current.name = _clean_text(update.get("name", ""))
        if update.get("title"):
            current.title = _clean_text(update.get("title", ""))
        if update.get("resource_type"):
            current.resource_type = _clean_text(update.get("resource_type", ""))
        if update.get("entry_type"):
            current.entry_type = _clean_text(update.get("entry_type", ""))
        occurred_at = _clean_text(update.get("occurred_at", ""))
        if action == "replace" and occurred_at:
            current.occurred_at = occurred_at
        elif not current.occurred_at and occurred_at:
            current.occurred_at = occurred_at
        current.metadata.update(metadata)
        existing_by_key[(memory_type, update_key)] = current


def extract_memories_incremental_llm(
    dataset: MemoryBenchmarkDataset,
    llm_config: Dict[str, Any],
) -> MIRIXMemoryStore:
    evidences = build_round_evidence(dataset, llm_config=llm_config)
    memories: List[MIRIXMemoryEntry] = []
    counters = {"id": 0, "order": 0}

    profile = dataset.data.get("character_profile", {})
    core = _extract_core_from_profile(profile, "core_0001")
    if core is not None:
        counters["id"] = 1
        counters["order"] = 1
        core.order_index = counters["order"]
        core.metadata["update_key"] = "core:profile"
        memories.append(core)

    store = MIRIXMemoryStore(
        memories=memories,
        extraction_mode="incremental_llm",
        extraction_model=llm_config.get("model", "gpt-4.1-mini"),
    )

    history_size = int(llm_config.get("history_size", 2))
    for window in iter_incremental_windows(evidences, history_size=history_size):
        newest = window[-1]
        if not newest.user_text and not newest.assistant_text and not newest.image_paths:
            continue
        try:
            meta = _call_meta_memory_agent(window, store, llm_config)
            updates: List[Dict[str, Any]] = []
            for memory_type in meta.get("memory_types", []) or []:
                focus_map = meta.get("focus", {}) if isinstance(meta.get("focus", {}), dict) else {}
                updates.extend(
                    _call_type_memory_agent(
                        memory_type,
                        window,
                        store,
                        llm_config,
                        focus=_clean_text(focus_map.get(memory_type, "")),
                    )
                )
            apply_memory_updates(store, updates, newest, counters)
        except Exception as exc:
            logger.error("Incremental LLM extraction failed for round %s: %s", newest.round_id, exc)

    logger.info(
        "Incremental LLM extraction: %d memories (%s)",
        len(store.memories),
        ", ".join(
            f"{t.value}={sum(1 for m in store.memories if m.memory_type == t.value)}"
            for t in MIRIXMemoryType
        ),
    )
    return store

# ---------------------------------------------------------------------------
# Retrieval — aligned with official MIRIX build_system_prompt_with_memories()
# ---------------------------------------------------------------------------

def _embedding_rank_entries(
    query: str,
    entries: List[MIRIXMemoryEntry],
    text_embedder: Optional[TextEmbedder],
    search_field: str = "details",
    limit: int = 10,
) -> List[MIRIXMemoryEntry]:
    """Rank entries by embedding similarity on specified field.

    Mirrors official MIRIX: ``search_method='embedding'``, searching on
    the ``details`` (or ``summary``) field of each entry.
    """
    if not entries or not query.strip():
        return []
    if text_embedder is None or not text_embedder.is_available:
        return []

    candidates: List[Tuple[str, str]] = []
    for e in entries:
        text = getattr(e, search_field, e.content) or e.content
        candidates.append((e.memory_id, text))

    scored = _dense_embedding_rank(query, candidates, text_embedder)
    id_map = {e.memory_id: e for e in entries}
    return [id_map[cid] for _, cid in scored[:limit] if cid in id_map]


def _recency_rank_entries(
    entries: List[MIRIXMemoryEntry],
    limit: int = 10,
) -> List[MIRIXMemoryEntry]:
    """Return most recent entries by occurred_at date (descending).

    Mirrors official MIRIX episodic "Most Recent" retrieval which orders by
    ``EpisodicEvent.occurred_at.desc()``. Falls back to order_index.
    """
    sorted_entries = sorted(
        entries,
        key=lambda e: (e.occurred_at or "", e.order_index),
        reverse=True,
    )
    return sorted_entries[:limit]


def _image_rank_entries(
    query: str,
    entries: List[MIRIXMemoryEntry],
    image_embedder: Optional[ImageEmbedder],
    limit: int = 5,
) -> List[MIRIXMemoryEntry]:
    """Rank entries by CLIP cross-modal image similarity.

    This is our extension for the vision-centric MemEye benchmark —
    not present in the official MIRIX evaluation code.
    """
    with_images = [e for e in entries if e.image_paths]
    if not with_images or not query.strip():
        return []
    if image_embedder is None or not image_embedder.is_available:
        return []

    candidates = [(e.memory_id, e.image_paths) for e in with_images]
    scored = _image_embedding_rank(query, candidates, image_embedder)
    id_map = {e.memory_id: e for e in with_images}
    return [id_map[cid] for _, cid in scored[:limit] if cid in id_map]


def retrieve_mirix_memories(
    store: MIRIXMemoryStore,
    query: str,
    config: Dict[str, Any],
    text_embedder: Optional[TextEmbedder] = None,
    image_embedder: Optional[ImageEmbedder] = None,
) -> Dict[str, List[MIRIXMemoryEntry]]:
    """Per-type retrieval aligned with official MIRIX.

    Official MIRIX retrieves each memory type independently and assembles
    them into separate system prompt sections. This function does the same.

    Returns:
        Dict mapping section names to lists of retrieved entries::

            {
                "core": [...],
                "episodic_recent": [...],
                "episodic_relevant": [...],
                "semantic": [...],
                "procedural": [...],
                "resource": [...],
                "knowledge_vault": [...],
                "image_relevant": [...],  # MemEye extension
            }
    """
    per_type_limit = int(config.get("per_type_limit", 10))
    image_limit = int(config.get("image_limit", 5))

    result: Dict[str, List[MIRIXMemoryEntry]] = {
        "core": [],
        "episodic_recent": [],
        "episodic_relevant": [],
        "semantic": [],
        "procedural": [],
        "resource": [],
        "knowledge_vault": [],
        "image_relevant": [],
    }

    # Core: always include all (typically 1 entry)
    result["core"] = store.get_by_type(MIRIXMemoryType.CORE)

    # Episodic: dual retrieval — recent + relevant (on details field)
    episodic = store.get_by_type(MIRIXMemoryType.EPISODIC)
    if episodic:
        result["episodic_recent"] = _recency_rank_entries(episodic, limit=per_type_limit)
        result["episodic_relevant"] = _embedding_rank_entries(
            query, episodic, text_embedder, search_field="details", limit=per_type_limit,
        )

    # Semantic: embedding search on details field
    semantic = store.get_by_type(MIRIXMemoryType.SEMANTIC)
    if semantic:
        result["semantic"] = _embedding_rank_entries(
            query, semantic, text_embedder, search_field="details", limit=per_type_limit,
        )

    # Procedural: embedding search (official uses summary field)
    proc_field = config.get("procedural_search_field", "summary")
    procedural = store.get_by_type(MIRIXMemoryType.PROCEDURAL)
    if procedural:
        result["procedural"] = _embedding_rank_entries(
            query, procedural, text_embedder, search_field=proc_field, limit=per_type_limit,
        )

    # Resource: embedding search (official uses summary field)
    resource_field = config.get("resource_search_field", "summary")
    resource = store.get_by_type(MIRIXMemoryType.RESOURCE)
    if resource:
        result["resource"] = _embedding_rank_entries(
            query, resource, text_embedder, search_field=resource_field, limit=per_type_limit,
        )

    # Knowledge Vault: embedding search (official uses caption field)
    kv_field = config.get("knowledge_vault_search_field", "caption")
    kv = store.get_by_type(MIRIXMemoryType.KNOWLEDGE_VAULT)
    if kv:
        result["knowledge_vault"] = _embedding_rank_entries(
            query, kv, text_embedder, search_field=kv_field, limit=per_type_limit,
        )

    # Image retrieval (MemEye extension): CLIP cross-modal search across all entries
    all_entries = store.all_entries()
    use_image = config.get("use_image_retrieval", True)
    if use_image:
        result["image_relevant"] = _image_rank_entries(
            query, all_entries, image_embedder, limit=image_limit,
        )

    return result


# ---------------------------------------------------------------------------
# System prompt assembly — mirrors official MIRIX XML section format
# ---------------------------------------------------------------------------

def format_retrieved_original(
    retrieved: Dict[str, List[MIRIXMemoryEntry]],
    question: str = "",
) -> str:
    """Format retrieved memories matching official MIRIX build_system_prompt().

    Reproduces the exact XML structure from the official public_evaluation branch
    (mirix/agent/agent.py:build_system_prompt).
    Key properties:
    - Includes <keywords> section (uses question text)
    - Episodic entries show summary + details char count (not full details)
    - Procedural/Resource show summary (not content)
    - Includes trailing disclaimer matching official text
    - No <image_relevant_memory> section (text-only)
    """
    sections: List[str] = []

    sections.append("Current Time: Not Specified")
    sections.append(
        f"User Focus:\n<keywords>\n{question}\n</keywords>\n"
        "These keywords have been used to retrieve relevant memories from the database."
    )

    # Core memory
    core = retrieved.get("core", [])
    if core:
        core_text = "\n".join(e.content for e in core)
        sections.append(f"<core_memory>\n{core_text}\n</core_memory>")

    # Episodic — Most Recent Events (Orderred by Timestamp)
    # Note: "Orderred" is the official typo preserved for fidelity
    ep_recent = retrieved.get("episodic_recent", [])
    if ep_recent:
        lines = []
        for idx, e in enumerate(ep_recent, 1):
            ts = e.occurred_at or e.timestamp or "Unknown"
            lines.append(
                f"- [{idx}] Timestamp: {ts} - {e.summary} "
                f"(Details: {len(e.details)} Characters)"
            )
        sections.append(
            "<episodic_memory> Most Recent Events (Orderred by Timestamp):\n"
            + "\n".join(lines)
            + "\n</episodic_memory>"
        )

    # Episodic — Most Relevant Events (Orderred by Relevance to Keywords)
    ep_relevant = retrieved.get("episodic_relevant", [])
    if ep_relevant:
        lines = []
        for idx, e in enumerate(ep_relevant, 1):
            ts = e.occurred_at or e.timestamp or "Unknown"
            lines.append(
                f"- [{idx}] Timestamp: {ts} - {e.summary} "
                f"(Details: {len(e.details)} Characters)"
            )
        sections.append(
            "<episodic_memory> Most Relevant Events (Orderred by Relevance to Keywords):\n"
            + "\n".join(lines)
            + "\n</episodic_memory>"
        )

    # Semantic
    sem = retrieved.get("semantic", [])
    if sem:
        sem_lines = [
            f"- [{idx}] Name: {e.name or e.summary or e.content}; Summary: {e.summary or e.details or e.content}"
            for idx, e in enumerate(sem, 1)
        ]
        sections.append(
            "<semantic_memory>\n" + "\n".join(sem_lines) + "\n</semantic_memory>"
        )

    # Resource
    res = retrieved.get("resource", [])
    if res:
        res_lines = [
            f"- [{idx}] Resource Title: {e.title or e.summary or e.content}; "
            f"Resource Summary: {e.summary or e.details or e.content} "
            f"Resource Type: {e.resource_type or 'reference'}"
            for idx, e in enumerate(res, 1)
        ]
        sections.append(
            "<resource_memory>\n" + "\n".join(res_lines) + "\n</resource_memory>"
        )

    # Procedural
    proc = retrieved.get("procedural", [])
    if proc:
        proc_lines = [
            f"- [{idx}] Entry Type: {e.entry_type or 'procedure'}; Summary: {e.summary or e.details or e.content}"
            for idx, e in enumerate(proc, 1)
        ]
        sections.append(
            "<procedural_memory>\n" + "\n".join(proc_lines) + "\n</procedural_memory>"
        )

    sections.append(
        "The above memories are retrieved based on the following keywords. "
        "If some memories are empty or does not contain the content related to "
        "the keywords, it is highly likely that memory does not contain any "
        "relevant information."
    )

    return "\n\n".join(sections)


def collect_images_from_retrieved(
    retrieved: Dict[str, List[MIRIXMemoryEntry]],
) -> List[str]:
    """Collect all unique image paths from retrieved entries."""
    seen: set = set()
    images: List[str] = []
    for entries in retrieved.values():
        for e in entries:
            for p in e.image_paths:
                if p not in seen:
                    seen.add(p)
                    images.append(p)
    return images
