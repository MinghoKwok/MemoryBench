import importlib.util
import json
import os
import sys
import types
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from openai import OpenAI

from .common import SCRIPT_DIR, write_json
from .dataset import MemoryBenchmarkDataset, validate_text_only_captions
from .methods import HistoryMethod


_GEN_AGENTS_ROOT = (Path(__file__).resolve().parent / "gen_agents" / "upstream").resolve()
_LOCAL_ENV_PATH = (SCRIPT_DIR / ".env.local").resolve()


def _install_langchain_compat() -> None:
    if importlib.util.find_spec("langchain.prompts") is not None:
        return

    from langchain_core.prompts import PromptTemplate

    prompts_mod = types.ModuleType("langchain.prompts")
    prompts_mod.PromptTemplate = PromptTemplate
    sys.modules["langchain.prompts"] = prompts_mod

    try:
        import langchain  # type: ignore

        setattr(langchain, "prompts", prompts_mod)
    except Exception:
        pass


def _ensure_gen_agents_importable() -> None:
    _install_langchain_compat()
    if str(_GEN_AGENTS_ROOT) not in sys.path:
        sys.path.insert(0, str(_GEN_AGENTS_ROOT))


def _read_local_env() -> Dict[str, str]:
    if not _LOCAL_ENV_PATH.exists():
        return {}
    values: Dict[str, str] = {}
    for raw_line in _LOCAL_ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key:
            values[key] = value
    return values


def _resolve_secret(
    primary_config: Dict[str, Any],
    fallback_config: Dict[str, Any],
    raw_key: str,
    env_key: str,
    default_env: str,
) -> str:
    raw = str(primary_config.get(raw_key, "")).strip()
    if raw:
        return raw
    raw = str(fallback_config.get(raw_key, "")).strip()
    if raw:
        return raw

    env_name = str(primary_config.get(env_key, "")).strip() or str(fallback_config.get(env_key, "")).strip() or default_env
    if env_name:
        env_value = str(os.getenv(env_name, "")).strip()
        if env_value:
            return env_value
        local_env = _read_local_env()
        return str(local_env.get(env_name, "")).strip()
    return ""


def _resolve_model_name(method_config: Dict[str, Any], model_config: Dict[str, Any], key: str, default: str) -> str:
    explicit = str(method_config.get(key, "")).strip()
    if explicit:
        return explicit
    if key == "llm_model":
        model_name = str(model_config.get("model", "")).strip()
        if model_name:
            return model_name
    return default


def _resolve_base_url(method_config: Dict[str, Any], model_config: Dict[str, Any]) -> str:
    raw = str(method_config.get("base_url", "")).strip()
    if raw:
        return raw
    raw = str(model_config.get("base_url", "")).strip()
    if raw:
        return raw
    return "https://api.openai.com/v1"


def _dialogue_text(round_payload: Dict[str, Any], speaker_a: str, speaker_b: str) -> str:
    parts: List[str] = []
    user_text = str(round_payload.get("user", "")).strip()
    assistant_text = str(round_payload.get("assistant", "")).strip()
    if user_text:
        parts.append(f"{speaker_a}: {user_text}")
    if assistant_text:
        parts.append(f"{speaker_b}: {assistant_text}")
    return "\n".join(parts).strip()


def _round_image_blocks(raw_dialogue: Dict[str, Any]) -> List[str]:
    input_images = raw_dialogue.get("input_image", []) or []
    captions = raw_dialogue.get("image_caption", []) or []
    blocks: List[str] = []
    for idx, _ in enumerate(input_images):
        caption = str(captions[idx]).strip() if idx < len(captions) else ""
        blocks.extend(
            [
                "image:",
                f"image_caption: {caption}",
            ]
        )
    return blocks


def _build_round_text(round_payload: Dict[str, Any], speaker_a: str, speaker_b: str) -> str:
    raw_dialogue = round_payload.get("raw", {}) or {}
    text = _dialogue_text(round_payload, speaker_a, speaker_b)
    lines: List[str] = [text] if text else []
    lines.extend(_round_image_blocks(raw_dialogue))
    return "\n".join(line for line in lines if line).strip()


def _question_with_image_caption(qa: Dict[str, Any], question: str) -> str:
    query = question.strip()
    question_image = qa.get("question_image")
    has_question_image = False
    if isinstance(question_image, str):
        has_question_image = bool(question_image.strip())
    elif isinstance(question_image, list):
        has_question_image = any(str(item).strip() for item in question_image)
    if not has_question_image:
        question_images = qa.get("question_images")
        if isinstance(question_images, list):
            has_question_image = any(str(item).strip() for item in question_images)
    if not has_question_image:
        return query

    question_caption = qa.get("image_caption")
    if not question_caption:
        return query

    if isinstance(question_caption, list):
        caption_text = " ".join(str(item).strip() for item in question_caption if str(item).strip())
    else:
        caption_text = str(question_caption).strip()
    if not caption_text:
        return query
    return f"{query}\nquestion's image:\nimage_caption: {caption_text}"


def _safe_task_name(dataset: MemoryBenchmarkDataset) -> str:
    task_name = str(dataset.data.get("task_name", "")).strip() or dataset.dialog_json_path.stem
    return task_name.lower().replace(" ", "_").replace("/", "_")


class OpenAIEmbeddingEncoder:
    def __init__(self, config: Any) -> None:
        self.config = config
        api_key = str(getattr(config, "api_key", "")).strip()
        if not api_key:
            raise ValueError(
                "OpenAI API key not found for Generative Agents embeddings. "
                "Set OPENAI_API_KEY in the environment or Benchmark_Pipeline/.env.local."
            )
        base_url = str(getattr(config, "base_url", "https://api.openai.com/v1")).strip() or "https://api.openai.com/v1"
        timeout = int(getattr(config, "timeout", 90) or 90)
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        self.model = str(getattr(config, "path", "")).strip() or str(getattr(config, "name", "")).strip()
        if not self.model:
            raise ValueError("OpenAIEmbeddingEncoder requires an embedding model name.")
        self.device = "cpu"

    def reset(self) -> None:
        return None

    def __call__(self, text: str, return_type: str = "numpy") -> Any:
        response = self.client.embeddings.create(model=self.model, input=[str(text)])
        embedding = np.asarray([response.data[0].embedding], dtype=np.float32)
        if return_type == "numpy":
            return embedding
        if return_type == "tensor":
            import torch

            return torch.from_numpy(embedding)
        raise ValueError(f"Unrecognized return type: {return_type}")


class GAMethod(HistoryMethod):
    name = "gen_agents"
    fixed_modality = "text_only"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config=config)
        self._dataset_key: Optional[int] = None
        self._memory: Optional[Any] = None
        self._speaker_a: str = "user"
        self._speaker_b: str = "assistant"
        self._debug_rows: List[Dict[str, Any]] = []
        self._empty_memory_text: str = str(self.config.get("empty_memory", "No relevant memory retrieved.")).strip()
        self._answer_client: Optional[OpenAI] = None
        self._answer_model: str = ""
        self._answer_timeout: int = 90

    def _debug_dir(self, dataset: MemoryBenchmarkDataset) -> Path:
        return (SCRIPT_DIR / "output" / _safe_task_name(dataset) / "gen_agents").resolve()

    def _flush_debug(self, dataset: MemoryBenchmarkDataset) -> None:
        if not self._debug_rows:
            return
        payload = {
            "dataset_path": str(dataset.dialog_json_path),
            "rows": self._debug_rows,
        }
        write_json(self._debug_dir(dataset) / "debug_trace.json", payload)

    def _ensure_caption_preprocessed(self, dataset: MemoryBenchmarkDataset) -> None:
        if not bool(self.config.get("caption_preprocessed", True)):
            return
        self.runtime_info.update(validate_text_only_captions(dataset.rounds))

    def _install_openai_encoder(self) -> None:
        _ensure_gen_agents_importable()
        import memengine.function as memengine_function  # type: ignore
        import memengine.function.Encoder as encoder_mod  # type: ignore
        import memengine.function.Retrieval as retrieval_mod  # type: ignore

        setattr(memengine_function, "OpenAIEmbeddingEncoder", OpenAIEmbeddingEncoder)
        setattr(encoder_mod, "OpenAIEmbeddingEncoder", OpenAIEmbeddingEncoder)
        setattr(retrieval_mod, "OpenAIEmbeddingEncoder", OpenAIEmbeddingEncoder)

    def _ensure_answer_client(self, method_config: Dict[str, Any], model_config: Dict[str, Any]) -> None:
        provider = str(model_config.get("provider", "")).strip().lower() or "openai_api"
        if provider != "openai_api":
            raise ValueError(
                f"Unsupported answer model provider for Generative Agents benchmark QA: {provider}. "
                "Use an OpenAI API model config for final answer generation."
            )
        api_key = _resolve_secret(model_config, method_config, "api_key", "api_key_env", "OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key not found for Generative Agents final answer model. "
                "Set OPENAI_API_KEY in the environment or Benchmark_Pipeline/.env.local."
            )
        self._answer_model = str(model_config.get("model", "")).strip() or "gpt-4.1-nano"
        base_url = _resolve_base_url(method_config, model_config)
        self._answer_timeout = int(model_config.get("timeout", 90) or 90)
        self._answer_client = OpenAI(api_key=api_key, base_url=base_url, timeout=self._answer_timeout)

    def _load_json_object(self, raw: str, fallback_key: str) -> str:
        try:
            payload = json.loads(raw)
            if isinstance(payload, dict):
                value = payload.get(fallback_key, "")
                if isinstance(value, str):
                    return value.strip()
        except Exception:
            pass
        return raw.strip()

    def _retrieve_context(self, qa: Dict[str, Any]) -> Dict[str, Any]:
        self._ensure_initialized(self._active_dataset)
        assert self._memory is not None

        from .runner import format_question

        question = format_question(qa)
        recall_query = _question_with_image_caption(qa, question)
        recall_text = self._memory.recall({"text": recall_query, "timestamp": "query"})
        retrieved_round_ids = list(getattr(self._memory.recall_op, "last_retrieved_ids", []) or [])
        recall_text_value = str(recall_text).strip()
        self.runtime_info.update(
            {
                "method_modality": self.modality,
                "history_source": "retrieval",
                "captions_loaded": True,
                "images_loaded": False,
                "history_turns_after_truncation": 1 if recall_text_value and recall_text_value != self._empty_memory_text else 0,
                "retrieved_round_ids": retrieved_round_ids,
            }
        )
        return {
            "question": question,
            "recall_query": recall_query,
            "recall_text": recall_text_value,
            "retrieved_round_ids": retrieved_round_ids,
        }

    def _build_ga_config(self, method_config: Dict[str, Any], model_config: Dict[str, Any]) -> Dict[str, Any]:
        api_key = _resolve_secret(method_config, model_config, "llm_api_key", "llm_api_key_env", "OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key not found for Generative Agents. "
                "Set OPENAI_API_KEY in the environment or Benchmark_Pipeline/.env.local."
            )

        base_url = _resolve_base_url(method_config, model_config)
        llm_model = _resolve_model_name(method_config, model_config, "llm_model", "gpt-4.1-mini")
        embedding_model = str(method_config.get("embedding_model", "text-embedding-3-small")).strip() or "text-embedding-3-small"
        llm_temperature = float(method_config.get("llm_temperature", 0.0))
        importance_post_scale = float(method_config.get("importance_post_scale", 10.0))
        retrieve_k = max(1, int(method_config.get("retrieve_k", 10)))
        reflection_threshold = float(method_config.get("reflection_threshold", 8.0))
        reflection_topk = max(1, int(method_config.get("reflection_topk", 2)))
        reflection_questions = max(1, int(method_config.get("reflection_question_number", 2)))
        reflection_insights = max(1, int(method_config.get("reflection_insight_number", 2)))
        recency_decay = float(method_config.get("recency_decay", 0.995))
        truncation_words = max(1, int(method_config.get("recall_truncation_words", 4000)))

        return {
            "global_config": {"usable_gpu": ""},
            "storage": {},
            "display": {
                "method": "ScreenDisplay",
                "prefix": "[%s]",
                "suffix": "",
                "key_value_sep": ": ",
                "key_format": "%s",
                "item_sep": "\n",
            },
            "store": {},
            "recall": {
                "empty_memory": self._empty_memory_text,
                "topk": retrieve_k,
                "truncation": {
                    "method": "LMTruncation",
                    "mode": "word",
                    "number": truncation_words,
                    "path": "bert-base-uncased",
                },
                "utilization": {
                    "method": "ConcateUtilization",
                    "prefix": "",
                    "suffix": "",
                    "list_config": {"index": True, "sep": "\n\n"},
                    "dict_config": {"key_value_sep": ": ", "key_format": "%s", "item_sep": "\n"},
                },
                "text_retrieval": {
                    "method": "TextRetrieval",
                    "topk": retrieve_k,
                    "mode": "cosine",
                    "encoder": {
                        "method": "OpenAIEmbeddingEncoder",
                        "path": embedding_model,
                        "api_key": api_key,
                        "base_url": base_url,
                        "timeout": int(method_config.get("embedding_timeout", model_config.get("timeout", 90) or 90)),
                    },
                },
                "time_retrieval": {
                    "method": "TimeRetrieval",
                    "topk": retrieve_k,
                    "mode": "exp",
                    "coef": {"decay": recency_decay},
                },
                "importance_retrieval": {
                    "method": "ValueRetrieval",
                    "topk": retrieve_k,
                    "mode": "identical",
                },
                "importance_judge": {
                    "method": "LLMJudge",
                    "post_scale": importance_post_scale,
                    "LLM_config": {
                        "method": "APILLM",
                        "name": llm_model,
                        "api_key": api_key,
                        "base_url": base_url,
                        "temperature": llm_temperature,
                    },
                    "prompt": {
                        "input_variables": ["message"],
                        "template": (
                            "Rate the importance of the following memory for a personal assistant's long-term memory "
                            "on a scale from 1 to 10. Respond with only a number.\n\nMemory:\n{message}"
                        ),
                    },
                },
            },
            "reflect": {
                "reflector": {
                    "threshold": reflection_threshold,
                    "reflection_topk": reflection_topk,
                    "question_number": reflection_questions,
                    "insight_number": reflection_insights,
                    "LLM_config": {
                        "method": "APILLM",
                        "name": llm_model,
                        "api_key": api_key,
                        "base_url": base_url,
                        "temperature": llm_temperature,
                    },
                    "question_prompt": {
                        "input_variables": ["information", "question_number"],
                        "template": (
                            "Given the following recent memories, generate exactly {question_number} high-level "
                            "questions that would help synthesize or interpret them. Return one question per line.\n\n"
                            "{information}"
                        ),
                    },
                    "insight_prompt": {
                        "input_variables": ["statements", "insight_number"],
                        "template": (
                            "Given the following evidence statements, generate exactly {insight_number} concise "
                            "insights that summarize durable information worth remembering. Return one insight per line.\n\n"
                            "{statements}"
                        ),
                    },
                }
            },
        }

    def _maybe_reflect(self, dataset: MemoryBenchmarkDataset) -> List[Dict[str, Any]]:
        assert self._memory is not None
        reflector = self._memory.reflect_op.reflector
        if reflector.get_current_accmulated_importance() < reflector.get_reflection_threshold():
            return []

        before = self._memory.storage.get_element_number()
        self._memory.manage("reflect")
        after = self._memory.storage.get_element_number()
        new_items: List[Dict[str, Any]] = []
        for mid in range(before, after):
            item = self._memory.storage.get_memory_element_by_mid(mid)
            if isinstance(item, dict):
                new_items.append(item)
        for item in new_items:
            self._debug_rows.append(
                {
                    "type": "reflection_memory",
                    "timestamp": str(item.get("timestamp", "")),
                    "source": item.get("source", False),
                    "text": str(item.get("text", "")),
                }
            )
        if new_items:
            self._flush_debug(dataset)
        return new_items

    def _ensure_initialized(self, dataset: MemoryBenchmarkDataset) -> None:
        dataset_id = id(dataset)
        if self._memory is not None and self._dataset_key == dataset_id:
            return

        self._active_dataset = dataset
        self._ensure_caption_preprocessed(dataset)
        self._install_openai_encoder()

        from memengine.config.Config import MemoryConfig  # type: ignore
        from memengine.memory.GAMemory import GAMemory  # type: ignore

        self._debug_rows = []
        model_config = dict(self.config.get("_model_cfg", {}))
        ga_config = self._build_ga_config(self.config, model_config)
        self._memory = GAMemory(MemoryConfig(ga_config))
        self._ensure_answer_client(self.config, model_config)

        character_profile = dataset.data.get("character_profile", {}) or {}
        speaker_name = str(character_profile.get("name") or character_profile.get("primary_user") or "").strip()
        self._speaker_a = f"user ({speaker_name})" if speaker_name else "user"
        self._speaker_b = "assistant"

        stored_count = 0
        for session_id in dataset.session_order():
            session = dataset.get_session(session_id)
            timestamp = str(session.get("date", "")).strip() or None
            for dialogue in session.get("dialogues", []):
                round_id = str(dialogue.get("round", "")).strip()
                if not round_id or round_id not in dataset.rounds:
                    continue
                round_payload = dataset.rounds[round_id]
                note_text = _build_round_text(round_payload, self._speaker_a, self._speaker_b)
                if not note_text:
                    continue
                observation = {
                    "text": note_text,
                    "timestamp": timestamp,
                    "dialogue_id": round_id,
                    "source": False,
                }
                self._memory.store(observation)
                stored_count += 1
                self._debug_rows.append(
                    {
                        "type": "stored_memory",
                        "round_id": round_id,
                        "session_id": round_payload.get("session_id", ""),
                        "timestamp": timestamp or "",
                        "text": note_text,
                    }
                )
                self._maybe_reflect(dataset)

        self._dataset_key = dataset_id
        self.runtime_info.update(
            {
                "method_modality": self.modality,
                "history_source": "retrieval",
                "captions_loaded": True,
                "images_loaded": False,
                "num_memories": stored_count,
                "debug_dir": str(self._debug_dir(dataset)),
            }
        )
        self._flush_debug(dataset)

    def build_history(self, dataset: MemoryBenchmarkDataset, qa: Dict[str, Any]) -> List[Dict[str, Any]]:
        self._active_dataset = dataset
        payload = self._retrieve_context(qa)
        question = payload["question"]
        recall_query = payload["recall_query"]
        recall_text_value = payload["recall_text"]
        retrieved_round_ids = payload["retrieved_round_ids"]

        history: List[Dict[str, Any]] = []
        if recall_text_value and recall_text_value != self._empty_memory_text:
            history.append(
                {
                    "role": "assistant",
                    "text": "Retrieved Generative Agents memory context:\n" + recall_text_value,
                    "images": [],
                }
            )

        self.runtime_info.update(
            {
                "method_modality": self.modality,
                "history_source": "retrieval",
                "captions_loaded": True,
                "images_loaded": False,
                "history_turns_after_truncation": len(history),
                "retrieved_round_ids": retrieved_round_ids,
            }
        )
        self._debug_rows.append(
            {
                "type": "qa_recall",
                "question_id": qa.get("question_id", ""),
                "question": question,
                "recall_query": recall_query,
                "retrieved_round_ids": retrieved_round_ids,
                "recall_text": recall_text_value,
            }
        )
        self._flush_debug(dataset)
        return history

    def answer(self, dataset: MemoryBenchmarkDataset, qa: Dict[str, Any], question: str) -> str:
        self._active_dataset = dataset
        payload = self._retrieve_context(qa)
        recall_text_value = payload["recall_text"]
        recall_query = payload["recall_query"]
        retrieved_round_ids = payload["retrieved_round_ids"]

        assert self._answer_client is not None
        options = qa.get("options") if isinstance(qa.get("options"), dict) else None
        if options:
            option_keys = [str(key).strip() for key in sorted(options.keys()) if str(key).strip()]
            user_prompt = (
                "Use only the retrieved context below to answer the multiple-choice question.\n"
                "If the context is insufficient, still choose the single best option.\n\n"
                f"Retrieved context:\n{recall_text_value or self._empty_memory_text}\n\n"
                f"Question:\n{question}\n\n"
                f"Return only one option letter from: {', '.join(option_keys)}."
            )
            response = self._answer_client.chat.completions.create(
                model=self._answer_model,
                messages=[
                    {"role": "system", "content": "You must respond with a JSON object."},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "mcq_response",
                        "schema": {
                            "type": "object",
                            "properties": {"answer": {"type": "string", "enum": option_keys}},
                            "required": ["answer"],
                            "additionalProperties": False,
                        },
                        "strict": True,
                    },
                },
                temperature=0.0,
            )
        else:
            user_prompt = (
                "Use only the retrieved context below to answer the question.\n"
                "Answer concisely and factually.\n\n"
                f"Retrieved context:\n{recall_text_value or self._empty_memory_text}\n\n"
                f"Question:\n{question}"
            )
            response = self._answer_client.chat.completions.create(
                model=self._answer_model,
                messages=[
                    {"role": "system", "content": "You must respond with a JSON object."},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "open_response",
                        "schema": {
                            "type": "object",
                            "properties": {"answer": {"type": "string"}},
                            "required": ["answer"],
                            "additionalProperties": False,
                        },
                        "strict": True,
                    },
                },
                temperature=0.0,
            )

        raw_response = response.choices[0].message.content
        answer = self._load_json_object(raw_response, "answer")
        self._debug_rows.append(
            {
                "type": "qa_answer",
                "question_id": qa.get("question_id", ""),
                "question": question,
                "recall_query": recall_query,
                "retrieved_round_ids": retrieved_round_ids,
                "recall_text": recall_text_value,
                "answer_prompt": user_prompt,
                "prediction": answer,
            }
        )
        self._flush_debug(dataset)
        return answer
