from pathlib import Path
from typing import Any, Dict, List, Optional

from .common import load_json


def resolve_image_path(
    raw_image_path: str, dialog_json_path: Path, image_root: Optional[Path]
) -> str:
    cleaned = raw_image_path.replace("file://", "")
    source_path = Path(cleaned)
    candidates: List[Path] = []

    if source_path.is_absolute():
        candidates.append(source_path)
    else:
        candidates.append((dialog_json_path.parent / source_path).resolve())
        candidates.append((dialog_json_path.parent.parent / source_path).resolve())

        normalized = cleaned
        for prefix in ("../image/", "./image/", "image/", "data/image/"):
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix) :]
                break

        if image_root is not None:
            candidates.append((image_root / normalized).resolve())
            candidates.append((image_root / source_path.name).resolve())

        default_image_root = (dialog_json_path.parent.parent / "image").resolve()
        candidates.append((default_image_root / normalized).resolve())

    deduped: List[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)

    for candidate in deduped:
        if candidate.exists():
            return str(candidate)

    raise FileNotFoundError(
        "Could not resolve image path "
        f"'{raw_image_path}' from dialog file '{dialog_json_path}'. "
        f"Tried: {[str(p) for p in deduped]}"
    )


def build_rounds(
    dialog_data: Dict[str, Any], dialog_json_path: Path, image_root: Optional[Path]
) -> Dict[str, Dict[str, Any]]:
    rounds: Dict[str, Dict[str, Any]] = {}
    for sess in dialog_data.get("multi_session_dialogues", []):
        sid = sess.get("session_id", "")
        for d in sess.get("dialogues", []):
            rid = d.get("round", "")
            images: List[str] = []
            for rel in d.get("input_image", []) or []:
                images.append(resolve_image_path(rel, dialog_json_path, image_root))
            rounds[rid] = {
                "session_id": sid,
                "round_id": rid,
                "user": d.get("user", ""),
                "assistant": d.get("assistant", ""),
                "images": images,
                "raw": d,
            }
    return rounds


def session_order(dialog_data: Dict[str, Any]) -> List[str]:
    return [s.get("session_id", "") for s in dialog_data.get("multi_session_dialogues", [])]


def get_qas(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    for key in ("human-annotated QAs", "human_annotated_qas", "qas"):
        value = data.get(key)
        if isinstance(value, list):
            return value
    return []


def history_from_round_ids(
    session: Dict[str, Any], rounds: Dict[str, Dict[str, Any]], allowed_round_ids: Optional[set[str]] = None
) -> List[Dict[str, Any]]:
    history: List[Dict[str, Any]] = []
    for d in session.get("dialogues", []):
        rid = d.get("round", "")
        if allowed_round_ids is not None and rid not in allowed_round_ids:
            continue
        r = rounds.get(rid, {})
        user_text = r.get("user", "")
        assistant_text = r.get("assistant", "")
        images = r.get("images", [])
        if user_text:
            history.append({"role": "user", "text": user_text, "images": images, "round_id": rid})
        if assistant_text:
            history.append({"role": "assistant", "text": assistant_text, "images": [], "round_id": rid})
    return history


class PittAdsDataset:
    def __init__(self, dialog_json_path: Path, image_root: Optional[Path] = None) -> None:
        self.dialog_json_path = dialog_json_path
        self.image_root = image_root
        self.data = load_json(dialog_json_path)
        self.rounds = build_rounds(self.data, dialog_json_path, image_root)
        self.qas = get_qas(self.data)
        self.sessions = self.data.get("multi_session_dialogues", [])

    def session_order(self) -> List[str]:
        return session_order(self.data)

    def get_session(self, session_id: str) -> Dict[str, Any]:
        return next(s for s in self.sessions if s.get("session_id") == session_id)

    def iter_qas(self, limit: int = 0) -> List[Dict[str, Any]]:
        if limit > 0:
            return self.qas[:limit]
        return self.qas
