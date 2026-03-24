import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..router import GeminiAPIRouter, OpenAIAPIRouter, QwenLocalRouter

from .common import (
    REPO_ROOT,
    SCRIPT_DIR,
    get_git_commit,
    load_yaml,
    resolve_config_path,
    resolve_dataset_path,
    write_json,
    write_jsonl,
)
from .dataset import MemoryBenchmarkDataset
from .evaluator import bleu_score, extract_choice, f1_score, score_open, summarize_results, to_mcq
from .methods import get_method


@dataclass
class LegacyRunOptions:
    config_path: str
    dialog_json: str = ""
    image_root: str = ""
    model_path: str = ""
    max_new_tokens: int = 0
    output_json: str = ""
    mode: str = ""
    max_questions: int = 0

def load_sys_prompt() -> str:
    prompt_path = Path(__file__).parent / "prompt" / "sys_prompt.txt"
    return prompt_path.read_text(encoding="utf-8").strip()


def instantiate_router(model_cfg: Dict[str, Any], system_prompt: str = ""):
    provider = model_cfg.get("provider", "qwen_local")
    if provider == "qwen_local":
        return QwenLocalRouter(
            model_path=str(model_cfg["model_path"]),
            max_new_tokens=int(model_cfg.get("max_new_tokens", 128)),
            system_prompt=system_prompt,
        )
    if provider == "openai_api":
        return OpenAIAPIRouter(
            model=str(model_cfg["model"]),
            api_key=str(model_cfg.get("api_key", "")),
            api_key_env=str(model_cfg.get("api_key_env", "OPENAI_API_KEY")),
            base_url=str(model_cfg.get("base_url", "https://api.openai.com/v1")),
            max_new_tokens=int(model_cfg.get("max_new_tokens", 128)),
            timeout=int(model_cfg.get("timeout", 90)),
            system_prompt=system_prompt,
        )
    if provider == "gemini_api":
        return GeminiAPIRouter(
            model=str(model_cfg["model"]),
            api_key=str(model_cfg.get("api_key", "")),
            api_key_env=str(model_cfg.get("api_key_env", "GEMINI_API_KEY")),
            base_url=str(model_cfg.get("base_url", "https://generativelanguage.googleapis.com/v1beta")),
            max_new_tokens=int(model_cfg.get("max_new_tokens", 128)),
            timeout=int(model_cfg.get("timeout", 90)),
            system_prompt=system_prompt,
        )
    raise ValueError(f"Unsupported provider: {provider}")


def merge_legacy_config(options: LegacyRunOptions) -> Dict[str, Any]:
    cfg = load_yaml(resolve_config_path(options.config_path))
    model = cfg.setdefault("model", {})
    dataset = cfg.setdefault("dataset", {})
    ev = cfg.setdefault("eval", {})

    if options.dialog_json:
        dataset["dialog_json"] = options.dialog_json
    if options.image_root:
        dataset["image_root"] = options.image_root
    if options.model_path:
        model["model_path"] = options.model_path
    if options.max_new_tokens:
        model["max_new_tokens"] = options.max_new_tokens
    if options.output_json:
        ev["output_json"] = options.output_json
    if options.mode:
        ev["mode"] = options.mode
    if options.max_questions:
        ev["max_questions"] = options.max_questions
    cfg.setdefault("method", {}).setdefault("name", "full_context")
    cfg.setdefault("task", {}).setdefault("name", "task")
    cfg.setdefault("run", {})
    return cfg


def compose_modular_config(
    task_config_path: str,
    model_config_path: str,
    method_config_path: str,
    output_root: str = "",
    mode: str = "",
    max_questions: int = 0,
) -> Dict[str, Any]:
    task_cfg = load_yaml(resolve_config_path(task_config_path))
    model_cfg = load_yaml(resolve_config_path(model_config_path))
    method_cfg = load_yaml(resolve_config_path(method_config_path))
    cfg = {
        "task": task_cfg,
        "dataset": task_cfg.get("dataset", {}),
        "eval": task_cfg.get("eval", {}),
        "model": model_cfg,
        "method": method_cfg,
        "run": {},
    }
    if output_root:
        cfg["run"]["output_root"] = output_root
    if mode:
        cfg["eval"]["mode"] = mode
    if max_questions:
        cfg["eval"]["max_questions"] = max_questions
    return cfg


def resolve_runtime_paths(cfg: Dict[str, Any], config_dir: Path) -> Dict[str, Path]:
    dataset_cfg = cfg.get("dataset", {})
    eval_cfg = cfg.get("eval", {})
    dialog_json = resolve_dataset_path(str(dataset_cfg["dialog_json"]), config_dir)
    image_root_raw = str(dataset_cfg.get("image_root", "")).strip()
    image_root = resolve_dataset_path(image_root_raw, config_dir) if image_root_raw else None

    output_json_raw = str(eval_cfg.get("output_json", "")).strip()
    if output_json_raw:
        output_json_path = Path(output_json_raw)
        output_json = (
            output_json_path
            if output_json_path.is_absolute()
            else (SCRIPT_DIR / output_json_path).resolve()
        )
    else:
        output_json = None
    output_root_raw = str(cfg.get("run", {}).get("output_root", "")).strip()
    if output_root_raw:
        output_root_path = Path(output_root_raw)
        output_root = output_root_path if output_root_path.is_absolute() else (SCRIPT_DIR / output_root_path)
        output_root = output_root.resolve()
    else:
        output_root = (SCRIPT_DIR / "runs").resolve()
    return {
        "dialog_json": dialog_json,
        "image_root": image_root,
        "output_json": output_json,
        "output_root": output_root,
    }


def default_run_dir(cfg: Dict[str, Any], output_root: Path) -> Path:
    task_name = str(cfg.get("task", {}).get("name", "task"))
    model_name = str(cfg.get("model", {}).get("name", "model"))
    method_name = str(cfg.get("method", {}).get("name", "method"))
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return output_root / task_name / f"{ts}_{model_name}_{method_name}"


def legacy_output_path(cfg: Dict[str, Any], output_json: Path) -> Path:
    task_name = str(cfg.get("task", {}).get("name", "task"))
    model_name = str(cfg.get("model", {}).get("name", "model"))
    method_name = str(cfg.get("method", {}).get("name", "method"))
    stem = output_json.stem
    suffix = output_json.suffix or ".json"
    task_dir = output_json.parent / task_name
    return task_dir / f"{stem}__{model_name}__{method_name}{suffix}"


def build_payload(
    cfg: Dict[str, Any],
    paths: Dict[str, Path],
    run_dir: Path,
    dataset: MemoryBenchmarkDataset,
    results: List[Dict[str, Any]],
    method_runtime: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    summary = summarize_results(results)
    model_ref = cfg["model"].get("model_path") or cfg["model"].get("model", "")
    payload = {
        "task_name": cfg.get("task", {}).get("name", "task"),
        "model_name": cfg.get("model", {}).get("name", "qwen_local"),
        "model_path": model_ref,
        "method_name": cfg.get("method", {}).get("name", "full_context"),
        "mode": cfg["eval"].get("mode", "open"),
        "num_qas": len(dataset.qas),
        "num_qas_run": len({r["idx"] for r in results}),
        "dialog_json": str(paths["dialog_json"]),
        "image_root": str(paths["image_root"]) if paths["image_root"] else "",
        "run_dir": str(run_dir),
        "git_commit": get_git_commit(REPO_ROOT),
        "summary": summary,
        "results": results,
    }
    if method_runtime:
        payload["method_runtime"] = method_runtime
    return payload


def run_benchmark(cfg: Dict[str, Any], config_dir: Path) -> Dict[str, Any]:
    paths = resolve_runtime_paths(cfg, config_dir)
    mode = str(cfg.get("eval", {}).get("mode", "open"))
    max_questions = int(cfg.get("eval", {}).get("max_questions", 0))

    dataset = MemoryBenchmarkDataset(paths["dialog_json"], paths["image_root"])
    method = get_method(
        str(cfg.get("method", {}).get("name", "full_context")),
        config=dict(cfg.get("method", {})),
    )
    router = instantiate_router(cfg["model"], system_prompt=load_sys_prompt())

    qas = dataset.iter_qas(limit=max_questions)
    results: List[Dict[str, Any]] = []
    for i, qa in enumerate(qas, start=1):
        question = qa.get("question", "")
        gt = qa.get("answer", "")
        history = method.build_history(dataset, qa)
        print(
            f"[INFO] QA {i}/{len(qas)} point={qa.get('point')} "
            f"method={method.name} history_turns={len(history)}"
        )

        if mode in {"open", "both"}:
            t0 = dt.datetime.now()
            pred = router.answer(history, question)
            latency_ms = int((dt.datetime.now() - t0).total_seconds() * 1000)
            exact, contains = score_open(pred, gt)
            f1 = f1_score(pred, gt)
            bleu_1 = bleu_score(pred, gt, weights=(1.0, 0.0, 0.0, 0.0))
            bleu_2 = bleu_score(pred, gt, weights=(0.5, 0.5, 0.0, 0.0))
            results.append(
                {
                    "idx": i,
                    "point": qa.get("point"),
                    "mode": "open",
                    "question": question,
                    "gt": gt,
                    "pred": pred,
                    "exact_match": exact,
                    "em": 1.0 if exact else 0.0,
                    "contains_gt": contains,
                    "f1": f1,
                    "bleu_1": bleu_1,
                    "bleu_2": bleu_2,
                    "latency_ms": latency_ms,
                    "method_name": method.name,
                    "history_turns": len(history),
                    "source_sessions": qa.get("session_id", []),
                    "clue_rounds": qa.get("clue", []),
                }
            )
            print(
                f"[OPEN][{i}] exact={exact} contains={contains} "
                f"f1={f1:.3f} bleu_1={bleu_1:.3f} bleu_2={bleu_2:.3f} latency_ms={latency_ms}"
            )

        if mode in {"mcq", "both"}:
            mcq_question = to_mcq(question)
            t0 = dt.datetime.now()
            pred_raw = router.answer(history, mcq_question)
            latency_ms = int((dt.datetime.now() - t0).total_seconds() * 1000)
            choice = extract_choice(pred_raw)
            results.append(
                {
                    "idx": i,
                    "point": qa.get("point"),
                    "mode": "mcq",
                    "question": mcq_question,
                    "gt": gt,
                    "pred_raw": pred_raw,
                    "pred_choice": choice,
                    "valid_choice": choice in {"A", "B", "C"},
                    "latency_ms": latency_ms,
                    "method_name": method.name,
                    "history_turns": len(history),
                    "source_sessions": qa.get("session_id", []),
                    "clue_rounds": qa.get("clue", []),
                }
            )
            print(f"[MCQ][{i}] choice={choice} latency_ms={latency_ms}")

    run_dir = default_run_dir(cfg, paths["output_root"])
    run_dir.mkdir(parents=True, exist_ok=True)
    method_runtime = dict(getattr(method, "runtime_info", {}) or {})
    payload = build_payload(cfg, paths, run_dir, dataset, results, method_runtime=method_runtime)
    cfg_to_write = dict(cfg)
    run_cfg = dict(cfg_to_write.get("run", {}))
    if method_runtime:
        run_cfg["method_runtime"] = method_runtime
    cfg_to_write["run"] = run_cfg

    write_json(run_dir / "config.json", cfg_to_write)
    write_json(run_dir / "metrics.json", {k: payload[k] for k in payload if k != "results"})
    write_jsonl(run_dir / "predictions.jsonl", results)

    if paths["output_json"] is not None:
        legacy_output_json = legacy_output_path(cfg, paths["output_json"])
        legacy_payload = {
            "dialog_json": payload["dialog_json"],
            "model_path": payload["model_path"],
            "mode": payload["mode"],
            "num_qas": payload["num_qas_run"],
            "summary": payload["summary"],
            "results": payload["results"],
            "method_name": payload["method_name"],
            "run_dir": payload["run_dir"],
            "git_commit": payload["git_commit"],
        }
        write_json(legacy_output_json, legacy_payload)
        print(f"[INFO] Saved legacy output: {legacy_output_json}")

    print(f"[INFO] Saved run artifacts: {run_dir}")
    return payload


def run_legacy_benchmark(options: LegacyRunOptions) -> Dict[str, Any]:
    cfg = merge_legacy_config(options)
    config_dir = resolve_config_path(options.config_path).parent
    return run_benchmark(cfg, config_dir)


def run_modular_benchmark(
    task_config_path: str,
    model_config_path: str,
    method_config_path: str,
    output_root: str = "",
    mode: str = "",
    max_questions: int = 0,
) -> Dict[str, Any]:
    cfg = compose_modular_config(
        task_config_path=task_config_path,
        model_config_path=model_config_path,
        method_config_path=method_config_path,
        output_root=output_root,
        mode=mode,
        max_questions=max_questions,
    )
    config_dir = resolve_config_path(task_config_path).parent
    return run_benchmark(cfg, config_dir)
