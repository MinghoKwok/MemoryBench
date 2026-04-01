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
from .evaluator import (
    bert_score_metric,
    bleu_score,
    extract_choice,
    f1_score,
    llm_judge_score,
    score_open,
    summarize_results,
    to_mcq,
)
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


def merge_legacy_config(opts: LegacyRunOptions) -> Dict[str, Any]:
    # Try to load base config from file
    base: Dict[str, Any] = {}
    if opts.config_path:
        try:
            base = load_yaml(resolve_config_path(opts.config_path))
        except Exception:
            pass

    # Build merged config; CLI opts override file-based config
    dataset = dict(base.get("dataset", {}))
    if opts.dialog_json:
        dataset["dialog_json"] = opts.dialog_json
    if opts.image_root:
        dataset["image_root"] = opts.image_root

    model = dict(base.get("model", {}))
    if opts.model_path:
        model.update({"provider": "qwen_local", "name": "legacy_model", "model_path": opts.model_path})
    if opts.max_new_tokens:
        model["max_new_tokens"] = opts.max_new_tokens
    model.setdefault("provider", "qwen_local")
    model.setdefault("name", "legacy_model")
    model.setdefault("max_new_tokens", 128)

    eval_cfg = dict(base.get("eval", {}))
    if opts.mode:
        eval_cfg["mode"] = opts.mode
    if opts.max_questions:
        eval_cfg["max_questions"] = opts.max_questions
    eval_cfg.setdefault("mode", "open")
    eval_cfg.setdefault("max_questions", 0)

    run_cfg = dict(base.get("run", {}))
    if opts.output_json:
        run_cfg["output_root"] = str(Path(opts.output_json).parent)

    return {
        "task": base.get("task", {"name": "legacy"}),
        "dataset": dataset,
        "eval": eval_cfg,
        "model": model,
        "method": base.get("method", {"name": "full_context"}),
        "run": run_cfg,
    }



def load_sys_prompt() -> str:
    """Load MemEye system prompt from benchmark/prompt/sys_prompt.txt."""
    prompt_path = Path(__file__).parent / "prompt" / "sys_prompt.txt"
    return prompt_path.read_text(encoding="utf-8").strip()


def instantiate_router(model_cfg: Dict[str, Any], system_prompt: str = ""):
    provider = model_cfg.get("provider", "qwen_local")
    if provider == "qwen_local":
        return QwenLocalRouter(
            model_path=str(model_cfg["model_path"]),
            max_new_tokens=int(model_cfg.get("max_new_tokens", 128)),
            system_prompt=system_prompt,
            max_time=model_cfg.get("max_time", 25),
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

    output_root_raw = str(cfg.get("run", {}).get("output_root", "")).strip()
    if output_root_raw:
        output_root_path = Path(output_root_raw)
        output_root = output_root_path if output_root_path.is_absolute() else (SCRIPT_DIR / output_root_path)
        output_root = output_root.resolve()
    else:
        output_root = (SCRIPT_DIR / "runs").resolve()
    output_json_raw = str(eval_cfg.get("output_json", "")).strip()
    if output_json_raw:
        oj = Path(output_json_raw)
        output_json: Optional[Path] = oj if oj.is_absolute() else (SCRIPT_DIR / oj).resolve()
    else:
        output_json = None

    return {
        "dialog_json": dialog_json,
        "image_root": image_root,
        "output_root": output_root,
        "output_json": output_json,
    }


def default_run_dir(cfg: Dict[str, Any], output_root: Path) -> Path:
    task_name = str(cfg.get("task", {}).get("name", "task"))
    model_name = str(cfg.get("model", {}).get("name", "model"))
    method_name = str(cfg.get("method", {}).get("name", "method"))
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return output_root / task_name / f"{ts}_{model_name}_{method_name}"



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


def run_benchmark(
    cfg: Dict[str, Any],
    config_dir: Path,
    enable_bert_score: bool = False,
    enable_llm_judge: bool = False,
    judge_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    paths = resolve_runtime_paths(cfg, config_dir)
    mode = str(cfg.get("eval", {}).get("mode", "open"))
    max_questions = int(cfg.get("eval", {}).get("max_questions", 0))

    dataset = MemoryBenchmarkDataset(paths["dialog_json"], paths["image_root"])
    method = get_method(
        str(cfg.get("method", {}).get("name", "full_context")),
        config=dict(cfg.get("method", {})),
    )
    # Agentic methods (for example M2A) own end-to-end inference via answer().
    # They bypass build_history() + router.answer() and may keep internal runtime state.
    is_agentic = hasattr(method, "answer") and callable(getattr(method, "answer"))
    router = None
    if not is_agentic:
        sys_prompt = load_sys_prompt()
        router = instantiate_router(cfg["model"], system_prompt=sys_prompt)

    # Build LLM judge client once before the loop
    _judge_client = None
    _judge_template = None
    _judge_model = None
    if enable_llm_judge:
        if not judge_config:
            raise ValueError("judge_config must be provided when enable_llm_judge=True")
        from openai import OpenAI
        _judge_client = OpenAI(
            api_key=judge_config.get("api_key") or None,
            base_url=judge_config.get("base_url") or None,
        )
        _judge_model = judge_config["model"]
        prompt_path = Path(__file__).parent / "llm_judge.txt"
        _judge_template = prompt_path.read_text(encoding="utf-8")

    qas = dataset.iter_qas(limit=max_questions)
    results: List[Dict[str, Any]] = []
    for i, qa in enumerate(qas, start=1):
        question = qa.get("question", "")
        gt = qa.get("answer", "")

        if is_agentic:
            history: List[Dict[str, Any]] = []
        else:
            history = method.build_history(dataset, qa)
        current_method_runtime = dict(getattr(method, "runtime_info", {}) or {})
        print(
            f"[INFO] QA {i}/{len(qas)} point={qa.get('point')} "
            f"method={method.name} history_turns={len(history)}"
        )

        if mode in {"open", "both"}:
            t0 = dt.datetime.now()
            if is_agentic:
                pred = method.answer(dataset, qa, question)
            else:
                pred = router.answer(history, question)
            latency_ms = int((dt.datetime.now() - t0).total_seconds() * 1000)

            exact, contains = score_open(pred, gt)
            _f1    = f1_score(pred, gt)
            _bleu  = bleu_score(pred, gt)
            _bleu1 = bleu_score(pred, gt, weights=(1, 0, 0, 0))
            _bleu2 = bleu_score(pred, gt, weights=(0.5, 0.5, 0, 0))
            _bert  = bert_score_metric(pred, gt) if enable_bert_score else None

            _judge: Optional[float] = None
            _judge_reasoning: Optional[str] = None
            if enable_llm_judge and _judge_client is not None:
                try:
                    jr = llm_judge_score(
                        question=question,
                        ground_truth=gt,
                        model_output=pred,
                        client=_judge_client,
                        model_name=_judge_model,
                        prompt_template=_judge_template,
                        max_retries=judge_config.get("max_retries", 3),
                        timeout=judge_config.get("timeout", 60),
                    )
                    _judge = jr["score"]
                    _judge_reasoning = jr.get("reasoning", "")
                except RuntimeError as exc:
                    print(f"[WARN] LLM judge failed for QA {i}: {exc}")

            open_result = {
                "idx": i,
                "point": qa.get("point"),
                "mode": "open",
                "question": question,
                "gt": gt,
                "pred": pred,
                "exact_match": exact,
                "em": 1.0 if exact else 0.0,
                "contains_gt": contains,
                "f1": _f1,
                "bleu": _bleu,
                "bleu_1": _bleu1,
                "bleu_2": _bleu2,
                "bert": _bert,
                "judge": _judge,
                "judge_reasoning": _judge_reasoning,
                "latency_ms": latency_ms,
                "method_name": method.name,
                "history_turns": len(history),
                "source_sessions": qa.get("session_id", []),
                "clue_rounds": qa.get("clue", []),
            }
            if current_method_runtime:
                open_result["method_runtime"] = current_method_runtime
            results.append(open_result)
            print(
                f"[OPEN][{i}] em={exact} f1={_f1:.3f} bleu={_bleu:.3f}"
                + (f" bert={_bert:.3f}" if _bert is not None else "")
                + (f" judge={_judge}" if _judge is not None else "")
                + f" latency_ms={latency_ms}"
            )

        if mode in {"mcq", "both"}:
            mcq_question = to_mcq(question)
            if is_agentic:
                pred_mcq = method.answer(dataset, qa, mcq_question)
            else:
                pred_mcq = router.answer(history, mcq_question)
            choice = extract_choice(pred_mcq)
            mcq_result = {
                "idx": i,
                "point": qa.get("point"),
                "mode": "mcq",
                "question": question,
                "gt": gt,
                "pred": pred_mcq,
                "choice": choice,
                "valid_choice": choice in {"A", "B", "C"},
                "method_name": method.name,
                "history_turns": len(history),
                "source_sessions": qa.get("session_id", []),
                "clue_rounds": qa.get("clue", []),
            }
            if current_method_runtime:
                mcq_result["method_runtime"] = current_method_runtime
            results.append(mcq_result)
            print(f"[MCQ][{i}] choice={choice} valid={choice in {'A', 'B', 'C'}}")

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
    print(f"[INFO] Saved run artifacts: {run_dir}")

    output_json = paths.get("output_json")
    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        model_name = cfg.get("model", {}).get("name", "model")
        method_name = cfg.get("method", {}).get("name", "method")
        stem = output_json.stem
        suffix = output_json.suffix or ".json"
        tagged_path = output_json.parent / f"{stem}__{model_name}__{method_name}{suffix}"
        write_json(tagged_path, {k: payload[k] for k in payload if k != "results"})
        print(f"[INFO] Saved output summary: {tagged_path}")

    return payload



def run_modular_benchmark(
    task_config_path: str,
    model_config_path: str,
    method_config_path: str,
    output_root: str = "",
    mode: str = "",
    max_questions: int = 0,
    enable_bert_score: bool = False,
    enable_llm_judge: bool = False,
    judge_config: Optional[Dict[str, Any]] = None,
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
    return run_benchmark(
        cfg,
        config_dir,
        enable_bert_score=enable_bert_score,
        enable_llm_judge=enable_llm_judge,
        judge_config=judge_config,
    )


def run_legacy_benchmark(
    opts: LegacyRunOptions,
    enable_bert_score: bool = False,
    enable_llm_judge: bool = False,
    judge_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    cfg = merge_legacy_config(opts)
    if opts.dialog_json:
        config_dir = Path(opts.dialog_json).parent
    else:
        config_dir = resolve_config_path(opts.config_path).parent
    return run_benchmark(
        cfg,
        config_dir,
        enable_bert_score=enable_bert_score,
        enable_llm_judge=enable_llm_judge,
        judge_config=judge_config,
    )
