"""
Preprocess dialog data into MIRIX-style structured memories.

Usage:
    python -m Benchmark_Pipeline.benchmark.mirix.preprocess \
        --task-config config/tasks/brand_memory_test.yaml \
        --model-config config/models/gpt_4_1_nano.yaml \
        --mode incremental_llm

    python -m Benchmark_Pipeline.benchmark.mirix.preprocess --all --mode incremental_llm
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from ..common import SCRIPT_DIR, load_yaml, resolve_config_path
from ..dataset import MemoryBenchmarkDataset
from ..runner import resolve_runtime_paths
from .memory import extract_memories_incremental_llm

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract MIRIX memories from dialog data.",
    )
    parser.add_argument(
        "--task-config",
        type=str,
        default="config/tasks/brand_memory_test.yaml",
        help="Path to task config YAML.",
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default="",
        help="Path to model config YAML (required for MIRIX preprocessing).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Output directory for memory cache. Default: data/mirix_cache/",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["incremental_llm"],
        default="incremental_llm",
        help="Extraction mode for the active MIRIX implementation.",
    )
    parser.add_argument("--history-size", type=int, default=2, help="History size for incremental_llm mode.")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all task configs in config/tasks/.",
    )
    return parser.parse_args()


def process_task(
    task_config_path: str,
    mode: str,
    model_config_path: str,
    output_dir: str,
    history_size: int = 2,
) -> None:
    task_cfg = load_yaml(resolve_config_path(task_config_path))
    cfg = {"dataset": task_cfg.get("dataset", {}), "eval": {}, "run": {}}
    paths = resolve_runtime_paths(cfg, SCRIPT_DIR)

    dialog_json = paths["dialog_json"]
    image_root = paths.get("image_root")
    dataset = MemoryBenchmarkDataset(dialog_json, image_root)

    scenario_name = dialog_json.stem
    if output_dir:
        out_path = Path(output_dir)
    else:
        out_path = dialog_json.parent.parent / "mirix_cache"
    cache_file = out_path / f"{scenario_name}_mirix_memories.json"

    logger.info("Processing: %s", scenario_name)
    logger.info("  Dialog: %s", dialog_json)
    logger.info("  Mode: %s", mode)
    logger.info("  Output: %s", cache_file)

    if not model_config_path:
        logger.error("--model-config required for MIRIX preprocessing")
        return
    model_cfg = load_yaml(resolve_config_path(model_config_path))
    llm_config = {
        "model": model_cfg.get("model", "gpt-4.1-mini"),
        "api_key": model_cfg.get("api_key", ""),
        "api_key_env": model_cfg.get("api_key_env", "OPENAI_API_KEY"),
        "base_url": model_cfg.get("base_url", "https://api.openai.com/v1"),
        "timeout": model_cfg.get("timeout", 120),
        "history_size": history_size,
    }
    store = extract_memories_incremental_llm(dataset, llm_config)

    store.save(cache_file)
    logger.info("  Saved %d memories", len(store.memories))


def main() -> None:
    args = parse_args()

    if args.all:
        tasks_dir = SCRIPT_DIR / "config" / "tasks"
        task_files = sorted(tasks_dir.glob("*.yaml"))
        if not task_files:
            logger.warning("No task configs found in %s", tasks_dir)
            return
        for task_file in task_files:
            try:
                process_task(
                    str(task_file),
                    args.mode,
                    args.model_config,
                    args.output_dir,
                    history_size=args.history_size,
                )
            except Exception:
                logger.error("FAILED processing %s", task_file, exc_info=True)
    else:
        process_task(
            args.task_config,
            args.mode,
            args.model_config,
            args.output_dir,
            history_size=args.history_size,
        )


if __name__ == "__main__":
    main()
