from __future__ import annotations

import argparse
from pathlib import Path

from ..common import load_yaml, resolve_config_path
from ..dataset import MemoryBenchmarkDataset
from ..methods import get_method


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect MIRIX retrieval/prompt output for one QA.")
    parser.add_argument("--task-config", required=True)
    parser.add_argument("--method-config", required=True)
    parser.add_argument("--qa-index", type=int, default=0, help="0-based QA index")
    args = parser.parse_args()

    task_cfg = load_yaml(resolve_config_path(args.task_config))
    method_cfg = load_yaml(resolve_config_path(args.method_config))

    root = Path(__file__).resolve().parents[2]
    dataset_cfg = task_cfg["dataset"]
    dialog_json = (root / dataset_cfg["dialog_json"]).resolve()
    image_root = (root / dataset_cfg["image_root"]).resolve()
    dataset = MemoryBenchmarkDataset(dialog_json, image_root)
    qa = dataset.qas[args.qa_index]

    method = get_method(method_cfg["name"], config=dict(method_cfg))
    history = method.build_history(dataset, qa)
    prompt = method.get_system_prompt("", dataset, qa)

    print(f"method={method.name}")
    print(f"question={qa.get('question', '')}")
    print(f"history_turns={len(history)}")
    print(f"prompt_chars={len(prompt)}")
    print(f"runtime_info={method.runtime_info}")
    print("--- prompt preview ---")
    print(prompt[:4000])
    for turn in history:
        if turn.get("images"):
            print("--- retrieved images ---")
            for image in turn["images"]:
                print(image)


if __name__ == "__main__":
    main()
