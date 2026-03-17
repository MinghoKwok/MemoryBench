import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Benchmark_Pipeline.benchmark import run_modular_benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run modular multimodal memory benchmark experiments.")
    parser.add_argument("--task-config", type=str, default="config/tasks/brand_memory_test.yaml")
    parser.add_argument("--model-config", type=str, default="config/models/qwen_local_default.yaml")
    parser.add_argument("--method-config", type=str, default="config/methods/full_context.yaml")
    parser.add_argument("--output-root", type=str, default="runs")
    parser.add_argument("--mode", type=str, choices=["open", "mcq", "both"], default="")
    parser.add_argument("--max-questions", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_modular_benchmark(
        task_config_path=args.task_config,
        model_config_path=args.model_config,
        method_config_path=args.method_config,
        output_root=args.output_root,
        mode=args.mode,
        max_questions=args.max_questions,
    )


if __name__ == "__main__":
    main()
