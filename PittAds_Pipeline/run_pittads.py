import argparse
from benchmark import LegacyRunOptions, run_legacy_benchmark


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run PittAds/Mem-Gallery style benchmark.")
    p.add_argument("--config", type=str, default="config/default.yaml")
    p.add_argument("--dialog-json", type=str, default="")
    p.add_argument("--image-root", type=str, default="")
    p.add_argument("--model-path", type=str, default="")
    p.add_argument("--max-new-tokens", type=int, default=0)
    p.add_argument("--output-json", type=str, default="")
    p.add_argument("--mode", type=str, choices=["open", "mcq", "both"], default="")
    p.add_argument("--max-questions", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_legacy_benchmark(
        LegacyRunOptions(
            config_path=args.config,
            dialog_json=args.dialog_json,
            image_root=args.image_root,
            model_path=args.model_path,
            max_new_tokens=args.max_new_tokens,
            output_json=args.output_json,
            mode=args.mode,
            max_questions=args.max_questions,
        )
    )


if __name__ == "__main__":
    main()
