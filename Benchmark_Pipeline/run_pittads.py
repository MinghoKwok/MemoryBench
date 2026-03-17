import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Benchmark_Pipeline.run_legacy_benchmark import main


if __name__ == "__main__":
    main()
