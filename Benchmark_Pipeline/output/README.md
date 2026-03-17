# Output Layout

This directory stores compatibility outputs and lightweight summaries.

## Conventions

- `output/<task>/`
  - Legacy-compatible JSON outputs grouped by task.
  - Filenames are typically:
    - `results_<task>.json`
    - `results_<task>__<model>__<method>.json`

- `runs/<task>/`
  - Canonical benchmark artifacts for each run.
  - Each run gets its own timestamped directory with:
    - `config.json`
    - `metrics.json`
    - `predictions.jsonl`

## Recommended Usage

- Use `runs/` for experiment tracking and comparisons.
- Treat `output/<task>/` as a convenience layer for quick inspection and backward compatibility.
- Avoid relying on a single flat `output/` filename, since multiple models and methods are expected.
