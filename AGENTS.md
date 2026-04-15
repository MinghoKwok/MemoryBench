# AGENTS.md

## Scope
This file defines repo-wide instructions for Codex agents.

More specific `AGENTS.md` files in subdirectories override or extend these instructions for their local subtree. For work inside `Benchmark_Pipeline/`, follow `Benchmark_Pipeline/AGENTS.md` as the primary implementation spec.

## API Keys
All API keys are stored in Benchmark_Pipeline/.env.local

## Enviroment
Use the conda env memorybench to run all scripts.

## General Working Style
- Prefer minimal, targeted changes over broad rewrites
- Read the relevant local code before editing
- Preserve existing project structure unless a refactor is necessary
- Keep code readable, modular, and easy to review
- Do not introduce hidden behavior or silent fallbacks
- Do not change models, prompts, or benchmark behavior unless explicitly required by local instructions

## Verification
- Verify changes with the smallest reliable check available
- Prefer code-path verification over assumptions
- When modifying evaluation logic, inspect for leakage, hidden shortcuts, and modality mixing

## Secrets and Safety
- Never hardcode secrets into source files, configs, notebooks, logs, or outputs
- Read secrets from environment variables only
- If a required secret or dependency is unavailable, report it clearly instead of silently working around it

## Outputs
- Keep run artifacts, logs, and generated outputs out of source directories unless the local project conventions explicitly require otherwise
- Prefer reproducible outputs and explicit logging over ad hoc prints

## Notes for This Repo
- For benchmark work, do not rely on general assumptions about dataset format when local files can be inspected directly
- Reuse existing code when practical instead of duplicating logic