# Repository Guidelines

## Project Structure & Module Organization
- `custom_optimizer.py`: AdamW variant with McGinley Dynamic smoothing.
- `sweep_train.py`: CIFAR‑10 training and W&B sweep runner.
- `data/`: Dataset cache (created by torchvision).
- `runs/`: Outputs per run (logs, checkpoints).
- `wandb/`: Local W&B run metadata.
- Keep new Python modules at repo root or create `src/` if the code grows; prefer one module per concern (optimizers, training, utils).

## Build, Test, and Development Commands
- Create env and install deps:
  - `python -m venv venv && source venv/bin/activate`
  - `pip install -r requirements.txt`
- Configure Weights & Biases:
  - `export WANDB_API_KEY=<your_key>`
- Run the sweep (default grid in file):
  - `python sweep_train.py`
- Quick smoke test (imports + one optimizer step):
  - `python - <<'PY'
import torch; from custom_optimizer import CustomAdamW
w = torch.nn.Parameter(torch.ones(2, requires_grad=True))
opt = CustomAdamW([w], lr=1e-2, dynamic_smoothing=True)
(w.sum()).backward(); opt.step(); print(w)
PY`

## Coding Style & Naming Conventions
- Follow PEP 8: 4‑space indent, 88–100 col soft limit.
- Names: `snake_case` for functions/vars, `PascalCase` for classes, `UPPER_SNAKE` for constants.
- Docstrings: Google‑style (`Args:`, `Returns:`) as used in `custom_optimizer.py`.
- Imports ordered: stdlib → third‑party → local.
- Prefer explicit logging via `logging` and W&B metrics; avoid print in library code.

## Testing Guidelines
- Run tests locally with pytest:
  - `pip install pytest`
  - `PYTHONPATH=. pytest -q`
- For quick functional checks without tests, reduce runtime:
  - Edit `num_epochs` and `count` in `sweep_train.py` (e.g., `num_epochs=1`, `count=1`).
  - Verify outputs in `runs/<run>/log.csv` and W&B charts.

## Commit & Pull Request Guidelines
- Commits: short imperative subject (<= 50 chars), details in body when needed.
- Scope examples: `optimizer: ...`, `train: ...`, `docs: ...`.
- PRs must include:
  - Clear description of changes and rationale.
  - Repro steps or commands (incl. env vars) and expected metrics.
  - Links/screenshots: W&B run URL and/or `runs/*/log.csv` snippet.
  - Note any API changes or new config flags.

## Security & Configuration Tips
- Do not commit secrets; use `WANDB_API_KEY` env var.
- Large artifacts (`runs/`, `wandb/`, datasets) should remain untracked; keep `.gitignore` updated.
- Use `requirements-pinned.txt` (Torch 2.7.x) for reproducible installs when sharing results.
  - If you already have torchaudio installed, align versions (e.g., torch/vision/audio 2.7.1/0.22.1/2.7.1) or uninstall torchaudio.
