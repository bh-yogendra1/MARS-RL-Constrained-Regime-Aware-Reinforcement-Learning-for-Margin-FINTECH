# Starter prompt for Gemini Antigravity (copy-paste)

You are continuing a research project, MARS-RL (margin-trading deep RL for portfolio management), built on Gu et al., IEEE TKDE 2026 (DOI 10.1109/TKDE.2026.3701681). Read `handoff/PROJECT_CONTEXT.xml`, `handoff/README_HANDOFF.md`, `handoff/docs/ARCHITECTURE.md`, `handoff/docs/EXPERIMENT_LOG.md`, `handoff/docs/NEXT_STEPS_NEW_DATASET.md`, then the code in `src/`.

Goal: improve the model's performance (annual return, Sharpe, drawdown, Calmar) using the new large dataset in `data/mega/` so that it can honestly beat the base paper's Table I (best Sharpe 1.05–1.16, annual return 19–31% on the 2020-05→2023-05 test window).

Hard rules: select designs on validation only; run the test window once per final configuration; log every variant; never claim a win the locked test does not show; no look-ahead (macro/fundamentals release-lagged); keep Track A and Track B separate; run `python -m pytest tests -q` before and after changes; make long jobs resumable.

First tasks: (1) inspect `data/mega/reports/data_quality.json` and the data; (2) alpha go/no-go with `src/alpha.py` (validation rank-IC); (3) report findings to the user (who writes in Hinglish) before spending hours of compute. The user prefers plain tables comparing the base paper vs our model, and no extra generated files until results are final.
