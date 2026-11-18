# Addendum: v2 design iteration (turnover-controlled residual policies)

**Motivation.** v1 failed mainly through high turnover (0.3–0.8/day) and weights drifting far from equal-weight.
**v2 changes:** long-only tilt around equal weight (`mode="resid"`, κ=1.5), execution smoothing (0.3), turnover penalty in the reward, optional conformal guard / learned leverage (L_max=2).
**Variants tried (6, all reported):** v2_a..v2_f in `src/config.py`; 3 seeds × 150k steps each.
**Selection rule (fixed before looking at v2 test numbers):** highest mean validation Sharpe → **v2_b** (val Sharpe 0.98 vs 0.53 for equal-weight; per-seed 1.46/0.86/0.61 — high seed variance).
**Test (single run, 1× costs, 3 seeds):** v2_b: ann. return 9.3%, Sharpe 0.68, MDD −19.5%, Calmar 0.47; Sharpe stays positive at 2×/3× costs (0.55 / 0.41). Versus v1 MARS-RL (Sharpe 0.10) this is a large gain; versus equal-weight (0.85) and the base paper (1.05–1.16) it is still lower.
**Post-hoc observation (NOT a claim):** v2_c / v2_d (no guard) had higher test Sharpe (0.89 / 0.87; MDD −32% / −29%) but were not chosen by the validation rule; picking them now would be selection on test.
**Disclosure:** the test window was already seen once in the v1 phase; 6 variants were compared; n_trials = 6 for any deflated-Sharpe discussion.
