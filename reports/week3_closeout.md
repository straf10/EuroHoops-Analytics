# Week 3 close-out (2026-09-25, branch `week-3-closeout`)

## Checklist
- [x] 2. GBL Elo tuning grid widened; live parameters frozen for 2026-27
- [ ] 1. GBL live logging ready for Sat 3 Oct
- [ ] 5. EL round-1 rows flagged as not provable
- [ ] 3. GBL box-score gaps classified
- [ ] 7a. EuroLeague shot coordinate system
- [ ] 7b. 2026-27 formats and tiebreak rules

## 2. GBL Elo tuning grid
The old grid's best value (K=40, HCA=130, reversion=0.25) sat on the edge of all three axes.
The GBL backtest now searches K ∈ {10…80} × HCA ∈ {0…260} × reversion ∈ {0…0.75}
(450 combinations; `src/eurohoops/config.py`), on the same splits as before
(warm-up 2018-21, tuning 2022-23, test 2024-25).

Top tuning-set results (log loss):

| K | HCA | reversion | tuning | test |
|---|-----|-----------|--------|------|
| 50 | 170 | 0 | 0.4124 | 0.4851 |
| 50 | 200 | 0 | 0.4127 | 0.4965 |
| 40 | 170 | 0 | 0.4131 | 0.4857 |
| 60 | 170 | 0 | 0.4140 | 0.4851 |
| 50 | 150 | 0 | 0.4144 | 0.4798 |
| **40** | **130** | **0.25** | 0.4313 | **0.4845** |

- The grid best (K=50, HCA=170, reversion=0) lies **inside** the grid on K and HCA. Reversion=0
  (no pull toward the mean) is a natural lower bound, not a grid limit.
- Grid best minus 40/130/0.25, paired bootstrap (1,000 resamples):
  tuning −0.0189 [−0.0323, −0.0050]; **test +0.0005 [−0.0194, +0.0234]**.
- **Decision: keep K=40, HCA=130, reversion=0.25** (the test CI spans 0), and freeze them for
  2026-27 via `Backtest.frozen`. The tuning-set gain doesn't carry over to test, which suggests
  the grid best overfits 2022-23. HCA=170 also implies a 73% home win at equal ratings,
  against 64% observed.
- Check: GBL test log loss 0.4845 (B0 0.6680) reproduces; model versions unchanged for all
  three reports (GBL `0.2.0+df05260c`, EL `0.2.0+425e6393`, EL history `0.2.0+db021063`).
  Each report now records `grid.best`, `grid.best_on_edge` and `grid.best_minus_tuned_log_loss`.
- Note: the EL live grid best also has reversion on the edge (0.25). EL parameters were
  frozen by an earlier decision (D-a), so this is only for information.
