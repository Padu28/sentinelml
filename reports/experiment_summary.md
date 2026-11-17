# SentinelML — Experiment Summary

*Generated 2026-11-17T13:14:12+00:00 · 5 seeds per condition · values are mean ± 95% CI*

> Data: synthetic controlled stream with known drift ground truth. Results characterise
> detector/policy behaviour under controlled conditions and are **not** claims about any real system.

| Drift | Policy | Post-drift AUC | Mean AUC | Retrains | Detection delay (records) | False alarms | Recovery (batches) |
|---|---|---|---|---|---|---|---|
| sudden | static | 0.709 ± 0.012 | 0.780 ± 0.008 | 0.0 ± 0.0 | 0 ± 0 | 0.0 | — |
| sudden | periodic | 0.901 ± 0.005 | 0.908 ± 0.004 | 6.0 ± 0.0 | 0 ± 0 | 0.0 | 7.4 ± 0.7 |
| sudden | drift_aware | 0.945 ± 0.003 | 0.937 ± 0.002 | 1.0 ± 0.0 | 0 ± 0 | 0.0 | 3.2 ± 0.6 |
| gradual | static | 0.727 ± 0.013 | 0.794 ± 0.008 | 0.0 ± 0.0 | 450 ± 139 | 0.0 | — |
| gradual | periodic | 0.904 ± 0.004 | 0.911 ± 0.003 | 6.0 ± 0.0 | 600 ± 170 | 0.0 | 8.8 ± 2.2 |
| gradual | drift_aware | 0.933 ± 0.005 | 0.931 ± 0.003 | 6.0 ± 0.0 | 450 ± 139 | 0.0 | 6.6 ± 0.7 |
| incremental | static | 0.737 ± 0.011 | 0.799 ± 0.007 | 0.0 ± 0.0 | 400 ± 170 | 0.0 | — |
| incremental | periodic | 0.913 ± 0.005 | 0.916 ± 0.003 | 6.0 ± 0.0 | 450 ± 139 | 0.0 | 8.2 ± 0.6 |
| incremental | drift_aware | 0.942 ± 0.005 | 0.935 ± 0.003 | 4.4 ± 2.7 | 400 ± 170 | 0.0 | 5.0 ± 0.0 |
| recurring | static | 0.815 ± 0.005 | 0.851 ± 0.003 | 0.0 ± 0.0 | 0 ± 0 | 0.0 | 8.0 ± 0.0 |
| recurring | periodic | 0.743 ± 0.004 | 0.802 ± 0.003 | 6.0 ± 0.0 | 0 ± 0 | 0.0 | — |
| recurring | drift_aware | 0.819 ± 0.008 | 0.853 ± 0.006 | 6.0 ± 0.0 | 0 ± 0 | 0.0 | 3.2 ± 0.6 |

## How to read this
- **H1** (faster recovery): compare *Post-drift AUC* and *Recovery* of `drift_aware` vs `static`.
- **H2** (fewer retrains): compare *Retrains* of `drift_aware` vs `periodic` at similar AUC.
- **H3** (quality ≠ drift): quality-faulted batches are gated and do not count as drift alarms.

## Limitations
- Single model family (configurable in `configs/model.yaml`); thresholds were set a-priori, not tuned on test streams.
- Labels are assumed to arrive one batch after prediction; real label delay is usually longer.
- Synthetic drift is cleaner than real-world drift; validate on a real dataset (NAB, UCI Electricity) before drawing conclusions.
