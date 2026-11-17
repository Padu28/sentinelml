# SentinelML — Architecture

```mermaid
flowchart LR
    subgraph Ingestion
        A[CSV / API / replay] --> B[StreamReplayer<br/>timestamp order]
    end
    B --> C{Quality engine<br/>schema · nulls · ranges<br/>dups · categories · outliers}
    C -- score < 80 --> Q[(quality event)]
    C --> D[clean_batch]
    D --> E[Champion model<br/>sklearn Pipeline]
    E --> F[Predictions]
    F -- labels arrive --> G[Performance monitor<br/>rolling AUC vs baseline]
    D --> H[Drift detectors<br/>PSI + KS · Page-Hinkley · ADWIN]
    G --> P{Retraining policy}
    H --> P
    P -- static --> X[no-op]
    P -- periodic --> R1[retrain on sliding window]
    P -- drift-aware --> R2[train candidate] --> V{Validation gate<br/>Δ AUC ≥ 0.01<br/>p95 ≤ 50 ms}
    V -- pass --> REG[(Model registry<br/>champion / archived)]
    V -- fail --> REJ[(rejected)]
    REG --> E
    subgraph Storage[SQLite]
        DB[(runs · batch_metrics · events<br/>model_versions · predictions)]
    end
    Q --> DB
    G --> DB
    H --> DB
    REG --> DB
    DB --> API[FastAPI]
    DB --> UI[Streamlit dashboard]
```

## Design decisions
| Decision | Rationale |
|---|---|
| PSI **and** KS must both fire per feature | KS alone over-fires on large batches; PSI alone ignores sample size. |
| ≥ 2 drifted features to raise an event | Reduces single-feature noise alarms (configurable). |
| Quality-faulted batches cannot trigger drift retraining | Separates preventable input problems from genuine drift (H3). |
| Candidate validated on the *latest* batch, held out from its training | Fair champion-vs-candidate comparison on unseen recent data. |
| Logistic regression default | Best validation AUC on this task (0.937 vs 0.915 RF / 0.910 HGB), ~8× faster than RF, exact additive explanations. |
| SQLite prototype | Zero-ops; the `Database` class isolates SQL so PostgreSQL is a DSN change. |
| No external drift library | Transparent, tested implementations; Evidently/River can be plugged in. |

## Module map
| Package | Responsibility |
|---|---|
| `ingestion/` | synthetic generator with ground truth, CSV loader, stream replayer |
| `validation/` | quality checks, scoring, batch cleaning |
| `features/` | leakage-safe `ColumnTransformer` |
| `models/` | training, metrics, latency profiling |
| `monitoring/` | PSI, KS, Page-Hinkley, ADWIN, rolling performance |
| `retraining/` | policies, simulation loop, champion/candidate registry |
| `explainability/` | permutation importance, SHAP / linear / substitution local explanations |
| `storage/` | SQLite persistence |
| `experiments/` | multi-seed runner, aggregation with 95 % CIs, report + figures |
