# Governance Benchmark: Veritas vs Naive Baselines

_Seed 20261012. Deterministic — rerun reproduces these numbers. Total runtime 0.7s. No network, no API calls._

## Results

| Scenario | Veritas | Latest-wins | Append-only | Unit |
|---|---|---|---|---|
| Contradiction storm: 50 facts x 4 writes, truth = 0.9-src value | 100.0% | 24.0% | 34.0% | % facts correct (higher better) |
| Cascade integrity: 20 chains x depth 50, retract roots | 0.0% | 100.0% | 100.0% | % orphans remaining (lower better) |
| Temporal accuracy: 30 facts x 2 windows, 10 time queries | 100.0% | n/a | n/a | % queries correct (higher better) |
| Deletion compliance: forget src_b + 1 proposition (recall/precision) | R:100.0% / P:100.0% (orphans:0) | R:63.0% / P:100.0% (orphans:20) | R:63.0% / P:100.0% (orphans:20) | recall / precision % (higher better) |
| Zombie resistance: P killed by Q, P re-asserted 5x | 100.0% | 0.0% | 0.0% | % pairs P stayed dead (higher better) |
| Audit completeness: why-retracted? state-at-T? | 100.0% | 0.0% | 0.0% | % questions answerable (higher better) |

## Reading the table

- **Contradiction storm**: high-reliability facts under attack by weaker sources. Veritas adjudicates by entrenchment; naive stores keep whoever wrote last.
- **Cascade integrity**: retracting a root must retract its derivations (TMS). Naive stores have no derivation tracking, so dependents are orphaned — measured honestly, not assumed.
- **Temporal accuracy**: naive stores have no validity-window concept, hence n/a (not a fake zero).
- **Deletion compliance**: recall = targets actually gone, precision = non-targets preserved. Veritas cascades deletion to derived beliefs.
- **Zombie resistance**: a killed belief must stay dead under re-assertion pressure.
- **Audit completeness**: can the system answer _why_ something was retracted and _what_ the state was at time T?

## Reproduce

```
python3 benchmarks/governance_bench/run.py
```
