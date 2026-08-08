# Signal Foundry Sprint 5 — monitoring, paper trading, and live readiness

**Outcome: the plumbing between a backtest and a broker now exists, and the gate
at the end of it is closed with all seventeen items unmet.**

![Sprint 5 close-out](evidence/signal_foundry_sprint_5/closeout/sprint_5_closeout.png)

*Every panel is computed from a committed record — raw nanosecond samples and a
Git-derived delivery ledger — under [ADR 0021](adr/0021-reproducible-sprint-evidence.md).
Four panels, three of them unflattering by design: the readiness gate with every
item unmet, the capital cap sitting at zero against its ceiling, and the measured
crossover showing this repository's own sweep falls below the point where
distributing it helps. Synthetic and simulated evidence; no market data, no broker
connection, no capital at risk. Regenerate with
`python scripts/publish_sprint_5_evidence.py`.*

---

## What shipped

| MR | Issue | Slice | Tests |
| --- | --- | --- | --- |
| SF-S5-MR2 | [#99](https://github.com/srgangaram-swe/AlphaForge/issues/99) | Broker connectivity requirements and selection (docs only) | — |
| SF-S5-MR3 | [#45](https://github.com/srgangaram-swe/AlphaForge/issues/45) | Broker-neutral contract and deny-by-default paper adapter | 87 |
| SF-S5-MR4 | [#46](https://github.com/srgangaram-swe/AlphaForge/issues/46) | Durable session state and halt-on-divergence reconciliation | 47 |
| SF-S5-MR8 | [#47](https://github.com/srgangaram-swe/AlphaForge/issues/47) | Bounded distributed research execution | 43 |
| SF-S5-MR9 | [#48](https://github.com/srgangaram-swe/AlphaForge/issues/48) | Checkpoint bindings and hard budgets | 51 |
| SF-S5-MR10 | [#49](https://github.com/srgangaram-swe/AlphaForge/issues/49) | Live-readiness gate and inert capital configuration | 46 |
| SF-S5-MR11 | [#112](https://github.com/srgangaram-swe/AlphaForge/issues/112) | Reproducible, content-addressed close-out evidence | — |

**274 test functions added across the merged slices** (they expand to more
collected cases under parametrization — the ledger counts functions and says so).
ADRs 0015–0021.

## Three findings worth keeping

**A 97% parallel workload can still lose to a `for` loop.** MR8 profiled the
robustness sweep at a 0.9727 parallel fraction — an Amdahl ceiling of 36.6×. That
number alone justifies distributing. Measuring the *fixed cost* of distributing
told the opposite story: below roughly 2 ms per task, a process pool is up to
fifty times slower than serial, and this repository's sweep runs at 0.49 ms per
point. Dask is selected and justified in ADR 0018, and its adoption is gated on a
measured per-task threshold rather than taken now. Distributing without the second
measurement would have been a large regression shipped as an improvement.

**Idempotency that lives in a process does not survive a restart.** MR3 made a
replayed decision cycle a no-op by remembering client order IDs in a dictionary.
MR4 found the hole: a restart empties it, and the broker's own dedupe window is far
shorter than the gap between an evening crash and a morning restart. Intent is now
persisted *before* the broker is contacted, because the window between "decided"
and "acknowledged" is exactly where duplicates are born.

**Repairing a divergence destroys the evidence needed to explain it.** When the
local book says 100 shares and the broker says 150, the cause is an unrecorded
fill, a duplicate submission, a manual intervention, or a bug — four causes with
four different correct responses. Reconciliation therefore halts and never
repairs, and never liquidates: an automatic flatten is a market order sized from
exactly the position you are unsure about.

## Where this leaves live trading

The [live-readiness gate](live_readiness.md) is **`NOT_READY`, 17 of 17 items
unmet**:

| Category | Unmet |
| --- | --- |
| Operational | 5 — rehearsal, broker-failure drill, kill switch, deactivation, audit/tax export |
| Evidence | 4 — qualified candidate, paper duration, paper stability, cost validation |
| Security | 3 — capital cap, risk limits, credential custody |
| Reconciliation | 2 — clean history, current broker state |
| Policy | 2 — employment policy, owner approval |
| Legal | 1 — legal and regulatory review |

The first evidence item is blocked by research rather than engineering: Sprint 4's
qualification verdict was `REJECTED` with 7 of 8 criteria failing. No amount of
infrastructure produces a qualified strategy.

**Capital at risk: $0.** The configuration is inert by default, and there is no
method anywhere capable of raising a cap.

## Honest limitations

- **The readiness framework cannot verify attestations.** Employment-policy and
  legal review are human judgements; a false attestation produces a READY verdict.
  The framework narrows the hole to a named person on a dated record with a
  180-day expiry, and prints "Recorded, not verified" on the record. It cannot
  close it.
- **No broker connection exists.** MR3's adapter is an in-process simulation with
  no network client, asserted by parsing its imports. Paper fills carry no queue
  position, no contention, and no borrow scarcity; they bound *operational*
  readiness only.
- **Two MR8 acceptance criteria were not claimed**: network-partition testing
  (there is no network) and GPU scheduling (declared and validated, honoured by no
  backend). Both were re-scoped in writing on [#47](https://github.com/srgangaram-swe/AlphaForge/issues/47)
  rather than marked done.
- **Signalattice's Sprint 5 track is separate and still open** — eight issues
  covering the FastAPI service, run registry, TypeScript console, telemetry,
  shadow forecasting, champion-challenger governance, signed release, and
  benchmark dossier. This report covers AlphaForge only.
- **All measurements are single-machine wall clock** on macOS/arm64 and labelled
  as such.

## What would change the verdict

In dependency order: a qualified candidate (research, not engineering), then a
paper-trading period long enough to produce stability and cost-validation
evidence, then the operational drills, then the two human attestations. The gate
is deliberately ordered so the first item cannot be satisfied by effort.


## Correction (SF-S5-MR11)

The first version of this report and its figure carried two defects, both filed
as [#112](https://github.com/srgangaram-swe/AlphaForge/issues/112) and corrected
before Sprint 5 was re-promoted:

- **A false provenance claim.** The generator's docstring asserted every panel was
  computed from source modules while two panels used hand-typed constants. The
  claim was repeated in a PR body, this report, and an issue-closing comment.
  Panels now read committed records, and a test parses the generator's AST to fail
  on module-level numeric literals.
- **A conflated count.** "330 tests" was pytest's *collected case* count labelled
  as tests added; parametrized functions expand. The Git-derived ledger reports
  **274 test functions** and the axis states exactly what is counted.

The crossover numbers were also re-measured with warmups and seven repetitions,
and [ADR 0018](adr/0018-bounded-distributed-research-execution.md) is annotated
rather than rewritten. Its decision is unchanged.
