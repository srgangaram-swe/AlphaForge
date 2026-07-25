# Reproducibility and experiment provenance

AlphaForge treats a research result as an evidence bundle, not as a notebook
state or an unversioned serialized object. Governed Signal Foundry runs publish
atomically into `runs/signal-foundry/<experiment-id>/` and never overwrite an
existing run.

## Experiment identity

`alphaforge.research.ExperimentManifest` defines schema `1.0.0`. Its
`experiment_id` is the SHA-256 digest of canonical JSON containing:

- full Git commit and dirty-worktree state;
- immutable dataset bundle identity, contract version, license boundary, and
  point-in-time limitations;
- sorted universe and research date bounds;
- feature, label, model, validation, and backtest-cost configuration;
- stable named seed streams; and
- exact runtime, platform, and dependency versions.

Canonical JSON uses sorted keys, compact separators, and ASCII encoding.
Execution timestamps, invocation arguments, result summaries, and artifact
hashes are recorded but do not change the semantic experiment identity. A
different dependency, platform, dataset, code state, or governed setting
therefore receives a different identity.

The outer `run_manifest.json` schema is `2.0.0`. It contains the experiment
manifest plus result pointers such as the selected candidate and trial-ledger
head. Consumers must validate the declared versions rather than infer a schema
from filenames.

## Randomness

One non-negative root seed is expanded into order-independent, named 32-bit
streams for Python, NumPy, scikit-learn, optional model backends, data loading,
and hyperparameter search. Adding or reordering a candidate cannot perturb an
existing stream. Algorithms remain responsible for accepting and using their
assigned stream; the manifest makes omissions reviewable.

## Environment and invocation safety

Environment capture is allowlisted. It records only reproducibility controls
such as thread counts and deterministic-runtime switches; it never copies the
process environment wholesale. CLI options containing credential-like names
(`api-key`, `password`, `secret`, or `token`) retain the option name and replace
the value with `[REDACTED]`.

Do not pass secrets on a command line: process listings and shell history exist
outside the manifest boundary. Market-data credentials belong in Keychain or
another approved secret store, and licensed observations remain local.

## Artifact integrity

Each artifact record contains a run-root-relative path, byte length, and
SHA-256 digest. Absolute paths, parent traversal, duplicate paths, invalid
hashes, and backward execution timestamps fail validation. The manifest itself
is excluded from its artifact inventory to avoid a recursive hash.

## Locked environments

`uv.lock` is the committed universal resolution for Python 3.12–3.14 and all
declared extras. CI pins the `setup-uv` action by full commit SHA and uv itself
to `0.9.7`; the container uses the same uv version and lockfile.

```bash
make install
uv run make check
uv run make demo
```

`make install` refuses to update the lock. Intentionally changing a dependency
requires regenerating `uv.lock`, reviewing the entire resolution diff, and
running the complete validation matrix. Network-requiring market-data tests are
separate from the deterministic offline suite.

## Current boundary

The versioned manifest is integrated into the governed Signal Foundry workflow.
Legacy demo and exploratory entry points have not all migrated to this contract
yet. Their outputs must not be represented as governed or final-holdout
evidence until that migration and independent validation are complete.
