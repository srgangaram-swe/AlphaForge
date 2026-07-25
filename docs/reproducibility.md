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
existing stream. The current execution boundary seeds Python, NumPy, PyTorch,
and TensorFlow when installed and injects the pre-registered root seed into
applicable scikit-learn, LightGBM, and torch model specifications without
overwriting an explicit model seed. Named streams reserved for future data
workers and search systems remain visible in the manifest; those systems must
accept their assigned stream before they may claim deterministic support.

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
to `0.9.7`; the container uses the same uv version and lockfile. CI exercises
the base environment on every supported Python version, the native and torch
boundaries, and separate locked import checks for the data, ML, and application
extras.

```bash
make install
uv run make check
uv run make demo
```

`make install` refuses to update the lock. Intentionally changing a dependency
requires regenerating `uv.lock`, reviewing the entire resolution diff, and
running the complete validation matrix. Network-requiring market-data tests are
separate from the deterministic offline suite.

## Typed configuration boundary

Every supported YAML surface has a dedicated frozen Pydantic schema under
`alphaforge.config`. Unknown keys, untyped model parameters, unsafe configured
paths, invalid ranges, and cross-field contradictions fail before network,
artifact, or model work begins. `make config-check` validates all seven
committed configurations. The generic permissive YAML loader has been retired;
adding a new YAML surface requires a named schema and negative tests.

## Safe tabular artifacts

Pipeline tables use the `1.0.0` `*.table.json` contract in
`alphaforge.research.artifacts`. The envelope contains an explicit format and
schema version around JSON Table Schema data. Readers validate the exact
envelope, field identity, resource bound, path suffix, and symlink boundary
before deserializing. Writers reject arbitrary Python objects and publish by an
atomic same-filesystem replace.

Pickle is not a supported research interchange format: loading it can execute
code and its implicit Python-object contract is unsuitable for durable evidence.
The walk-forward, feature, training, backtest, paper, visualization, and API
paths all consume the versioned non-executable format. Existing local pickle
runs are legacy artifacts and must be regenerated; they are never migrated by
loading and reserializing untrusted pickle bytes.

## Sprint visual evidence

Sprint-close plots are generated through Seaborn's plotting API and theme
system with a colorblind palette; Matplotlib supplies only the non-interactive
rendering, layout, annotation, and export backend. Plots must be regenerated
from machine-readable run evidence, visually inspected, linked from the sprint
report, and labeled honestly as synthetic, historical backtest, paper, or live
evidence. Restricted market observations remain local; only licensed-safe
aggregates, synthetic fixtures, and their reproducible generation instructions
may be committed.
