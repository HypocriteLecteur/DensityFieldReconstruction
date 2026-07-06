# Contributing to DFR

DFR contains research code whose numerical behavior matters more than cosmetic
uniformity. Refactor in small, reviewable slices and protect existing behavior
with tests before moving it.

## Environment

- Use Python 3.12 or 3.13.
- Install a PyTorch build appropriate for the local CUDA setup before building
  the custom rasterizers.
- Install the editable package and tests with
  `python -m pip install -e ".[test]"`.
- Work from the repository root while legacy experiment paths are being
  migrated.

## Verification tiers

Run the cheapest relevant tier after every change:

1. Syntax and whitespace:

   ```powershell
   python -m compileall -q dfr experiments tests
   git diff --check
   ```

2. CPU characterization tests:

   ```powershell
   python -m pytest -m "not cuda"
   ```

3. CUDA extension smoke tests, when the rasterizers are installed:

   ```powershell
   python -m pytest -m cuda
   ```

4. For scientific changes, run the smallest representative analysis or
   reconstruction comparison and record the command, hardware, seed, inputs,
   and tolerances in `TODO.md`.

Never report an unavailable CUDA test as passing. A clean skip and its reason
are part of the handoff.

## Code boundaries

- Reusable loading, analysis, camera, reconstruction, evaluation, plotting, and
  artifact behavior belongs under `dfr/`.
- Files under `experiments/` should become thin configuration/CLI entry points.
- Core `dfr` modules must not import from `experiments`.
- Computation should return typed data; plotting and persistence should be
  separate, explicit operations.
- Preserve compatibility adapters until active callers have migrated and
  characterization tests cover the replacement.

No repository-wide formatter is configured yet. Match the surrounding style,
keep public docstrings accurate, and avoid unrelated formatting churn.

## Data and artifacts

- Treat `dataset/` and scenario source data as read-only inputs.
- Put new generated work under `outputs/<workflow>/<run-id>/`.
- Do not add new producers for `figs/`, `results/`, scenario log directories,
  or root-level result files.
- Do not commit large datasets, caches, checkpoints, images, or videos.
- Small deterministic fixtures belong in `tests/fixtures/`.
- Library functions must not write unless the caller supplies an explicit
  output configuration/path.

## Updating the refactor handoff

`TODO.md` is part of the implementation, not optional project prose. Before
ending a refactor session:

1. Check off only completed items and update the current phase.
2. Record new architectural decisions and their reasons.
3. Add a newest-first handoff entry with files/commits, test results, known
   failures, and the exact next task.
4. Keep local commits narrowly scoped. Version history remains local unless the
   owner explicitly changes that policy.
