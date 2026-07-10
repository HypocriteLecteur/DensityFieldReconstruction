# DFR Documentation

This directory holds lightweight, hand-maintained docs for the refactored
package API. The root `README.md` remains the broad project overview; these
files are narrower handoff guides for people writing or maintaining code.

| Document | Use it when |
|---|---|
| [`WORKFLOW.md`](WORKFLOW.md) | You want runnable examples for `load_dataset -> analyze -> reconstruct -> evaluate -> plot`. |
| [`MODULE_OWNERSHIP.md`](MODULE_OWNERSHIP.md) | You need to decide where a new helper, script, or output-writing behavior belongs. |
| [`COMMAND_VERIFICATION.md`](COMMAND_VERIFICATION.md) | You need to know which README/docs commands were actually run and which remain CUDA/data-dependent. |

Keep these docs in sync with `TODO.md` whenever a phase changes public API
ownership or output policy.
