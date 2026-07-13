# DFR Documentation

This directory holds lightweight, hand-maintained docs for the refactored
package API. The root `README.md` remains the broad project overview; these
files are narrower handoff guides for people writing or maintaining code.

| Document | Use it when |
|---|---|
| [`WORKFLOW.md`](WORKFLOW.md) | You want runnable examples for `load_dataset -> analyze -> reconstruct -> evaluate -> plot`. |
| [`MODULE_OWNERSHIP.md`](MODULE_OWNERSHIP.md) | You need to decide where a new helper, script, or output-writing behavior belongs. |
| [`COMMAND_VERIFICATION.md`](COMMAND_VERIFICATION.md) | You need to know which README/docs commands were actually run and which remain CUDA/data-dependent. |
| [`PHASE8_COMPATIBILITY_INVENTORY.md`](PHASE8_COMPATIBILITY_INVENTORY.md) | You are starting cleanup and need the current compatibility-wrapper and legacy-output inventory. |
| [`PHASE8_ARCHIVE_POLICY.md`](PHASE8_ARCHIVE_POLICY.md) | You are deciding whether a legacy wrapper, copied source tree, or generated-output directory may be removed. |
| [`RELEASE_NOTES_v0.2.0.md`](RELEASE_NOTES_v0.2.0.md) | You are migrating a local `v0.1.0` workflow to the refactored release. |

Keep these docs in sync with `TODO.md` whenever a phase changes public API
ownership or output policy.
