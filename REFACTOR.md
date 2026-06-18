# Refactor Plan

## Baseline

- The current state was committed as `827bab0 chore: snapshot current state before refactor`.
- Tests were not run during the pre-refactor assessment because the rasterizer test requires CUDA and installed CUDA extensions.

## Goals

- Make the project importable and testable from a stable package entrypoint.
- Separate reconstruction orchestration, rendering, camera geometry, training, experiments, and evaluation.
- Reduce duplicated experiment code while preserving scientific behavior.
- Make future refactors verifiable with focused tests.

## Refactor Roadmap

1. Stabilize baseline.
   - Add package metadata and a package entrypoint.
   - Remove root-working-directory import hacks from active scripts.
   - Fix tests so importing a test module does not execute GPU rendering or open plots.
   - Mark CUDA-only tests clearly.
   - Verification: import `dfr` from outside the repo root and run non-CUDA smoke tests.

2. Introduce typed configs and results.
   - Replace loose training and reconstruction dictionaries with typed config objects.
   - Define explicit frame input and frame result structures.
   - Keep existing parameter names where practical to reduce migration risk.
   - Verification: existing reconstruction callers can be migrated without changing behavior.

3. Split pipeline services.
   - Break `DensityReconstructor` into center estimation, scale selection, scale-space generation, GMM initialization, and training coordination.
   - Remove dependencies from core `dfr` modules to `experiments`.
   - Keep `DensityReconstructor.process_frame()` as a compatibility wrapper until scripts are migrated.
   - Verification: one-frame reconstruction produces compatible scale, GMM shape, and loss outputs before and after the split.

4. Untangle rendering and rasterizer selection.
   - Keep camera geometry separate from renderer implementations.
   - Move CuPy circle rendering and Gaussian rasterizer calls behind explicit renderer classes or functions.
   - Make device selection explicit instead of relying on hardcoded `cuda`.
   - Provide one canonical place to choose between small, large, and decoupled rasterizer variants.
   - Verification: projection-only rendering works without CUDA; CUDA renderers fail with clear messages when dependencies are unavailable.

5. Decompose `GaussianModel`.
   - Separate model parameters from optimizer setup, learning-rate scheduling, regularization, pruning, splitting, logging, and checkpoint serialization.
   - Resolve the currently referenced but undefined decoupled model path.
   - Make pruning and splitting policies configurable and testable.
   - Verification: checkpoint load/save round trips and optimizer state migration are covered by focused tests.

6. Consolidate experiment runners.
   - Extract shared scenario loading, camera setup, reconstruction loop, metrics, logging, and output writing from duplicated experiment scripts.
   - Leave table, flock, UE4, and angle-sweep scripts as thin parameter/config entrypoints.
   - Verification: one representative scenario runner and one metrics-only runner produce the same output schema as before.

7. Repository hygiene.
   - Keep legacy snapshots, generated outputs, caches, and build artifacts out of the active refactor path.
   - Review whether local helper files such as `draft.py`, copied plotting scripts, and tool-local settings should remain tracked.
   - Clarify the canonical rasterizer source location and ignore policy for build/vendor artifacts.
   - Verification: `git status --ignored` shows generated rasterizer outputs ignored and source files tracked intentionally.

## Success Criteria

- Imports work without `sys.path.append(os.getcwd())`.
- Tests do not execute work at import time.
- Core reconstruction flow remains behavior-compatible.
- Experiment scripts become thin parameter/config entrypoints.
- CUDA-only behavior is isolated and clearly marked.

## Risks And Constraints

- CUDA rasterizer is required for full verification.
- Root `density_field_rasterizer/` has tracked source files and ignored build/vendor outputs.
- Legacy directories should not be refactored unless explicitly brought back into scope.

## Coding Guidance

减少常见LLM编码错误的行为准则。可根据需要与项目特定指令合并。

**权衡：**这些准则倾向于谨慎而非速度。对于简单任务，自行判断即可。

## 1. 先思考再写代码

**不要假设。不要隐藏困惑。把权衡摆到台面上。**

在动手实现之前：

- 明确说出你的假设。不确定就问。
- 如果存在多种理解方式，全部列出来——不要默默选一个。
- 如果有更简单的方案，说出来。该反驳就反驳。
- 如果有什么不清楚的，停下来。说明哪里让你困惑。提问。

## 2. 简洁优先

**用最少的代码解决问题。不写投机性代码。**

- 不加超出需求的功能。
- 一次性代码不搞抽象。
- 没人要求的"灵活性"和"可配置性"不要加。
- 不要为不可能出现的场景写错误处理。
- 如果你写了200行但50行就能搞定，重写。问自己一句："一个资深工程师会说这写复杂了吗？"如果是，简化。

## 3. 精准修改

**只动必须动的地方。只清理自己制造的问题。**

编辑已有代码时：

- 不要顺手"改进"旁边的代码、注释或格式。
- 没坏的东西不要重构。
- 匹配现有风格，即使你会用不同的写法。
- 如果注意到不相关的死代码，提一嘴就好——别删。

当你的修改产生了孤立代码时：

- 移除因你的改动而变成未使用的import、变量和函数。
- 不要动原本就存在的死代码，除非被明确要求。

检验标准：每一行改动都应该能直接追溯到用户的需求。

## 4. 目标驱动执行

**定义成功标准。循环验证直到确认通过。**

把任务转化为可验证的目标：

- "加验证"→"为非法输入写测试，然后让测试通过"
- "修这个bug"→"写一个能复现它的测试，然后让测试通过"
- "重构X"→"确保重构前后测试都能通过"

对于多步骤任务，列出简要计划：

1. [步骤]→验证：[检查项]
2. [步骤]→验证：[检查项]
3. [步骤]→验证：[检查项]

强成功标准让你能独立循环推进。弱标准（"让它能跑"）则需要不断澄清。

---

**这些准则起作用的标志是：**diff中不必要的改动更少了，因过度复杂化而返工更少了，澄清性问题出现在实现之前而不是犯错之后。
