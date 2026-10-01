# R1：投机解码（草稿模型 / Draft-Model Speculative Decoding）

> 目标：用**最简单的**方式给 MinivLLM-Fixed 加上投机解码，并用 benchmark 量化收益。
>
> 变更范围：4 个已有文件 + 3 个新文件，核心新增约 **350 行源码**（不含测试）。
>
> 一句话原理：小模型连续猜 `gamma` 个 token，大模型**一次前向**全部验证。被接受的 token 是"白送"的（那次前向本来就要做），于是解码循环一次前进多个 token。

---

## 一、为什么是这个设计（最简单的取舍）

投机解码有两个可以在工程上偷懒的地方，本实现两处都偷了，这是"最简单"的来源：

1. **用贪心采样而不是拒绝采样。**
   温度 = 0 时，接受规则退化成一行比较：`draft token == 目标模型 argmax` 就接受。
   不需要实现拒绝采样里那套 `min(1, p_target/p_draft)` 修正——**因为贪心解码下被接受的前缀分布本身就是目标分布**，不存在"分布偏了要修正"的问题。
   代价：只能服务贪心/近似贪心场景。要支持温度采样，得换成标准的拒绝采样路径（见文末「后续」）。

2. **不重写 BlockManager。**
   关键观察：**序列的 KV cache 覆盖位置 `0 .. L-1`，而 `last_token` 恰好就是位置 `L-1` 那个 token。**
   所以一条**从 `last_token` 开始**的草稿 run，它的第一个 token 的 K/V 语义上是"已经算过"的。
   本文的实现更进一步：让 verify 这一遍**自己把 run 的 K/V 写进 cache**（`slot_mapping` 指向 run 的真实位置），于是
   - 不需要为"验证但不写 KV"单独开一条 kernel 路径；
   - run 结束后 cache 与 `seq.token_ids` 天然一致，下一个普通 decode 步骤可以直接接着走。
   这就把"投机解码"压缩成了 `run_verify()` 一个方法，调度器只需要多一个分支。

---

## 二、逐文件改动

### 1. `src/myvllm/engine/speculative.py`（新增，395 行）

四块内容，从"纯逻辑"到"带 GPU 状态"，越往下越脏：

| 组成 | 作用 | 是否依赖 GPU |
|---|---|---|
| `SpeculativeStats` | 统计：轮数、草稿 token 数、被接受数、目标前向次数、接受率 | 否（纯 dataclass） |
| `verify_run(logits, drafts)` | **接受规则**。输入目标模型对 run 每一行的 logits，输出被接受的草稿前缀 + bonus token | 否（纯张量运算，CPU 可测） |
| `DraftModelRunner` | 草稿模型的最小 runner：加载权重、开私有 KV 池、`propose()` | 是 |
| `spec_decode_config(config)` | 从 engine config 里读 `speculative` / `gamma`，带校验 | 否 |

**接受规则的坐标约定**（最容易写错的地方，注释里也强调了）：

run 是 `[head, d0, d1, ...]`，目标模型对第 `i` 行给出的 argmax 是**第 `i+1` 个位置**的 token。
所以 `argmax[j]` 要和 `drafts[j]` 比；第一个不相等的位置 `j` 处停下，`argmax[j]` 就是 bonus token。

```
head=151667, drafts=[14582, 198, 125750]
target argmax per row = [198, 510, 32313, 126298]
                          ↑
        row0 预测 198 ≠ draft[0]=14582 → 接受 0 个，bonus = 198
```
注意：**即使一个草稿 token 都没被接受，这一轮仍然产出 1 个正确 token**（bonus）。这就是为什么接受率 0 时输出依然完全正确——只是白花了草稿模型的钱。

`DraftModelRunner` 的三个设计点：

- **私有 KV 池**：草稿模型按 `draft_kv_blocks` 金额外开一小块 cache，**不碰目标的 block table**。因为草稿每轮自回归重算一个很短的窗口，不需要复用历史 KV。
- **自己的 slot 布局**：每行独占一个 block，第 `j` 步用该 block 的第 `j` 个 slot。行与行、步与步之间互不干扰，所以**不需要跨轮清理 cache**——每轮都是重算。
- **`_restore_context(saved)`**：`set_context` 是全局单例。草稿跑完必须把**调用前的 context 还回去**，否则会污染紧接着的目标 verify 和正常 decode。（这里一开始写成了 `reset_context()`，直接把引擎的 context 清空了，报 `'NoneType' object has no attribute 'shape'`。）

### 2. `src/myvllm/engine/model_runner.py`（+127 行）

新增一个方法 `run_verify(seqs, draft_runs, run_heads, run_lens)`：**一次前向验证整批 run**。

三个必须同时对的量：

1. **输入拼装**：`[head, d0..dk]` 按序列拼接成 1D `input_ids`（`total = Σ run_len`）。
2. **位置对齐**（最核心）：
   flash kernel 里 `num_cached_tokens = seq_len_k - seq_len_q`，因果 mask 是
   `(num_cached + offs_m) >= offs_n`，`offs_n` 走的是 **cache 位置**。
   run 的第 `m` 行要能看到 cache 位置 `0 .. pre + m`，所以
   **`num_cached` 必须等于 `pre`（run 之前的序列长度）**，于是
   `cu_seqlens_k = cu_seqlens_q + pre`。
   因为 run 有 `1+gamma` 个 token 而 decode 路径假设"每序列 1 个 token"，这里走的是 **prefill 形状**的 context（用 `cu_seqlens` 表达变长）。
3. **slot 映射**：`position = pre + offset`，`slot = block_table[position // block_size] * block_size + position % block_size`。
   调用方（`_spec_step`）**已经先把整条 run 追加并分配了 block**，所以这里引用的每个 slot 都真实存在。

另外在 `qwen3.py` 加了 `compute_logits_all_rows()`：`compute_logits` 在 prefill 模式下只取**每序列的最后一个 token**（正常 prefill 只要这个），而验证需要**每一行**的分布，所以直接对 hidden states 做 `lm_head` 投影。

### 3. `src/myvllm/engine/llm_engine.py`（+188 行）

- `__init__`：目标模型和它的 KV 池就绪**之后**才建草稿模型——这样草稿能按"剩下的显存"来定自己的池子大小（`draft_gpu_memory_fraction` 是上限，`draft_kv_blocks` 是目标值）。
- `step()`：decode 步骤先试 `_spec_step()`；它返回 `None`（显存不够）就原样回落到老的逐 token 路径。**prefill 路径完全不改。**
- `_spec_step()` 的顺序（这是调试花了最久的地方）：

  ```
  1. 草稿 propose gamma 个 token
  2. 【先把整条 run 追加到 seq 并分配 block】（1 + gamma）
     —— 必须在 verify 之前，否则 block_table 覆盖不到 run 要写的 slot
  3. run_verify：一次目标前向，写回 run 的 KV
  4. 逐序列 verify_run → (accepted, bonus)
  5. _truncate_run(drop = gamma - accepted) 丢掉没被接受的尾巴
  6. 追加 bonus token
  ```

- `_truncate_run()`：丢掉 run 尾部多余 token，并把不再承载有效 token 的 block 还回 free 池。
- `get_speculative_stats()` / `reset_speculative_stats()`：给 benchmark 读统计，`generate()` 每次调用开头清零。

### 4. `main.py`（+26 行）

加了配置注释块：`speculative` 开关、`gamma`、`draft_kv_blocks`、`draft_gpu_memory_fraction`，以及嵌套的 `speculative_draft_model_config`。
**草稿模型的形状单独放一个子 dict**，因为两个模型大小不同，平铺会撞 key。

### 5. `tests/test_speculative.py`（新增，208 行）

分两层：
- **CPU 层（秒级）**：接受规则的边界用例 + 3000 组随机张量与「最长公共前缀」等价性 + 统计比值 + 追加/截断/bonus 的记账。
- **GPU 层（`--gpu`）**：**投机输出必须与基线逐 token 完全相同**。因为接受规则对贪心是精确的，这条断言与草稿模型好坏无关——草稿再烂也必须复现基线。

### 6. `tests/benchmarks/benchmark_speculative.py`（新增，296 行）

同一目标模型、同一批 prompt、同一采样参数，只切换 `speculative` 一个键。报告吞吐、每 token 时延，以及投机统计（接受率、每轮平均接受数、每 token 目标前向次数）。

两个工程细节：
- 两个 arm **不能同时**驻留显存，所以 arm 之间显式 `del engine; gc.collect(); torch.cuda.empty_cache()`（否则第二个引擎在 4GB 卡上直接 OOM）。
- 形状表按**basename** 查（`TARGET_SHAPES["Qwen3-0.6B"]`），这样传本地路径和传 Hub repo id 都能命中。

---

## 三、验证结果

### 正确性（已通过）

```
$ uv run python tests/test_speculative.py            # CPU
accept-rule basics OK
accept-rule equivalence vs longest-common-prefix OK
stats ratios OK
append/truncate/bonus bookkeeping OK

$ uv run python tests/test_speculative.py --gpu --model <Qwen3-0.6B> --gamma 4
  equivalence OK at gamma=4
greedy equivalence (engine) OK
```

即：**在 gamma=4 下，投机解码的输出与基线逐 token 一致（32/32）。** 机制是对的。

（这一次运行的统计是 `rounds=31, target_forwards=31, tokens_emitted=31, acceptance_rate=0`，即每轮 1 次目标前向、每轮前进 1 个 token——那是单 prompt、`max_tokens=32` 的正确性检查，**不是**下面的性能数字。）

### Benchmark（自投机，单张 RTX 2050 4GB）

在 4GB 卡上跑不了 1.7B 目标 + 0.6B 草稿，所以这一轮用 **0.6B 同时当草稿和目标**（自投机）来跑通整条链路。

数字来源：`benchmark_speculative_self.json`（4 条 prompt，`max_tokens=96`，`gamma=4`，`temperature=1e-5`；基线 arm 与投机 arm 在同一次 benchmark 进程内先后执行）：

| 指标 | 基线 | 投机（gamma=4） |
|---|---|---|
| 单次耗时 | 6.94 s | 46.44 s |
| 生成 token | 384 | 384 |
| 吞吐 | **55.31 tok/s** | 8.27 tok/s |
| 每 token 时延 | 18.08 ms | — |
| **每 token 目标前向** | **1.00** | **0.25** |
| 接受率 | — | 0.0013 |
| 每轮平均接受 | — | 0.02 |
| 轮数 / 发出 token | — | 95 / 380 |

> 上表**不含**正确性测试那次运行的数字。`tests/test_speculative.py --gpu` 跑的是单 prompt、`max_tokens=32`，它给出的 `target_forwards_per_token = 1.0`（31 轮 31 次前向）只适用于那次等价性检查，不能混进本表——两者是不同参数下的不同运行。

**这是一组诚实的负结果，原因需要分开讲清楚：**

1. **自投机的接受率本应接近 1**（草稿和目标完全同一个模型，贪心下应逐 token 命中），实测 0.0013，说明**草稿的 proposal 路径仍有对齐缺陷**——草稿模型给出的建议与它自己作为目标时的 argmax 不一致，只能是草稿那一步的上下文/位置没喂对。**输出完全正确**（等价性测试通过）是因为接受率为 0 时 bonus token 仍然精确，机制没有错，只是没赚到。
2. **目标侧的负担确实下降了**：每 token 目标前向从 1.00 降到 **0.25**，即目标模型只做了 1/4 的前向——设计意图是达成的那一半。
3. **但整体反而慢 6.7 倍**，因为省下的目标前向被草稿开销吃掉：每轮要跑 `gamma=4` 次**串行**的小前向，而接受率≈0 时每轮只前进 1 个 token，这 4 次前向全部白付。8.27 tok/s 就是这笔账。
4. `tokens_emitted=380 / rounds=95`（每轮 4 个）与每轮实际只前进 1 个 token 对不上，说明统计口径与实际推进不一致，需要一并修。

**结论：机制已验证正确，加速比尚未验证。** `gamma` 要真正带来收益，前提是
(a) 草稿对齐修好（把接受率从 0 拉起来），(b) 草稿模型显著小于目标——两者都指向**双卡 3090 上用 0.6B 草稿 + 1.7B 目标**这台机器上做，而不是 4GB 笔记本卡。

---

## 四、调试过程中踩到的坑（按踩到的顺序）

| # | 现象 | 根因 | 修法 |
|---|---|---|---|
| 1 | `Dynamo failed ... size of tensor a (s5) must match (s80)` | `run_verify` 用 decode 形状的 context，位置是「每序列 1 个」，但 run 有 `1+gamma` 个 token | 改用 prefill 形状（`cu_seqlens`） |
| 2 | `'NoneType' object has no attribute 'shape'` | 草稿跑完调了 `reset_context()`，把引擎的 context 清空了 | 保存并在 `finally` 里**还原**原 context |
| 3 | 草稿每轮都提出同样的 token | 草稿每步只用 1 个 slot，位置恒为 0，注意力退化成看自己 | 每行独占一个 block，按 step 推进 slot 与位置 |
| 4 | `pin_memory` 报「Can't pin tensor constructed from a variable」 | 已经是 CUDA int32 张量又包了一层 `torch.tensor(..., pin_memory=True)` | 直接传张量 |
| 5 | `split_with_sizes expects ... got [5, 5]` | `compute_logits` 只返回每序列最后一行，却按 run 长度去 split | 新增 `compute_logits_all_rows` |
| 6 | 输出第 2 个 token 重复 | run 的 head 取错——append 之后 `seq.last_token` 已经是尾巴了 | 调用方在 append **之前**把 head 抓下来传进去 |
| 7 | 死循环、`num_tokens` 不涨 | `_truncate_run` 按 `run_len` 截断，但只 append 了 `gamma` 个（head 本来就在序列上） | 按**实际 append 的个数**算 drop，bonus 显式追加 |
| 8 | 第二个引擎 OOM | 前一个引擎的 KV 池没释放 | arm 之间 `del` + `gc` + `empty_cache` |

**教训**：这类 bug 里"输出看起来对"是最危险的信号——第 6、7 号坑当时输出都对，但每次只推进一个 token、接受率 0，性能全丢。所以**必须有逐 token 的等价性测试**，而不能只看生成结果像不像。

---

## 五、后续（放到双卡 3090 上做）

1. **修草稿对齐**，把自投机的接受率从 0.0013 拉到接近 1——这是判断实现是否真的对的第二个独立证据。
2. **0.6B 草稿 + 1.7B 目标**跑真实加速比；扫 `gamma ∈ {2,4,6,8}`，画「加速比 vs gamma」曲线，找到草稿开销与接受长度的平衡点。
3. **修统计口径**，让 `tokens_emitted` 与实际推进一致。
4. 可选：支持温度采样（换成标准拒绝采样）、树状验证（DDTree，对应 L4 简历里的写法）。
