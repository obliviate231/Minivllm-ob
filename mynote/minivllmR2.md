# R2

## diff相关语法

```sh
(myvllm) ob@obliviate:~/infra/MinivLLM-Fixed$ git log --oneline acdac94..798b455
798b455 (tag: release-2) solve bug for single sequence exceeds total kv cache space
(myvllm) ob@obliviate:~/infra/MinivLLM-Fixed$ git diff acdac94..798b455 --stat
 src/myvllm/engine/block_manager.py |  5 +++--
 src/myvllm/engine/llm_engine.py    |  3 ++-
 src/myvllm/engine/scheduler.py     | 37 +++++++++++++++++++++++++++++++++++--
 tests/test_scheduler.py            | 80 +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++---------
 4 files changed, 111 insertions(+), 14 deletions(-)
```

```
git diff release-1 release-2
git diff acdac94 798b455 -- src/myvllm/engine/scheduler.py
↑↓看代码 q退出
git diff 798b455 686b547 -- 
```

## 总结

直接搬[release-2](https://github.com/KevinWu8192/MinivLLM-Fixed/blob/main/releases/release-2_zh.md)的了

### 引擎与请求级上下文限制

**涉及文件**

- `src/myvllm/engine/llm_engine.py`
- `src/myvllm/engine/scheduler.py`
- `tests/test_scheduler.py`

**问题**

- 引擎允许 `max_model_length` 超过物理 KV Cache 容量。([原始调度器初始化](https://github.com/KevinWu8192/MyMinivLLM/blob/acdac942b3e918855125da7e03d63cba3feb8fc9/src/myvllm/engine/scheduler.py#L6-L15))
- 请求可以设置高于引擎上限的上下文长度。([原始请求准入](https://github.com/KevinWu8192/MyMinivLLM/blob/acdac942b3e918855125da7e03d63cba3feb8fc9/src/myvllm/engine/scheduler.py#L18-L23))
- Prompt 已经占满上下文时仍能进入队列，没有为生成 Token 留出空间。([原始请求准入](https://github.com/KevinWu8192/MyMinivLLM/blob/acdac942b3e918855125da7e03d63cba3feb8fc9/src/myvllm/engine/scheduler.py#L18-L23))
- 超长请求可能永远无法被调度，而不是提前返回清晰错误。([原始 Prefill 调度](https://github.com/KevinWu8192/MyMinivLLM/blob/acdac942b3e918855125da7e03d63cba3feb8fc9/src/myvllm/engine/scheduler.py#L25-L41))
  - 因为调度器的 `waiting` 队列是 **FIFO（先进先出）** 或基于优先级的循环。这个超长请求每次都排在队尾，每次都跑出来“浪费一次调度机会”，占用 CPU 资源判断，然后灰溜溜回去。**它既不会前进到 `running` 阶段生成 Token，也不会因为超时被系统踢出**

**示例**

如果引擎拥有 8 个 KV Block，且 `block_size = 16`，最多只能保存 128 个 Token。此时配置 `max_model_length = 256`，请求虽然能通过 API 校验，但执行时永远无法保存完整 Context。类似地，即使引擎上限正确设为 128，接受一个 128 Token 的 Prompt 也没有空间保存第一个生成 Token。这些请求会在内存压力下卡住或反复抢占，而不是在准入阶段立即失败。

**修复**

- 将引擎的 `max_model_length` 传入调度器。([引擎配置](https://github.com/KevinWu8192/MyMinivLLM/blob/798b4552a1401f1eebbfff865816a95169aada5f/src/myvllm/engine/llm_engine.py#L47-L56))
- 构造引擎时，将 Block 容量换算为 Token 数（`max_cached_blocks * block_size`），无法被物理 KV Cache 容纳的引擎 Context 上限会立即被拒绝。([容量验证](https://github.com/KevinWu8192/MyMinivLLM/blob/798b4552a1401f1eebbfff865816a95169aada5f/src/myvllm/engine/scheduler.py#L6-L26))
- 请求准入时只解析一次有效上限：未设置则继承引擎值；允许更小的正数请求上限；非正数或超过引擎的值会在进入 `waiting` 前被拒绝。([请求验证](https://github.com/KevinWu8192/MyMinivLLM/blob/798b4552a1401f1eebbfff865816a95169aada5f/src/myvllm/engine/scheduler.py#L34-L58))
- 强制满足 `num_prompt_tokens < request_max_length`，把解析后的上限写入 Sequence，并在每次采样 Token 后检查总长度，使生成在边界处结束并立即释放 Block。([Prompt 验证](https://github.com/KevinWu8192/MyMinivLLM/blob/798b4552a1401f1eebbfff865816a95169aada5f/src/myvllm/engine/scheduler.py#L48-L58)，[停止条件](https://github.com/KevinWu8192/MyMinivLLM/blob/798b4552a1401f1eebbfff865816a95169aada5f/src/myvllm/engine/scheduler.py#L113-L126))

### Decode Block 分配边界

**问题**

`postprocess()` 会先将新采样的 Token 追加到 Sequence，然后才调用 `can_append()`。旧逻辑检查 `num_tokens % block_size == 0`，因此检查的是一个 Block 的末尾，而不是下一个 Block 的首个 Token，可能在没有所需 KV Cache 空间的情况下继续 Decode。([原始检查](https://github.com/KevinWu8192/MyMinivLLM/blob/acdac942b3e918855125da7e03d63cba3feb8fc9/src/myvllm/engine/block_manager.py#L157-L161))

**示例**

假设 `block_size = 4`，Sequence 当前有 4 个 Token，存放在 Block 0 中。`postprocess()` 追加 Token 5 后，`num_tokens = 5`，这个 Token 属于新的 Block 1。旧条件检查 `5 % 4 == 0`，结果为 false，因此认为不需要新 Block。下一轮 Decode 随后会为 Block 1 构造 Slot Mapping，但这个 Block 从未被分配。由于 Token 已经追加，正确边界应为 `5 % 4 == 1`。

**修复**

让 `can_append()` 与 `append()` 使用相同的“Token 已追加”状态约定：仅在 `num_tokens % block_size == 1` 时要求空闲 Block；其他情况下，新 Token 仍位于已经分配的最后一个 Block 内。这样容量准入与后续 Block Table 更新保持一致。([边界修复](https://github.com/KevinWu8192/MyMinivLLM/blob/798b4552a1401f1eebbfff865816a95169aada5f/src/myvllm/engine/block_manager.py#L153-L163))

### 验证

调度器测试覆盖了继承和请求级上下文上限、无效上限、没有生成空间的 Prompt、KV Cache 容量溢出，以及在上下文边界准确停止。([测试](https://github.com/KevinWu8192/MyMinivLLM/blob/798b4552a1401f1eebbfff865816a95169aada5f/tests/test_scheduler.py#L190-L250))

## 具体细节

```python
src/myvllm/engine/block_manager.py

    def can_append(self, seq: Sequence) -> bool:
-       if seq.num_tokens % self.block_size == 0:
+		if seq.num_tokens % self.block_size == 1:
            return len(self.free_block_ids) > 0
        return True
```

```python
src/myvllm/engine/llm_engine.py 

-            eos=config.get("eos", 50256)
+            eos=config.get("eos", 50256),
+            max_model_length=config["max_model_length"], 
				#调度器（Scheduler）层面最大上下文长度
         )
```


```python
src/myvllm/engine/scheduler.py
 class Scheduler:
-    def __init__(self, max_num_sequences: int, max_num_batched_tokens: int, max_cached_blocks: int, block_size: int, eos: int):
+    def __init__(self, max_num_sequences: int, max_num_batched_tokens: int,
+                 max_cached_blocks: int, block_size: int, eos: int,
+                 max_model_length: int):


+        if max_model_length <= 0:
+            raise ValueError("max_model_length must be greater than 0")
+
+        max_kv_cache_tokens = max_cached_blocks * block_size
+        if max_model_length > max_kv_cache_tokens:
+            raise ValueError(
+                f"max_model_length ({max_model_length}) exceeds the KV cache "

  		...
+        self.max_model_length = max_model_length

     def add_sequence(self, sequence: Sequence):
+        request_max_length = sequence.max_model_length # 最大上下文长度
+        if request_max_length is None:
+            request_max_length = self.max_model_length
+        elif request_max_length > self.max_model_length:
+            raise ValueError(
+                f"Request max_model_length ({request_max_length}) exceeds the "
+                f"engine limit ({self.max_model_length})"
+            )
+        elif request_max_length <= 0:
+            raise ValueError("Request max_model_length must be greater than 0")
+
+        # Prefill 阶段必须预留至少一个生成 Token 的位置
+        # 我们直接拒绝过长的 Prompt，而不是静默地截断其前缀。
+        if sequence.num_prompt_tokens >= request_max_length:
+            raise ValueError(
+                f"Prompt length ({sequence.num_prompt_tokens}) must be smaller "
+                f"than the request context limit ({request_max_length})"
+            )
+
+        sequence.max_model_length = request_max_length
         self.waiting.append(sequence)
		"""
		self.max_model_length >=[右为None时赋等值] sequence.max_model_length(上下文长度) > 0
		sequence.max_model_length(上下文长度) > sequence.num_prompt_token   

		Q：为什么需要 request_max_length 
		反例
            输入：
				sequence.max_model_length = None, 
				num_prompt_tokens = 3000, 
				self.max_model_length = 2048
            步骤① 后：sequence.max_model_length 变为 2048
            步骤② 判断：3000 >= 2048 成立
            步骤③ 抛出异常，函数退出
            结果：sequence 对象内部的 max_model_length 被永久改成了 2048，但该请求并未入队，
            且日志里会记录"上限2048，长度3000"——自相矛盾，对象状态被污染。
		"""

```

```python
tests/test_scheduler.py 
     def make_scheduler(
         max_num_sequences=10,
         max_cached_blocks=100,
         block_size=4,
+    	max_model_length=100,
     ):
         return Scheduler(
             max_num_sequences=max_num_sequences,
             max_cached_blocks=max_cached_blocks,
             block_size=block_size,
             eos=0,
+        	max_model_length=max_model_length,
         )

    def _run(self, scheduler: Scheduler):
-        seq_a = Sequence([1, 2, 3])
-        seq_b = Sequence([4, 5, 6])
-        seq_c = Sequence([7, 8, 9])
+        seq_a = Sequence([1, 2, 3], block_size=4)
+        seq_b = Sequence([4, 5, 6], block_size=4)
+        seq_c = Sequence([7, 8, 9], block_size=4)
         inject_running(scheduler, seq_a, seq_b, seq_c)

	# 这个 block_size=4 是作为属性存进 Sequence 对象里的，它不影响 token 数据
```

```python
tests/test_scheduler.py 
+class TestContextLength:
+    def test_request_inherits_engine_context_limit(self): 
    	 # 继承默认值
+        scheduler = make_scheduler(max_model_length=16)
+        seq = Sequence([1, 2], block_size=4)
+
+        scheduler.add_sequence(seq)
+
+        assert seq.max_model_length == 16
+
+    def test_request_can_choose_a_smaller_context_limit(self):
		 # 允许用户设更小的值
+        scheduler = make_scheduler(max_model_length=16)
+        params = SamplingParams(max_model_length=8)
+        seq = Sequence([1, 2], block_size=4, sampling_params=params)
+
+        scheduler.add_sequence(seq)
+
+        assert seq.max_model_length == 8
+
+    def test_rejects_prompt_that_leaves_no_generation_room(self):
		# 拒绝没有生成空间的 Prompt
+        scheduler = make_scheduler(max_model_length=8)
+        seq = Sequence(list(range(8)), block_size=4)
+
+        with pytest.raises(ValueError, match="Prompt length"): 
			# 断言“执行下面的代码一定会抛出 ValueError
+            scheduler.add_sequence(seq)
+
+    def test_rejects_request_limit_above_engine_limit(self):
		 # 拒绝用户超限请求
+        scheduler = make_scheduler(max_model_length=8)
+        params = SamplingParams(max_model_length=9)
+        seq = Sequence([1], block_size=4, sampling_params=params)
+
+        with pytest.raises(ValueError, match="exceeds the engine limit"):
+            scheduler.add_sequence(seq)
+
+    def test_engine_limit_must_fit_in_kv_cache(self):
		 # 启动配置必须适配物理容量（最底层）
+        with pytest.raises(ValueError, match="exceeds the KV cache capacity"):
+            make_scheduler(
+                max_cached_blocks=2,
+                block_size=4,
+                max_model_length=9,
+            )
+
+    def test_generation_stops_at_request_context_limit(self):
		 # 生成必须在边界精准停止（端到端）
		 # Decode 追加 Token 后，如果触及上限，立马停止并释放资源
+        scheduler = make_scheduler(max_model_length=8)
+        params = SamplingParams(max_tokens=10, max_model_length=4)
+        seq = Sequence([1, 2, 3], block_size=4, sampling_params=params)
+        seq.status = SequenceStatus.RUNNING
+        seq.block_table = [0]
+        scheduler.block_manager.blocks[0].ref_count = 1
+        scheduler.block_manager.free_block_ids.remove(0)
+        scheduler.block_manager.used_block_ids.add(0)
+        scheduler.running.append(seq)
+
+        scheduler.postprocess([seq], [42])
+
+        assert seq.is_finished # 状态是finished
+        assert seq.num_tokens == 4 # 没有多生成token
+        assert seq not in scheduler.running # 从running移除
(END)
```

