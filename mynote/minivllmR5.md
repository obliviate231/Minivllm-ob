# R5

```
‎benchmark_tps.py & main.py &  main_llama32.py  

config = {
	...
+   'dtype': 'bfloat16',
-	'max_num_batch_tokens': 4096,
	...
}
```

## 多 GPU Worker 协同

**涉及文件**

- `src/myvllm/engine/sequence.py`   
- `src/myvllm/engine/model_runner.py`
- `src/myvllm/engine/llm_engine.py`

**问题**

- Worker 反序列化时遗漏 `Sequence.block_size`，导致 Rank 0 之外的进程调用 `num_blocks` 等方法时失败。([原始序列化](https://github.com/KevinWu8192/MyMinivLLM/blob/4d1760ded441988663c59f5bd9479a80de234731/src/myvllm/engine/sequence.py#L94-L110))
- 所有引擎使用同一个分布式 Rendezvous 端口和共享内存名称，并发引擎之间会发生冲突。([原始 Worker 配置](https://github.com/KevinWu8192/MyMinivLLM/blob/4d1760ded441988663c59f5bd9479a80de234731/src/myvllm/engine/model_runner.py#L20-L31))
- 写入共享内存前没有检查序列化 Payload 大小。([原始共享内存写入](https://github.com/KevinWu8192/MyMinivLLM/blob/4d1760ded441988663c59f5bd9479a80de234731/src/myvllm/engine/model_runner.py#L124-L143))
- Worker 初始化和清理顺序可能将异常表现为 NCCL 卡死，或者重复执行 Cleanup。([原始 Worker 生命周期](https://github.com/KevinWu8192/MyMinivLLM/blob/4d1760ded441988663c59f5bd9479a80de234731/src/myvllm/engine/model_runner.py#L145-L169)，[原始引擎清理](https://github.com/KevinWu8192/MyMinivLLM/blob/4d1760ded441988663c59f5bd9479a80de234731/src/myvllm/engine/llm_engine.py#L58-L62))

**示例**

同一主机上启动两个引擎时，它们会尝试绑定相同的 Rendezvous 端口并创建同名共享内存。第二个引擎可能在初始化时失败，也可能连接到第一个引擎拥有的资源。另一方面，Rank 0 序列化 Sequence 时遗漏了 `block_size`；Worker 恢复后计算 `num_blocks` 会抛出 `AttributeError: 'Sequence' object has no attribute 'block_size'`。此时 Rank 0 可能正在等待 NCCL Collective，使真正的 Python 异常看起来像分布式卡死。

**修复**

- [x] 将 `block_size` 与其他 Sequence 状态一起序列化和恢复。([Sequence 状态](https://github.com/KevinWu8192/MyMinivLLM/blob/ee950af0b35f5ac87fa6a495238cf6ebb57aa99d/src/myvllm/engine/sequence.py#L95-L122))
- [x] Spawn 进程前根据可用 CUDA 设备验证 `world_size`，让操作系统选择空闲 TCP 端口，并生成基于 UUID 的共享内存名称。两个值都通过 Worker Config 传递，使同一引擎内所有 Rank 保持一致，不同引擎之间相互隔离。([引擎初始化](https://github.com/KevinWu8192/MyMinivLLM/blob/ee950af0b35f5ac87fa6a495238cf6ebb57aa99d/src/myvllm/engine/llm_engine.py#L29-L59))
- [x] Rank 0 创建并拥有共享内存，其他 Rank 按名称连接。每个命令先序列化，再将字节长度与配置容量比较，避免写入被截断的 Payload。([共享内存](https://github.com/KevinWu8192/MyMinivLLM/blob/ee950af0b35f5ac87fa6a495238cf6ebb57aa99d/src/myvllm/engine/model_runner.py#L119-L169))
- [x] 使 Worker 和引擎 Shutdown 具备幂等性，并在清理时 Join Worker 进程。([Worker 清理](https://github.com/KevinWu8192/MyMinivLLM/blob/ee950af0b35f5ac87fa6a495238cf6ebb57aa99d/src/myvllm/engine/model_runner.py#L174-L196)，[引擎清理](https://github.com/KevinWu8192/MyMinivLLM/blob/ee950af0b35f5ac87fa6a495238cf6ebb57aa99d/src/myvllm/engine/llm_engine.py#L92-L99))

### sequence.py

在 Python 中，默认参数在函数定义时（即模块加载时）就会被求值一次，生成一个 SamplingParams 实例。之后所有未显式传入 sampling_params 的 Sequence 都会共享同一个 SamplingParams 对象。
这就导致了一个隐蔽的 Bug：如果某个请求修改了共享对象中的字段（例如 temperature），其他所有使用默认参数的请求都会被“传染”，行为变得不可预测。

```python
class Sequence:
    counter = count()

-   def __init__(self, token_ids: list[int], block_size: int, sampling_params = SamplingParams()):
+   def __init__(self, token_ids: list[int], block_size: int, sampling_params=None):
+       # 修复可变默认参数陷阱：每次调用都创建独立的 SamplingParams 实例
+       if sampling_params is None:
+           sampling_params = SamplingParams()
        self.block_size = block_size  # 每个块的 token 数量
        # 记录序列 ID
        self.seq_id = next(Sequence.counter)
        # ... 其他初始化 ...

    def __getstate__(self):
        return (
+           self.block_size,          # 新增：序列化块大小
            self.num_tokens,
            self.num_prompt_tokens,
            self.num_cached_tokens,
            # ... 其他字段 ...
        )

    def __setstate__(self, state):
        (
+           self.block_size,          # 新增：反序列化块大小
            self.num_tokens,
            self.num_prompt_tokens,
            self.num_cached_tokens,
            # ... 其他字段 ...
        ) = state
        # ... 恢复其他属性 ...
        # 恢复 last_token 属性
        self.last_token = self.token_ids[-1] if self.token_ids else None
```

### model_runner.py

```python
‎src/myvllm/engine/model_runner.py
class ModelRunner:
    def __init__(self, config: dict, rank: int, event: Event | list[Event]):
-       dist.init_process_group('nccl', "tcp://localhost:12345", world_size=config['world_size'], rank=rank)
        dist.init_process_group(
            'nccl',
+           config.get('distributed_init_method', "tcp://127.0.0.1:12345"),
            world_size=config['world_size'],
            rank=rank,
        )
```

#### **配置数据类型：支持 'float16', 'bfloat16', 'float32'**

```python
+       dtype_config = config.get('dtype', 'bfloat16')
+       if isinstance(dtype_config, str):
+           dtype_name = dtype_config.removeprefix('torch.')
+           dtype_mapping = {
+               'float16': torch.float16,
+               'bfloat16': torch.bfloat16,
+               'float32': torch.float32,
+           }
+           if dtype_name not in dtype_mapping:
+               raise ValueError(f"Unsupported dtype: {dtype_config}")
+           self.default_dtype = dtype_mapping[dtype_name]
+       else:
+           self.default_dtype = dtype_config
+       self.model = self.model.to(device=f'cuda:{rank}', dtype=self.default_dtype)
- 		self.model = self.model.cuda(rank)
-       self.default_dtype = torch.get_default_dtype()
```

#### **分配共享内存**

旧逻辑 先清理旧的名为'myvllm'的SharedMemory，再分配，shm_size固定为2**20(1MB)
新逻辑 先算次完整请求批次的序列化数据最大可能大小，再分配
旧逻辑之所以要“先删旧的”，是因为名称写死了，怕有残留；新逻辑名称是唯一的，不存在“旧的”，所以不需要删——而且删了反而危险

```python
        if self.world_size > 1:
            # Synchronize before setting up shared memory
            dist.barrier()
            if self.rank == 0:
-                # Try to clean up existing shared memory first
-                try:
-                    old_shm = SharedMemory(name='myvllm')
-                    old_shm.close()
-                    old_shm.unlink()
-                except FileNotFoundError:
-                    pass  # Doesn't exist, which is fine
-                self.shm = SharedMemory(name='myvllm', create=True, size=2**20)

+                self.shm_size = config.get(
+                    'shared_memory_size',
+                    max(
+                        2**20,
+                        config['max_num_batched_tokens'] * 16
+                        + config['max_num_sequences'] * 4096,
+                    ),
+                )
+                self.shm = SharedMemory(
+                    name=config['shared_memory_name'],
+                    create=True,
+                    size=self.shm_size,
+                )
                # Barrier to ensure rank 1 waits until shared memory is created
                dist.barrier()
            else:
                # Wait for rank 0 to create shared memory
                dist.barrier()
-               self.shm = SharedMemory(name='myvllm')                
+               self.shm = SharedMemory(name=config['shared_memory_name'])
                # Don't call self.loop() here - let the spawning code handle it
                # Otherwise we'll be stuck in an infinite loop during __init__
```

```python
	# 仅 rank != 0 时调用 read_shm
    def read_shm(self): 
+   # 防御：检查长度合法性，防止越界

    # 仅 rank == 0 时调用 write_shm 
    def write_shm(self, method_name: str, args: tuple):	
+	# 检查 payload 是否超过共享内存容量，防止截断
```

```python
    # 预热模型
    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_tokens = self.config['max_num_batch_tokens']
        max_tokens = self.config['max_num_batched_tokens']
        max_model_length = self.config['max_model_length']
        
-       batch_size = max_tokens // max_model_length
+       batch_size = max(1, max_tokens // warmup_length)

-       seqs = [
			Sequence(token_ids=[0]*max_model_length, 
			block_size=self.config['block_size']) for _ in range(batch_size)]  
+       seqs = [
			Sequence(token_ids=[0] * min(max_tokens, max_model_length), 
			block_size=self.config['block_size']) for _ in range(batch_size)]
        """
        seqs长这样：
        [
          Sequence(
            token_ids=[0, 0, 0, 0],
            block_size=2
            // 可能还有其他默认属性，如 seq_id、num_cached_tokens 等
          ),
          Sequence(
            token_ids=[0, 0, 0, 0],
            block_size=2
          )
        ]
        """
        self.run(seqs, is_prefill=True)
        torch.cuda.empty_cache()
```

### llm_engine.py

#### 多了一些assert,浅拷贝,config

```python
‎src/myvllm/engine/llm_engine.py 

import atexit
import torch.distributed as dist
+import torch
import time
import torch.multiprocessing as mp
+ import socket
+ import uuid

from myvllm.engine.sequence import Sequence
from myvllm.engine.scheduler import Scheduler
@@ -24,75 +27,114 @@ def worker_process(config, rank, event):

class LLMEngine:
    def __init__(self, config: dict):
-       self.config = config
+       self.config = dict(config)  # !!!浅拷贝，防止外部修改影响内部状态
        world_size = config.get("world_size", 1)
+       if world_size <= 0:
+           raise ValueError("world_size must be greater than 0")
+       if world_size > torch.cuda.device_count():
+           raise ValueError(
+               f"world_size ({world_size}) exceeds available CUDA devices "
+               f"({torch.cuda.device_count()})"
+           )
+       # 校验 RoPE 容量 >= 最大模型长度
+       max_position = self.config.get(
+           "max_position", self.config.get("max_position_embeddings")
+       )
+       if max_position is not None and max_position < self.config["max_model_length"]:
			# 模型最大上下文长度（max_model_length）”不能超过“位置编码支持的最大长度（max_position）
+           raise ValueError(
+               f"Rotary embedding capacity ({max_position}) is smaller than "
+               f"max_model_length ({self.config['max_model_length']})"
+           )

+       # 分布式通信：若未指定 init_method，则让 OS 自动分配空闲端口
+       # 检查用户是否在 config 里手动指定了通信地址（比如 "tcp://192.168.1.1:29500"）
+       if "distributed_init_method" not in self.config:
+           # 创建一个 IPv4 的 TCP 套接字（上下文管理器确保用完自动关闭）
+           with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
+              # 绑定到 127.0.0.1（本地回环），端口号写 **0**
+               sock.bind(("127.0.0.1", 0))
+               # getsockname() 返回实际绑定的地址和端口，取索引 [1] 即端口号
+               port = sock.getsockname()[1]
+           # 把自动获取的端口拼成 PyTorch 分布式需要的 URL 格式
+           self.config["distributed_init_method"] = f"tcp://127.0.0.1:{port}"
+       # ?多 GPU 时生成唯一共享内存id，避免引擎间冲突
+       if world_size > 1:
+           self.config.setdefault(
+               "shared_memory_name", f"myvllm-{uuid.uuid4().hex}"
+           ) # 唯一id
        ctx = mp.get_context("spawn")
        self.processes = []
        self.events = []
        for i in range(1, world_size):
            event = ctx.Event()
            process = ctx.Process(target=worker_process, args=(self.config, i, event))
            self.events.append(event)
            self.processes.append(process)
            process.start()
        # start the engine only on the master thread with rank = 0
        self.model_runner = ModelRunner(self.config, rank=0, event=self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.get("model_name_or_path", "gpt2"))

+       # EOS Token 规范化：优先从 Tokenizer 获取，确保 Worker 间一致
+       configured_eos = self.tokenizer.eos_token_id
+       if configured_eos is None:
+           configured_eos = self.config.get("eos")
+       if configured_eos is None:
+           raise ValueError("Tokenizer/config must provide an EOS token ID")
+       self.config["eos"] = configured_eos
+
        # scheduler needs to init after model_runner: when world_size > 1,
        # ModelRunner.__init__ calls dist.init_process_group() which is a
        # collective barrier — rank-0 blocks until all worker ranks have joined.
        # The scheduler should only be created after that rendezvous completes.
        # When world_size == 1 there is no barrier and no real dependency.
        self.scheduler = Scheduler(
            max_num_sequences=self.config.get("max_num_sequences", 16),
            max_num_batched_tokens=self.config.get("max_num_batched_tokens", 1024),
            max_cached_blocks=self.config.get("max_cached_blocks", 1024),
            block_size=self.config.get("block_size", 256),
-           eos=self.config.get("eos", 50256),
+           eos=configured_eos if configured_eos is not None else self.tokenizer.eos_token_id,
            max_model_length=self.config["max_model_length"],
        )

        atexit.register(self.exit)
```

#### exit中间变量的妙用（调度问题的解决）

```python
    def exit(self):
-       self.model_runner.call("exit")
-       del self.model_runner

+       if getattr(self, "model_runner", None) is None:
+           return # 若已释放则直接返回
+       model_runner = self.model_runner
+       self.model_runner = None
+       model_runner.call("exit")
        for process in self.processes:
            process.join()
```

call 是一个“命令调度器”，负责在主进程（Rank 0）上执行方法，并将同一个命令同步到所有 Worker 进程（Rank 1..N）去执行
del 和 None的区别
	场景 A：使用 del
		del self.model_runner
		print(hasattr(self, "model_runner"))  # 输出: False（属性彻底没了）
		print(self.model_runner)             # 抛出 AttributeError
 	场景 B：赋值为 None
		self.model_runner = None
		print(hasattr(self, "model_runner"))  # 输出: True（属性还在）
		print(self.model_runner)             # 输出: None（不会报错）
如果先call("exit") 再等于None, 可能重复释放资源

#### step：修复Prefix Cache需要长度计算逻辑

``` python
-   def step(self) -> tuple[list[int], bool]:
+   def step(self) -> tuple[list[tuple[int, list[int]]], int, bool]:
        scheduled_sequences, is_prefill = self.scheduler.schedule()
        if not scheduled_sequences:
-           return [], is_prefill
+           return [], 0, is_prefill   # 统一返回三元组，避免解包错误
      
        outputs = self.model_runner.call("run", scheduled_sequences, is_prefill)
        if outputs is not None:
            outputs = outputs.cpu().tolist()
-		# 旧逻辑: 无论序列中有多少 token 已经命中前缀缓存，都按 len(seq) 全量计入
-       num_processed_tokens = 
			sum(len(seq) for seq in scheduled_sequences) if is_prefill 
			else len(scheduled_sequences)
+       # 新逻辑：总长度 - 缓存命中长度（仅 Prefill 时统计）
+       num_processed_tokens = (
+           sum(len(seq) - seq.num_cached_tokens for seq in scheduled_sequences)
+           if is_prefill else len(scheduled_sequences)
+       )
        # postprocess(后处理) the outputs
        self.scheduler.postprocess(scheduled_sequences, outputs)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in scheduled_sequences if seq.is_finished]
        return outputs, num_processed_tokens, is_prefill
```

一个批次里，既有完成（Finished）的请求，也有还没完成（Running）的请求。

| 序列   | 本次生成的 Token | 是否达到停止条件？ | 处理结果                                                     |
| :----- | :--------------- | :----------------- | :----------------------------------------------------------- |
| 序列 A | `42`             | 是（遇到了 EOS）   | `postprocess` 标记为 `Finished`，放入最终 `outputs` 返回给调用者。 |
| 序列 B | `99`             | 否（正常续写）     | `postprocess` 更新状态（记录新 token），**但**不放入 `outputs`（因为还没结束，它要继续留在调度器的 Running 队列里）。 |

#### add_prompt防御检查

``` python
    def add_prompt(self, prompt: str, sampling_params: SamplingParams) -> None:
-       self.scheduler.add_sequence(Sequence(token_ids=self.tokenizer.encode(prompt), block_size=self.config['block_size'],sampling_params=sampling_params))

+       token_ids = self.tokenizer.encode(prompt)
+       if not token_ids:
+           raise ValueError("Prompt must contain at least one token")
+       self.scheduler.add_sequence(Sequence(token_ids=token_ids, block_size=self.config['block_size'],sampling_params=sampling_params))
```
旧逻辑：直接将 prompt 编码为 token_ids，然后立即构造 Sequence 并加入调度器。没有任何防御检查。
新逻辑：
	先将 prompt 编码为 token_ids。
	显式检查 token_ids 是否为空列表（not token_ids）。
	如果为空，立即抛出 ValueError("Prompt must contain at least one token")，拒绝此请求。
	只有非空时才构造 Sequence 并加入调度器

``` python
-   def generate(self, prompts: list[str], sampling_params: SamplingParams) -> list[str]:
+	def generate(self, prompts: list[str], sampling_params: SamplingParams) -> dict[str, list]:
		...

       	output = {
           	'text': [
-				self.tokenizer.decode(tokens)
+               self.tokenizer.decode(tokens, skip_special_tokens=True)
               	for tokens in generated_tokens
          	],
           	'token_ids': generated_tokens,
      	}
        return output
```




## Prefix Cache 复用正确性

**涉及文件**

- `src/myvllm/engine/block_manager.py`
- `src/myvllm/models/qwen3.py`

**问题**

- 一个 Prefix Hash 只能映射到一个物理 Block。多个有效副本会互相覆盖，释放其中一个还可能删除另一个副本的映射。([原始标量 Hash Map](https://github.com/KevinWu8192/MyMinivLLM/blob/4d1760ded441988663c59f5bd9479a80de234731/src/myvllm/engine/block_manager.py#L24-L35)，[原始替换逻辑](https://github.com/KevinWu8192/MyMinivLLM/blob/4d1760ded441988663c59f5bd9479a80de234731/src/myvllm/engine/block_manager.py#L46-L63))
- Prefix Cache 命中后，Qwen3 RoPE Position 会从未缓存后缀的零位置重新开始，而不是从缓存 Context 后继续。([原始位置计算](https://github.com/KevinWu8192/MyMinivLLM/blob/4d1760ded441988663c59f5bd9479a80de234731/src/myvllm/models/qwen3.py#L209-L229))

**示例**

假设物理 Block 3 和 9 包含相同 Token Prefix，因此拥有同一个 Hash。标量 `hash -> block_id` 只能记住 Block 9；如果 Block 9 被回收，删除该映射也会使仍然有效的 Block 3 无法访问。对于 RoPE，如果缓存了 96 个 Token、后缀有 32 个 Token，其 Position 应为 96–127；若从 0–31 重新开始，Q/K 会像 Prefix 不存在一样被错误旋转。

**修复**

- [x] 将标量映射改为 `hash -> set[block_id]`。注册时添加一个物理副本；回收时只移除对应 Block ID，仅当 Set 为空时才删除 Hash Key。([Hash 索引](https://github.com/KevinWu8192/MyMinivLLM/blob/ee950af0b35f5ac87fa6a495238cf6ebb57aa99d/src/myvllm/engine/block_manager.py#L30-L57))
- [x] 对每个逻辑 Prefix Block 遍历预期 Hash 的所有物理候选项，通过 Token ID 校验防止 Hash Collision，选择有效 Block 时不破坏其他副本。([Prefix 选择](https://github.com/KevinWu8192/MyMinivLLM/blob/ee950af0b35f5ac87fa6a495238cf6ebb57aa99d/src/myvllm/engine/block_manager.py#L91-L116))
- [x] 对变长 Batch 中每个 Sequence 计算 `cached_len = full_kv_len - query_len`，为每个新 Query 重复该 Offset 并加上本地后缀位置，在保留扁平 Batch 表示的同时生成绝对 RoPE Position。([位置计算](https://github.com/KevinWu8192/MyMinivLLM/blob/ee950af0b35f5ac87fa6a495238cf6ebb57aa99d/src/myvllm/models/qwen3.py#L6-L20)，[使用位置](https://github.com/KevinWu8192/MyMinivLLM/blob/ee950af0b35f5ac87fa6a495238cf6ebb57aa99d/src/myvllm/models/qwen3.py#L230-L235))



```python

```

### hash -> set[block_id]

旧逻辑:hash->物理块，hash碰撞将旧的覆盖，一个 Prefix Hash 只能映射到一个物理 Block。多个有效副本会互相覆盖，释放其中一个还可能删除另一个副本的映射。
新逻辑:hash->多个物理块, hash碰撞

``` python
src/myvllm/engine/block_manager.py 

class BlockManager:
    def __init__(self, num_blocks: int, block_size: int):
        self.block_size: int = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
-      	self.hash_to_block_id: dict[int, int] = {}  #（哈希值 → 单个物理块 ID）
+       self.hash_to_block_ids: dict[int, set[int]] = {} #（哈希值 → 多个物理块 ID 的集合）
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()
        
    def compute_hash(self, token_ids: list[int], prefix_hash_value: int) -> int: ...

+   def _register_hash(self, hash_value: int, block_id: int) -> None:
+       if hash_value != -1:
+           self.hash_to_block_ids.setdefault(hash_value, set()).add(block_id)

+   def _unregister_hash(self, hash_value: int, block_id: int) -> None:
+       block_ids = self.hash_to_block_ids.get(hash_value)
+       if block_ids is None:
+           return
+       block_ids.discard(block_id)
+       if not block_ids:
+           del self.hash_to_block_ids[hash_value]

     # move this block to used list and update all information of the block
    def _allocate_new_block(self, block_id: int, hash: int, token_ids: list[int]) -> Block:
        block = self.blocks[block_id]
        assert block.ref_count == 0, "Block is already allocated"        old_hash = block.hash
        
-       if (
-			old_hash != -1 # 这个块之前被用过
-			and self.hash_to_block_id.get(old_hash) == block_id # 旧块和新块一样
-       ):
-      		del self.hash_to_block_id[old_hash]
+       self._unregister_hash(old_hash, block_id)
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        block.update(hash, token_ids)
        block.ref_count += 1
        return block

    def _allocate_hit_block(self, block_id: int) -> Block:
        block = self.blocks[block_id]
        assert block.ref_count == 0, "块已被分配"
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        block.ref_count += 1
        return block

    def _deallocate_block(self, block_id: int) -> None:
        assert self.blocks[block_id].ref_count == 0, "块仍在被使用"
        block = self.blocks[block_id]
        # block.token_ids = [] 
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def _get_prefix_hit_blocks(self, seq: Sequence) -> list[Block]:
        """返回 seq 的最长连续可复用前缀块列表。

        最后一个提示词 token 仍需计算才能产生 logits，因此故意排除该 token 所在的块。
        """
        h = -1
        hit_blocks = []
        max_cacheable_blocks = max(0, (seq.num_tokens - 1) // self.block_size)

        for i in range(max_cacheable_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
-           # 旧代码：block_id = self.hash_to_block_id.get(h)
-           # if block_id is None: break
-           # block = self.blocks[block_id]
-           # if block.token_ids != token_ids: break   # 哈希碰撞
+           block_ids = self.hash_to_block_ids.get(h)
+           if not block_ids:
+               break
+           matching_blocks = [
+               self.blocks[block_id]
+               for block_id in block_ids
+               if self.blocks[block_id].token_ids == token_ids
+           ]
+           if not matching_blocks:   # 哈希碰撞（无候选匹配）
+               break
+           # 优先复用正在使用的块（无需额外消耗空闲块）
+           block = max(matching_blocks, key=lambda candidate: candidate.ref_count > 0)
            hit_blocks.append(block)

        return hit_blocks

    def can_allocate(self, seq: Sequence) -> bool: ...

+   def get_num_cached_tokens(self, seq: Sequence) -> int:
+       """返回可复用的前缀长度，不修改序列状态。"""
+       return len(self._get_prefix_hit_blocks(seq)) * self.block_size

    def allocate(self, seq: Sequence) -> None:
        hit_blocks = self._get_prefix_hit_blocks(seq)
        h = -1
        for i in range(seq.num_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids=token_ids, prefix_hash_value=h) if len(token_ids) == self.block_size else -1
            if i < len(hit_blocks):
                block = hit_blocks[i]
                seq.num_cached_tokens += self.block_size
                if block.ref_count == 0:
                    self._allocate_hit_block(block.block_id)
                else:
                    block.ref_count += 1
                seq.block_table.append(block.block_id)
                continue
            block = self._allocate_new_block(self.free_block_ids[0], h, token_ids)
-           # 旧代码：if h != -1: self.hash_to_block_id[h] = block.block_id
+           self._register_hash(h, block.block_id)
            seq.block_table.append(block.block_id)

    def deallocate(self, seq: Sequence) -> None: ...

    def can_append(self, seq: Sequence) -> bool: ...

    # 实际执行追加 token 的块分配（新 token 已加入 seq，但尚未分配 GPU 块）
    def append(self, seq: Sequence) -> None:
        block_tables = seq.block_table
        last_block_for_seq_id = block_tables[-1]
        token_ids = seq.block(seq.num_blocks - 1)

        # 如果最后一个块已满，则计算哈希并注册
        if seq.num_tokens % self.block_size == 0:
            h = self.compute_hash(token_ids = token_ids,
                                  prefix_hash_value = -1 if len(block_tables) == 1 else self.blocks[block_tables[-2]].hash)
            block = self.blocks[last_block_for_seq_id]
            block.update(h=h, token_ids=seq.block(seq.num_blocks - 1))
-           # 旧代码：self.hash_to_block_id[h] = block.block_id
+           self._register_hash(h, block.block_id)
        # 如果需要一个新块
        elif seq.num_tokens % self.block_size == 1:
            # 前一个块应该已经完成（哈希不为 -1）
            assert self.blocks[last_block_for_seq_id].hash != -1
            block = self._allocate_new_block(self.free_block_ids[0], -1, token_ids)
            block_tables.append(block.block_id)
        # 否则无需操作
        else:
            assert last_block_for_seq_id in self.used_block_ids, "最后一个块应已分配"
            assert self.blocks[last_block_for_seq_id].hash == -1, "最后一个块应为部分块，哈希为 -1"
```

### RoPE Position(支持 Prefix Cache)

```python
src/myvllm/models/qwen3.py

from myvllm.layers import *
import torch 
import torch.nn as nn

# ========== 新增函数：计算 RoPE positions，支持 Prefix Cache ==========
+ def get_qwen_positions(context, device: torch.device, num_tokens: int) -> torch.Tensor:
+     # Prefill且部分前缀已缓存
+     if context.is_prefill and context.cu_seqlens_q is not None:
+         cu_q = context.cu_seqlens_q
+         cu_k = context.cu_seqlens_k

+         query_lens = cu_q[1:] - cu_q[:-1]
+         cached_lens = (cu_k[1:] - cu_k[:-1]) - query_lens

+         query_starts = torch.repeat_interleave(cu_q[:-1], query_lens)
+         # 为每个 token 生成其所在序列的缓存长度（重复）
+         cached_starts = torch.repeat_interleave(cached_lens, query_lens)
+         # 位置 = 全局索引 - 查询起始 + 缓存起始
+         return (
+             torch.arange(num_tokens, device=device)
+             - query_starts.to(device=device)
+             + cached_starts.to(device=device)
+         ).long()
+     # 普通 prefill（单序列或无缓存）
+     if context.is_prefill:
+         return torch.arange(num_tokens, device=device)
+     # decode 阶段：当前位置为 context_lens - 1
+     return context.context_lens - 1

"""
	A缓存96查询32 B缓存50查询20
	batch = 32 + 20 = 52
	cu_q = [0, 32, 52]
	cu_k = [0, 128(96+32), 198(128+50+20)]
	query_lens 
		= cu_q[1:] - cu_q[:-1] 
		= [32, 52] -  [0, 32]
		= [32, 20]
	cached_lens 
		= (cu_k[1:] - cu_k[:-1]) - query_lens
		= [128, 198] - [0, 128]
		= [128, 70]
	query_starts = torch.repeat_interleave(cu_q[:-1], query_lens)
		= [0,0,...,0, 32,32,...,32]（前32个0，后20个32）
	cached_starts = torch.repeat_interleave(cached_lens, query_lens)
		= [96,96,...,96, 50,50,...,50]（前32个96，后20个50）
	return = (
			torch.arange(num_tokens, device=device)
           	- query_starts.to(device=device)
           	+ cached_starts.to(device=device)
         	).long()
		= [96, 97, 98, ..., 127, 50, 51, ..., 69]
"""

class Qwen3DecoderLayer(nn.Module):

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None) -> torch.Tensor:
        if residual is not None:
            x, residual = self.input_layernorm(x, residual)
        else:
            residual = x  # 归一化前保存残差
            x = self.input_layernorm(x)
        # 根据上下文计算位置（支持变长 prefill 的序列边界）
        from myvllm.utils import get_context
        context = get_context()
-       if context.is_prefill and context.cu_seqlens_q is not None:
-           # 变长 prefill：为每个序列从 0 开始创建位置
-           positions = []
-           cu_seqlens = context.cu_seqlens_q.cpu().tolist()
-           for i in range(len(cu_seqlens) - 1):
-               seq_len = cu_seqlens[i+1] - cu_seqlens[i]
-               positions.extend(range(seq_len))
-           positions = torch.tensor(positions, dtype=torch.long, device=x.device)
-       elif context.is_prefill:
-           # 单序列 prefill：使用顺序位置
-           positions = torch.arange(x.size(0), device=x.device)
-       else:
-           # decode 阶段：使用 context_lens - 1 作为当前位置
-           positions = context.context_lens - 1
+       # Prefix Cache 命中后，RoPE 位置必须反映完整序列中的绝对位置，而非从 0 重启。
+       positions = get_qwen_positions(context, x.device, x.size(0))

        x = self.self_attn(x, positions=positions)
        # 注意力输出始终进行残差连接
        x, residual = self.post_attention_layernorm(x, residual)
        x = self.mlp(x)
        return x, residual
```

## Prefill Budget 与调度公平性


```python
‎src/myvllm/engine/scheduler.py

class Scheduler:
    def __init__(self, max_num_sequences: int, max_num_batched_tokens: int,
                 max_cached_blocks: int, block_size: int, eos: int,
                 max_model_length: int):
		...
+       预填充（Prefill）和解码（Decode）使用不同的计算内核。
+		当两个队列中都有待处理任务时，应交替执行，这样新到达的提示词请求和解码请求都不会被饿死
+       self._last_scheduled_prefill = False

	def is_finished(self):	...
	def add_sequence(self, sequence: Sequence): ...
```

**问题：**

- Prefill Token Budget 会计算无需重新计算的缓存 Token

- 单个未缓存 Prompt 大于 Batch Budget 时，由于尚未实现 Chunked Prefill，请求可能永远阻塞

> 缓存 Token Budget： Prompt 有 128 个 Token，Prefix Cache Hit 为 96 个 Token，且 max_num_batched_tokens = 32 时，实际只需计算 32 个 Token。旧调度器却收取全部 128 个 Token 的 Budget，错误拒绝了一个恰好能够执行的请求。

- 持续到来的 Prefill 请求可能使已经进入 Decode 的请求饥饿：

> ​	假设 Decode 请求 D 位于 `running`，而每轮调度前都有一个新 Prefill 请求到达。旧调度器总是先尝试 `waiting`，只要成功调度任何 Prefill 就立即返回，因此即使 Decode Work 已经就绪，D 仍可能无限等待

**修复**

- [x] 分配前先从 BlockManager 获取预期 Prefix Hit，在准入阶段只收取 `len(sequence) - cached_tokens`；分配后使用 Sequence 记录的 `num_cached_tokens` 累加当前 Batch Budget，使预测值与实际记账一致。
- [x] 对不支持的超长 Prefill 给出可操作的错误，而不是无限期留在队列中。([超长 Prefill](https://github.com/KevinWu8192/MyMinivLLM/blob/ee950af0b35f5ac87fa6a495238cf6ebb57aa99d/src/myvllm/engine/scheduler.py#L69-L77))
- [x] 记录上一次成功调度是否为 Prefill。当两个队列都有任务时，Prefill Pass 后跳过一次 Prefill，执行 Decode，再清除状态，使下一轮可以继续 Prefill。这样既限制 Decode 等待时间，也不会在没有 Running Request 时阻止 Prefill。([交替状态](https://github.com/KevinWu8192/MyMinivLLM/blob/ee950af0b35f5ac87fa6a495238cf6ebb57aa99d/src/myvllm/engine/scheduler.py#L29-L31)，[状态转换](https://github.com/KevinWu8192/MyMinivLLM/blob/ee950af0b35f5ac87fa6a495238cf6ebb57aa99d/src/myvllm/engine/scheduler.py#L61-L114))
- [ ] 验证 Sampler 为每个已调度 Sequence 返回且仅返回一个 Token。([输出验证](https://github.com/KevinWu8192/MyMinivLLM/blob/ee950af0b35f5ac87fa6a495238cf6ebb57aa99d/src/myvllm/engine/scheduler.py#L125-L132))

```python
    def schedule(self) -> tuple[list[Sequence], bool]:
        scheduled_sequences = []
        current_scheduled_tokens = 0

+       # 修复3: 只有在未设置“上一轮为 Prefill”或 running 为空时才尝试 Prefill
+       try_prefill = self.waiting and (
+           not self.running or not self._last_scheduled_prefill
+       )
-       # while self.waiting and len(scheduled_sequences) < self.max_num_sequences:
+       while try_prefill and self.waiting and len(scheduled_sequences) < self.max_num_sequences:

			seq = self.waiting[0]

+           # 修复1: 计算实际所需 Prefill token 数（扣除缓存命中）
+           num_prefill_tokens = (
+               len(seq) - self.block_manager.get_num_cached_tokens(seq)
+           )

+           # 修复2: 若单个请求所需 token 超过 max_num_batched_tokens，直接报错（不支持分块）
+           if num_prefill_tokens > self.max_num_batched_tokens:
+               raise ValueError(
+                   f"Request needs {num_prefill_tokens} prefill tokens, but "
+                   f"max_num_batched_tokens is {self.max_num_batched_tokens}; "
+                   "chunked prefill is not supported"
+               )

+           if self.block_manager.can_allocate(seq) and num_prefill_tokens + current_scheduled_tokens <= self.max_num_batched_tokens:
-           # if self.block_manager.can_allocate(seq) and len(seq) + current_scheduled_tokens <= self.max_num_batched_tokens:			

                seq = self.waiting.popleft() # remove from waiting
                self.block_manager.allocate(seq)
                seq.status = SequenceStatus.RUNNING
                self.running.append(seq)
                scheduled_sequences.append(seq)
                
-               # current_scheduled_tokens += len(seq)
+               current_scheduled_tokens += len(seq) - seq.num_cached_tokens   # 仅累加实际计算部分
```

``` python
            else:
                break
        if scheduled_sequences:
+           self._last_scheduled_prefill = True # 修复3
            return scheduled_sequences, True
        
        # 尝试从运行队列调度 Decode
        while self.running:
            seq = self.running.popleft()
            # 检查能否追加一个 token
            if not self.block_manager.can_append(seq):
                if self.running:
                    self.running.appendleft(seq)
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                if current_scheduled_tokens >= self.max_num_batched_tokens or len(scheduled_sequences) >= self.max_num_sequences:
                    self.running.appendleft(seq)
                    break
                # 追加一个 token
                self.block_manager.append(seq)
                scheduled_sequences.append(seq)
                current_scheduled_tokens += 1  # Decode 每次只生成一个 token

        # 将调度到的 Decode 序列按原顺序放回 running 队列
        if scheduled_sequences:
            self.running.extendleft(reversed(scheduled_sequences))
+           self._last_scheduled_prefill = False # 修复3

        return scheduled_sequences, False

    def preempt(self, seq: Sequence) -> None: # seq被抢占
        self.block_manager.deallocate(seq)
        seq.status = SequenceStatus.WAITING
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> None:
+       # 修复4: 验证采样器返回的 token 数量与序列数量一致
+       if token_ids is None or len(seqs) != len(token_ids):
+           actual = 0 if token_ids is None else len(token_ids)
+           raise RuntimeError(
+               f"Sampler returned {actual} tokens for {len(seqs)} sequences"
+           )
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            # 检查停止条件：EOS、达到最大生成长度、达到最大模型长度
            stop_due_to_eos = not seq.ignore_eos and token_id == self.eos
            stop_due_to_max_tokens = seq.num_completion_tokens >= seq.max_tokens
            stop_due_to_max_length = seq.max_model_length is not None and seq.num_tokens >= seq.max_model_length

            if stop_due_to_eos or stop_due_to_max_tokens or stop_due_to_max_length:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)	
	
```



## 数值稳定性与模型配置

- 使用rsqrt而不是1/sqrt
- fp16/bf16 -> fp32 -> fp16/bf16

```
‎src/myvllm/layers/linear.py

class QKVColumnParallelLinear(ColumnParallelLinear):
    def __init__(...):
+		if num_heads % self.tp_size != 0:
+           raise ValueError("num_heads must be divisible by tensor parallel size")
+       if num_kv_heads % self.tp_size != 0:
+           raise ValueError("num_kv_heads must be divisible by tensor parallel size")

‎src/myvllm/layers/sampler.py

    @torch.compile
    def forward(self, logits: torch.Tensor, temperature: torch.Tensor) -> torch.Tensor:
        # Sampling probabilities should be computed in FP32 even when model
        # weights and logits use BF16/FP16.
+       logits = logits.float() / temperature.float().unsqueeze(-1)
-       logits/= temperature.unsqueeze(-1)

        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        return sample_tokens

‎src/myvllm/models/llama.py

-   LayerNorm(gamma)
+	LayerNorm(gamma, eps=rms_norm_epsilon)

‎src/myvllm/sampling_parameters.py
    def __post_init__(self):
        # 旧版：使用 assert 校验，可被 python -O 禁用
-       assert self.temperature > 1e-10, "greedy sampling is not permitted"
        # 新版：使用 raise ValueError，校验始终生效
+       if self.temperature <= 1e-10:
+           raise ValueError("temperature must be greater than 0")
        # 新增：max_tokens 必须为正整数
+       if self.max_tokens <= 0:
+           raise ValueError("max_tokens must be greater than 0")
        # 新增：max_model_length 若指定，必须为正整数
+       if self.max_model_length is not None and self.max_model_length <= 0:
+           raise ValueError("max_model_length must be greater than 0")
```

```
‎src/myvllm/layers/layernorm.py

    @torch.compile
    def rms_forward(self, x: torch.Tensor) -> torch.Tensor:
        # RMSNorm(x) = (x / sqrt(mean(x²) + ε)) ⊙ γ
    -   variance = x.pow(2).mean(dim=-1, keepdim=True) + self.eps
    -   sqrt_variance = variance.sqrt()
    -   x_norm = (x / sqrt_variance * self.weight)
    -
    -   return x_norm
    +   input_dtype = x.dtype                    # 记录原始 dtype（fp16/bf16）
    +   x = x.float()                            # 转 fp32 计算，防止精度丢失
    +   variance = x.pow(2).mean(dim=-1, keepdim=True) + self.eps  # fp32 下求方差
    +   x = x * torch.rsqrt(variance)        # 用 rsqrt 一次性归一化
    +   return (x * self.weight.float()).to(input_dtype)  # 权重也用 fp32，最后转回原 dtype
```

## **捕获 CUDA Graph（Decode 加速）**

```python
class ModelRunner:
    def __init__(self, config: dict, rank: int, event: Event | list[Event]):
		...
       	if not self.enforce_eager:
            self.capture_cudagraph()
		...
```


```python
	# 捕获 CUDA Graph（Decode 加速）
    @torch.inference_mode()
    def capture_cudagraph(self) -> None:
		----- 1. 尺寸计算与静态缓冲区分配 -----
    	max_bs = self.config['max_num_sequences']       # 最大并发序列数（batch size 上限）
    	max_len = self.config['max_model_length']       # 单个序列最大长度（token 数上限）
    	max_num_blocks = math.ceil(max_len / self.block_size)  # 单个序列最多占多少个物理块

        # 静态缓冲分配（注意 outputs 形状为 [max_bs, hidden_size]）
        input_ids = torch.zeros(max_bs, dtype=torch.long, device=f'cuda:{self.rank}') # 当前步输入的 token ID
        slot_mapping = torch.zeros(max_bs, dtype=torch.long, device=f'cuda:{self.rank}') # 每个新 token 的 Key/Value 需要写入 KV Cache 的具体物理位置
        context_lens = torch.zeros(max_bs, dtype=torch.long, device=f'cuda:{self.rank}') # 每个序列当前的总长度
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32, device=f'cuda:{self.rank}') # 每个序列的物理块映射表
+       # 修复：outputs 应为 hidden_states 的形状 [max_bs, hidden_size]，而非 vocab_size
+       outputs = torch.zeros(
+           max_bs,
+           self.config['hidden_size'],
+           device=f'cuda:{self.rank}',
+           dtype=self.default_dtype,
+       )

       	----- 2.生成合法的 batch size 列表（去重、过滤）-----
+       batch_sizes = sorted({
+           bs for bs in [1, 2, 4, 8, max_bs, *range(16, max_bs + 1, 16)] # 从 16 开始，每 16 取一个值，直到 max_bs
+           if bs <= max_bs
+       })

		----- 3. 遍历捕获循环（核心） -----
		# 初始化容器
        self.graphs = {} # 空字典
        graph_pool = None # 内存池

        for batch_size in reversed(batch_sizes): # 从大到小遍历 batch sizes
            graph = torch.cuda.CUDAGraph() # 创建 CUDAGraph 对象
            set_context(
                is_prefill=False,                # 当前是 Decode 阶段
                cu_seqlens_q=None,               # Decode 不用变长累积
                cu_seqlens_k=None,
                max_seqlen_q=0,
                max_seqlen_k=0,
                slot_mapping=slot_mapping[:batch_size],   # 传入静态缓冲的切片
                context_lens=context_lens[:batch_size],
                block_tables=block_tables[:batch_size],
            ) # 设置上下文

+           outputs[:batch_size] = self.model(input_ids[:batch_size]) # Warmup：在 capture 前先执行一次模型

            with torch.cuda.graph(graph, graph_pool): # 进入捕获块
                outputs[:batch_size] = self.model(input_ids[:batch_size])
            # =============================================================================
            # 进入 CUDA Graph 捕获块后发生的事：
            #   - torch.cuda.graph(graph, graph_pool) 是 CUDA Graph 的上下文管理器。
            #   - 进入 with 块后，所有 CUDA 操作不再立即执行，而是被录制到 graph 中。
            #   - 块内的 self.model(...) 调用会被完整录制，包括：
            #       * Embedding 查表
            #       * 每一层的 Attention、MLP、Norm
            #       * RoPE 旋转
            #       * KV Cache 写入
            #       * 输出投影
            #   - 关键：录制时 graph_pool 会被用作显存池。
            #       第一次传 None，CUDA 自动创建；
            #       之后传已存在的池，复用同一块显存。
            #
            # 为什么用静态缓冲？
            #   - CUDA Graph 要求所有输入输出的内存地址固定。
            #   - 所以必须用预先分配好的 input_ids、outputs 等，直接原地写入/读出。
            #
            # 回放流程：
            #   - 把真实数据 copy_ 进 input_ids；
            #   - 调用 graph.replay()；
            #   - 输出自动出现在 outputs 中。
            # =============================================================================


+           # 在 with 块外获取 graph_pool，避免在 capture 前访问 pool
+           if graph_pool is None:
+               graph_pool = graph.pool()
            self.graphs[batch_size] = graph # 保存图到字典
			
            torch.cuda.synchronize() # 等待当前 batch size 的图捕获完全结束
            reset_context() # 清空全局上下文

        self.graph_vars = dict(
            input_ids=input_ids,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        ) # 将所有静态缓冲区保存到 self.graph_vars 字典中，供后续回放时填充实际数据。
```

```python
    def run_model(self, input_ids: torch.Tensor, is_prefill: bool) -> torch.Tensor:
        if is_prefill or self.enforce_eager:
            # For varlen prefill, keep input_ids as 1D (concatenated tokens)
            # Do NOT unsqueeze - flash_attn_varlen_func expects 1D input with cu_seqlens
            hidden_states = self.model(input_ids)
            logits = self.model.compute_logits(hidden_states)
        else:
			# Step 1：找到适配的 graph
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[next(bs_ for bs_ in self.graphs.keys() if bs_ >= bs)] # 最小能装下的graph
            vars = self.graph_vars # 字典

			# Step 2：重置静态缓冲区的 padding(填充)区（关键！）
			# vars['input_ids'][:captured_bs].zero_()
            vars['slot_mapping'][:captured_bs].fill_(-1)
            vars['context_lens'][:captured_bs].zero_()
            # vars['block_tables'][:captured_bs].fill_(-1)

			# Step 3：把真实数据拷贝到静态缓冲区
            vars['input_ids'][:bs].copy_(input_ids)
            vars['slot_mapping'][:bs].copy_(context.slot_mapping)
            vars['context_lens'][:bs].copy_(context.context_lens)
            vars['block_tables'][
                :bs, :context.block_tables.size(1)
            ].copy_(context.block_tables)

            # Step 4：Replay 并取结果
            graph.replay() # 用录制好的 kernel 序列，以固定的 grid/block 维度和固定的地址执行整个模型 forward
            logits = self.model.compute_logits(vars['outputs'][:bs])

        return logits
```

| 缓冲区         | 残留会怎样                                             | 必须重置吗？ |
| :------------- | :----------------------------------------------------- | :----------- |
| `slot_mapping` | **致命**：padded token 会污染真实 KV Cache             | ✅ 必须       |
| `context_lens` | **危险**：残留长度会让 kernel 读到错误的历史范围       | ✅ 必须       |
| `input_ids`    | 无害：padded token 的 embedding 是垃圾，但输出会被丢弃 | ❌ 可省       |
| `block_tables` | 无害：配合 `context_lens=0`，根本不会被访问            | ❌ 可省       |

## 验证

```python
 tests/test_engine_boundaries.py  
 tests/test_scheduler.py          
```

