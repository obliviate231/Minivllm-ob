# R1

```python
BlockManager：
 	1. def __init__(self, num_blocks: int, block_size: int):
 	2. def can_allocate(self, seq: Sequence) -> bool:
 	3. def allocate(self, seq: Sequence) -> None:
 	4. def deallocate(self, seq: Sequence) -> None:
 	5. def can_append(self, seq: Sequence) -> bool:
 	6. def append(self, seq: Sequence) -> None:
```

## Q1：请求释放时会清空 Block 的 Token 元数据，导致后续请求无法复用已经缓存的 Prefix

假设 `block_size = 4`，请求 A 的 Token 是 `[10, 11, 12, 13, 20]`，第一个 Block `[10, 11, 12, 13]` 可以作为 Prefix 复用。A 完成后，请求 B 携带 `[10, 11, 12, 13, 30]` 到达。旧释放逻辑会清空保存的 Token ID，因此 B 要么 Cache Miss，要么把残留的 Hash 映射判断为 Collision；即使找到了物理 Block，通用分配路径也会在复用前将其重置。

```python
原始逻辑
seqA来，保存块a，块a只能被hash到block_id1，
此时block_id1.token_ids=[]，cache miss, 修改block_id1.token_ids = [10, 11, 12, 13]
seqA走，释放块a, 修改block_id1.token_ids = []
seqB来，保存块a，块a只能被hash到block_id1，
此时block_id1.token_ids=[]，cache miss ×
...

修改后：
seqA来，保存块a，块a只能被hash到block_id1，
此时block_id1.token_ids=[]，cache miss, 修改block_id1.token_ids = [10, 11, 12, 13]
seqA走，释放块a, # 不修改block_id1.token_ids = []
seqB来，保存块a，块a只能被hash到block_id1，
此时block_id1.token_ids=[]，cache found ✔
```

## Q2：Cache Hit 和新 Block 使用同一套分配流程，在复用命中结果前就会重置仍然有效的缓存元数据

```python
原始逻辑
当一个空闲块（ref_count == 0 但 token_ids 和 hash 仍然有效）被命中时，
原代码仍然调用 _allocate_block，
而该函数内部会执行 block.reset()，将 hash 置为 -1、ref_count 置为 0、token_ids 清空。
结果就是：虽然找到了正确的物理块，但在复用之前它的缓存元数据已经被抹掉了，缓存实际上没有被利用。

# 新增两个独立的分配函数
    def _allocate_new_block(self, block_id: int, hash: int, token_ids: list[int]) -> Block:
        block = self.blocks[block_id]
        assert block.ref_count == 0, "Block is already allocated"
        # 防止出现old_hash已经映射到其他的block上了导致误删
        # 上述情况在prefix未连续命中会发生（如prefix的前面一部分被释放了，但是后面还有没被释放的）
        # 因此后面没释放的也不能命中，要新allocate block，但是这些block会计算出hash map中已有的hash值，改变block id指向
        old_hash = block.hash
        if (
            old_hash != -1
            and self.hash_to_block_id.get(old_hash) == block_id
        ):
            del self.hash_to_block_id[old_hash]
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        block.update(hash, token_ids)
        block.ref_count += 1
        return block

    def _allocate_hit_block(self, block_id: int) -> Block:
        # 只增加引用计数，不修改任何缓存内容
        block = self.blocks[block_id]
        assert block.ref_count == 0, "Block is already allocated"
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        block.ref_count += 1
        return block

在 allocate 中：

    if i < len(hit_blocks):
        block = hit_blocks[i]
        if block.ref_count == 0:
            self._allocate_hit_block(block.block_id)   # 保留缓存
        else:
            block.ref_count += 1
        seq.block_table.append(block.block_id)
        continue

效果：
命中空闲块时调用 _allocate_hit_block，它不会调用 reset()，因此块中的 token_ids、hash 全部保留，真正实现了前缀缓存的复用。
```

## Q3: Prefix Miss 后仍可能继续匹配后续 Block，但 Prefix Cache 只能复用连续前缀

```python
原始逻辑
在 allocate 循环中逐个块独立判断是否命中：
    for i in range(seq.num_blocks):
        token_ids = seq.block(i)
        h = self.compute_hash(...)
        block_id = self.hash_to_block_id.get(h, -1)
        if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
            no_cache_found = True

        if not no_cache_found:
            # 命中，加入 block_table
            ...
        else:
            # 未命中，分配新块
        ...

即使第 `i` 个块未命中（`no_cache_found = True`），循环仍会继续处理第 `i+1` 个块。如果第 `i+1` 个块恰好命中，它也会被当作缓存复用。
此时直接复用第 1 块的 KV cache 会导致缓存不连续，模型计算会出错（因为第 1 块的 hidden states 依赖于第 0 块的输出，而第 0 块没有被正确缓存）。

修改后：

新增 `_get_prefix_hit_blocks`，提前计算最长连续命中前缀：

    def _get_prefix_hit_blocks(self, seq):
        h = -1
        hit_blocks = []
        max_cacheable_blocks = max(0, (seq.num_tokens - 1) // self.block_size) # 最后一块满了也不能作为纯前缀缓存
        for i in range(max_cacheable_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id.get(h)
            if block_id is None:
                break          # 一旦 miss，立即停止
            block = self.blocks[block_id]
            if block.token_ids != token_ids:
                break          # 碰撞也停止
            hit_blocks.append(block)
        return hit_blocks

在 `allocate` 中：
    hit_blocks = self._get_prefix_hit_blocks(seq)
    for i in range(seq.num_blocks):
        if i < len(hit_blocks):
            # 命中分支（hit_blocks 保证连续）
            ...
        else:
            # 未命中分支，所有后续块都走这里
            block = self._allocate_new_block(...)
```

## Q4：最后一个 Block 恰好填满时，其 Token 数量会被错误地计算为零

```python
    def last_block_num_tokens(self):
        full_blocks = int(math.floor(self.num_tokens / self.block_size))
        return len(self.token_ids[full_blocks * self.block_size : ])

	 @property
    def last_block_num_tokens(self):
        if self.num_tokens == 0:
            return 0
        rest_tokens_num = self.num_tokens % self.block_size
        if rest_tokens_num == 0:
            return self.block_size
        else:
            return rest_tokens_num
```

## Q5：can_allocate会将可以装下的seq返回false

```python
    # 原来的没有考虑prxfix cache不占块
    def can_allocate(self, seq: Sequence) -> bool:		
        return len(self.free_block_ids) >= seq.num_blocks

	# 修改后考虑了
    def can_allocate(self, seq: Sequence) -> bool:
        hit_blocks = self._get_prefix_hit_blocks(seq)

        # 命中一个正在使用的块只会增加其 ref_count，因此不需要空闲块。
        # 如果命中的块 ref_count == 0，它仍在空闲队列中；分配该请求会将它从空闲队列中移除，
        # 所以它必须仍计入容量检查中。
        num_shared_hit_blocks = sum(block.ref_count > 0 for block in hit_blocks)
        num_required_free_blocks = seq.num_blocks - num_shared_hit_blocks
        return len(self.free_block_ids) >= num_required_free_blocks
```



