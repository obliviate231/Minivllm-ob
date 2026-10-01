# R3

```python
def flash_attention_varlen_kernel(
    Q, K, V, O,
    cu_seqlens_q_ptr,
----- 1. 函数签名新增参数 ------
+    cu_seqlens_k_ptr, # 累积长度, 每个序列在 K/V 张量中的起始和结束位置
+    k_cache_ptr,
+    v_cache_ptr, # 分页 KV 缓存张量的指针，供分页模式读取。
+    block_tables_ptr,
    scale,
    num_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
+    block_size: tl.constexpr, # 每个物理块包含的 token 数（编译时常量）
+    max_num_blocks: tl.constexpr, # 每个序列最多可分配的物理块数（编译时常量）
+    USE_PAGED_KV: tl.constexpr, # 布尔常量，控制是否启用分页缓存读取路径。
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
```

```python
----- 2.序列长度计算改为分离 Q 和 K/V ----- 
    # Program IDs
    start_m = tl.program_id(0) # block index
    off_h = tl.program_id(1) # head index
    seq_idx = tl.program_id(2) # sequence index

    # Determine which KV head to use (for GQA)
    kv_head_idx = off_h // (num_heads // num_kv_heads)

    # Load sequence boundaries
    seq_start = tl.load(cu_seqlens_q_ptr + seq_idx)
    seq_end = tl.load(cu_seqlens_q_ptr + seq_idx + 1)
-    seq_len = seq_end - seq_start
+    seq_len_q = seq_end - seq_start # 

+    seq_start_k = tl.load(cu_seqlens_k_ptr + seq_idx)
+    seq_end_k = tl.load(cu_seqlens_k_ptr + seq_idx + 1)
+    seq_len_k = seq_end_k - seq_start_k

+    # Prefix-cache hits remove full prefix blocks from Q, while K/V still
+    # represent the full sequence through the paged KV cache.
+    num_cached_tokens = seq_len_k - seq_len_q 
	# 已经缓存在 KV 中的前缀 token 数量 = 
	#	seq_len_k（完整 KV 上下文的长度）
	#	- seq_len_q（当前查询块的长度）

"""
eg:
	系统提示词（缓存前缀）："You are a helpful AI." → 假设编码后占 5 个 token（位置 0~4）。
	用户本轮新输入："What is AI?" → 假设编码后占 3 个 token（位置 5~7）。
	由于系统提示词命中缓存（之前算过，存进 KV Cache 了），
	系统为了省算力，只把新输入的 3 个 token 传进来当 Q。
	seq_len_q = 3（本次只有 3 个新 token 需要计算 Query）
	seq_len_k = 8（完整的上下文长度：5个缓存 + 3个新 token）
	num_cached_tokens = 8 - 3 = 5（有 5 个 token 是缓存的）
"""
-    if start_m * BLOCK_M >= seq_len:
+    if start_m * BLOCK_M >= seq_len_q:
        return
# 当前这个 GPU 线程块负责处理的 Query token 的起始位置就是 start_m * BLOCK_M

    # Offset for this block of queries
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, head_dim)

    # Query pointers: Q has shape (total_tokens, num_heads, head_dim)
    q_ptrs = Q + (seq_start + offs_m[:, None]) * num_heads * head_dim + off_h * head_dim + offs_d[None, :]

    # Load Q block - shape (BLOCK_M, head_dim)
-    mask_m = offs_m < seq_len
+    mask_m = offs_m < seq_len_q

	q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)

    # Initialize output accumulators
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - 1e10
    acc = tl.zeros([BLOCK_M, head_dim], dtype=tl.float32)

    # Number of blocks to process
-    num_blocks = tl.cdiv(seq_len, BLOCK_N)
+    num_blocks = tl.cdiv(seq_len_k, BLOCK_N)

	# Loop over K, V blocks
    for block_n in range(num_blocks):
        start_n = block_n * BLOCK_N
        offs_n = start_n + tl.arange(0, BLOCK_N)

        # Mask for valid positions
-    	mask_n = offs_n < seq_len
+    	mask_n = offs_n < seq_len_k
```

```python
----- 3. 重写 K 加载部分，增加分页分支 ----- 
-    	k_ptrs = 
			K + (seq_start + offs_n[None, :]) * num_kv_heads * head_dim 
			+ kv_head_idx * head_dim 
			+ offs_d[:, None]
+       if USE_PAGED_KV:
+            # Prefix hit: current K/V were stored into the paged cache before
+            # this kernel launch, so both the cached prefix and new suffix can
+            # be read through the request's block table.
+            logical_block_idx = offs_n // block_size # 逻辑块号
+            block_offset = offs_n % block_size
+            block_ids = tl.load(
+                block_tables_ptr
+                + seq_idx * max_num_blocks
+                + logical_block_idx,
+                mask=mask_n,
+                other=0,
+            )
+            cache_offsets = (
+                block_ids * block_size * num_kv_heads * head_dim
+                + block_offset * num_kv_heads * head_dim
+                + kv_head_idx * head_dim
+            )
+            k_ptrs = k_cache_ptr + cache_offsets[None, :] + offs_d[:, None]
+       else:
+            # No prefix hit: K is contiguous and has the same sequence layout
+            # as Q.
+            k_ptrs = K + (seq_start + offs_n[None, :]) * num_kv_heads * head_dim + kv_head_idx * head_dim + offs_d[:, None]

		# Load K block - shape (head_dim, BLOCK_N)
        k = tl.load(k_ptrs, mask=mask_n[None, :], other=0.0)

        # Compute QK^T - shape (BLOCK_M, BLOCK_N)
        qk = tl.dot(q, k)
        qk = qk * scale

        # Apply causal mask: only attend to positions <= current position
-    	v_ptrs = V + (seq_start + offs_n[:, None]) * num_kv_heads * head_dim + kv_head_idx * head_dim + offs_d[None, :]		
+        if USE_PAGED_KV:
+            v_ptrs = v_cache_ptr + cache_offsets[:, None] + offs_d[None, :]
+        else:
+            v_ptrs = V + (seq_start + offs_n[:, None]) * num_kv_heads * head_dim + kv_head_idx * head_dim + offs_d[None, :]
"""
    k_cache(num_blocks, block_size：4, num_kv_heads：2, head_dim：8)
    当前要加载的 K 块是第 1 个逻辑块（也就是 4~7 号 token），所以 offs_n = [4, 5, 6, 7]
    当前处理的是第 0 个 KV head，即 kv_head_idx = 0
    这个序列的 block_tables 映射中，逻辑块 1 映射到了 物理块 7

    logical_block_idx = offs_n // block_size   # [4//4, 5//4, 6//4, 7//4] = [1, 1, 1, 1]
    block_offset      = offs_n % block_size    # [4%4, 5%4, 6%4, 7%4] = [0, 1, 2, 3]
    block_ids = tl.load(block_tables_ptr + seq_idx * max_num_blocks + logical_block_idx)
    # 得到 block_ids = [7, 7, 7, 7]
    token_stride = num_kv_heads * head_dim = 2 * 8 = 16
    block_stride = block_size * token_stride = 4 * 16 = 64
    cache_offsets = (
        block_ids * block_stride          # 跳到物理块 7 的起始位置
        + block_offset * token_stride     # 在块内跳到第几个 token
        + kv_head_idx * head_dim          # 跳到第几个 KV head
    ) # [448, 464, 480, 496]
    k_ptrs = k_cache_ptr + cache_offsets[None, :] + offs_d[:, None]
    k = tl.load(k_ptrs, mask=mask_n[None, :], other=0.0)
"""
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)

        # Accumulate weighted values
        acc = acc + tl.dot(p.to(v.dtype), v)

        # Update normalizer
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_i_new

    # Final normalization
    acc = acc / l_i[:, None]

    # Store output: O has shape (total_tokens, num_heads, head_dim)
    o_ptrs = O + (seq_start + offs_m[:, None]) * num_heads * head_dim + off_h * head_dim + offs_d[None, :]
    tl.store(o_ptrs, acc.to(O.dtype.element_ty), mask=mask_m[:, None])
```

```python
def flash_attention_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
-    cu_seqlens: torch.Tensor,
+    cu_seqlens_q: torch.Tensor,
+    cu_seqlens_k: torch.Tensor,
+    k_cache: torch.Tensor,
+    v_cache: torch.Tensor,
+    block_tables: torch.Tensor | None,
+    block_size: int,
    scale: float,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """
    Optimized Flash Attention for prefill phase with variable-length sequences.
    
    Args:
        q: (total_tokens, num_heads, head_dim)
        k: (total_tokens, num_kv_heads, head_dim)
        v: (total_tokens, num_kv_heads, head_dim)
-    cu_seqlens: cumulative sequence lengths
+    cu_seqlens_q: cumulative lengths of newly computed query tokens
+    cu_seqlens_k: cumulative lengths of the full KV context
        scale: attention scale factor
    
    Returns:
        output: (total_tokens, num_heads, head_dim)
    """
    # Make tensors contiguous
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    # Allocate output
    output = torch.empty_like(q)

    # Conservative block sizes to avoid OOM on shared memory
    # Shared memory usage ~ BLOCK_M * BLOCK_N * 4 bytes (for float32 attention scores)
    # + BLOCK_M * head_dim * 4 (for Q)
    # + BLOCK_N * head_dim * 4 (for K, V)
    # Want to keep total < 48KB for most GPUs

    if head_dim <= 64:
        BLOCK_M = 64
        BLOCK_N = 64
    elif head_dim <= 128:
        BLOCK_M = 32
        BLOCK_N = 32
    else:
        BLOCK_M = 16
        BLOCK_N = 16

    # Number of sequences
-    num_seqs = cu_seqlens.shape[0] - 1
+    num_seqs = cu_seqlens_q.shape[0] - 1
-    cu_seqlens_cpu = cu_seqlens.cpu()
+    cu_seqlens_cpu = cu_seqlens_q.cpu()

    max_seq_len = (cu_seqlens_cpu[1:] - cu_seqlens_cpu[:-1]).max().item()

    # Calculate grid dimensions - launch all kernels at once
    grid = (triton.cdiv(max_seq_len, BLOCK_M), num_heads, num_seqs)

----- 4.
+    use_paged_kv = block_tables is not None
+    block_tables_ptr = block_tables if use_paged_kv else cu_seqlens_q
+    k_cache_ptr = k_cache if use_paged_kv else k
+    v_cache_ptr = v_cache if use_paged_kv else v
+    max_num_blocks = block_tables.shape[1] if use_paged_kv else 1	

     flash_attention_varlen_kernel[grid](
         q, k, v, output,
-        cu_seqlens,
+        cu_seqlens_q,
+        cu_seqlens_k,
+        k_cache_ptr,
+        v_cache_ptr,
+        block_tables_ptr,
         scale,
         num_heads=num_heads,
         num_kv_heads=num_kv_heads,
         head_dim=head_dim,
+        block_size=block_size,
+        max_num_blocks=max_num_blocks,
+        USE_PAGED_KV=use_paged_kv,
         BLOCK_M=BLOCK_M,
         BLOCK_N=BLOCK_N,
     )
```

·







