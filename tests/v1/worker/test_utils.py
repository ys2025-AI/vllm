# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.v1.core.kv_cache_utils import KVCacheBlockCopy
from vllm.v1.worker.utils import (
    _kv_cache_block_major_bytes,
    bind_kv_cache,
    copy_kv_cache_blocks_inplace,
)


def test_bind_kv_cache(default_vllm_config):
    from vllm.model_executor.layers.attention import Attention

    ctx = {
        "layers.0.self_attn": Attention(32, 128, 0.1, prefix="layers.0.self_attn"),
        "layers.1.self_attn": Attention(32, 128, 0.1, prefix="layers.1.self_attn"),
        "layers.2.self_attn": Attention(32, 128, 0.1, prefix="layers.2.self_attn"),
        "layers.3.self_attn": Attention(32, 128, 0.1, prefix="layers.3.self_attn"),
    }
    kv_cache = {
        "layers.0.self_attn": torch.zeros((1,)),
        "layers.1.self_attn": torch.zeros((1,)),
        "layers.2.self_attn": torch.zeros((1,)),
        "layers.3.self_attn": torch.zeros((1,)),
    }
    runner_kv_caches: list[torch.Tensor] = []
    bind_kv_cache(kv_cache, ctx, runner_kv_caches)
    assert ctx["layers.0.self_attn"].kv_cache is kv_cache["layers.0.self_attn"]
    assert ctx["layers.1.self_attn"].kv_cache is kv_cache["layers.1.self_attn"]
    assert ctx["layers.2.self_attn"].kv_cache is kv_cache["layers.2.self_attn"]
    assert ctx["layers.3.self_attn"].kv_cache is kv_cache["layers.3.self_attn"]

    assert runner_kv_caches[0] is kv_cache["layers.0.self_attn"]
    assert runner_kv_caches[1] is kv_cache["layers.1.self_attn"]
    assert runner_kv_caches[2] is kv_cache["layers.2.self_attn"]
    assert runner_kv_caches[3] is kv_cache["layers.3.self_attn"]


def test_bind_kv_cache_non_attention(default_vllm_config):
    from vllm.model_executor.layers.attention import Attention

    # example from Jamba PP=2
    ctx = {
        "model.layers.20.attn": Attention(32, 128, 0.1, prefix="model.layers.20.attn"),
        "model.layers.28.attn": Attention(32, 128, 0.1, prefix="model.layers.28.attn"),
    }
    kv_cache = {
        "model.layers.20.attn": torch.zeros((1,)),
        "model.layers.28.attn": torch.zeros((1,)),
    }

    runner_kv_caches: list[torch.Tensor] = []
    bind_kv_cache(kv_cache, ctx, runner_kv_caches)

    assert ctx["model.layers.20.attn"].kv_cache is kv_cache["model.layers.20.attn"]
    assert ctx["model.layers.28.attn"].kv_cache is kv_cache["model.layers.28.attn"]

    assert runner_kv_caches[0] is kv_cache["model.layers.20.attn"]
    assert runner_kv_caches[1] is kv_cache["model.layers.28.attn"]


def test_bind_kv_cache_draft_model(default_vllm_config):
    from vllm.model_executor.layers.attention import Attention

    layer_names = [
        "model.layers.0.attn",
        "model.layers.1.attn",
        "draft_model.layers.0.attn",
        "draft_model.layers.1.attn",
    ]
    ctx = {
        layer_name: Attention(32, 128, 0.1, prefix=layer_name)
        for layer_name in layer_names
    }
    kv_cache = {layer_name: torch.zeros((1,)) for layer_name in layer_names}
    runner_kv_caches: list[torch.Tensor] = []
    bind_kv_cache(kv_cache, ctx, runner_kv_caches)

    assert ctx["model.layers.0.attn"].kv_cache is kv_cache["model.layers.0.attn"]
    assert ctx["model.layers.1.attn"].kv_cache is kv_cache["model.layers.1.attn"]
    assert (
        ctx["draft_model.layers.0.attn"].kv_cache
        is kv_cache["draft_model.layers.0.attn"]
    )
    assert (
        ctx["draft_model.layers.1.attn"].kv_cache
        is kv_cache["draft_model.layers.1.attn"]
    )

    # caches are ordered by layer_index, interleaving target and draft model
    assert runner_kv_caches[0] is kv_cache["model.layers.0.attn"]
    assert runner_kv_caches[1] is kv_cache["draft_model.layers.0.attn"]
    assert runner_kv_caches[2] is kv_cache["model.layers.1.attn"]
    assert runner_kv_caches[3] is kv_cache["draft_model.layers.1.attn"]


def _block_major_bytes(tensor: torch.Tensor, num_blocks: int) -> torch.Tensor:
    return _kv_cache_block_major_bytes(tensor, num_blocks).view(num_blocks, -1)


def _assert_copy(
    probe: torch.Tensor,
    kv_caches: list,
    num_blocks: int,
    src_block: int,
    dst_block: int,
    expected: torch.Tensor,
) -> None:
    copy_kv_cache_blocks_inplace(
        kv_caches, num_blocks, [KVCacheBlockCopy(src_block, dst_block)]
    )
    after = _block_major_bytes(probe, num_blocks)
    assert torch.equal(after[dst_block], expected[src_block])
    for block in range(num_blocks):
        if block != dst_block:
            assert torch.equal(after[block], expected[block])


def test_copy_kv_cache_blocks_aligned_over_allocation():
    """Regression test: the platform allocator may over-allocate a KV cache
    tensor to align ``data_ptr`` (e.g. vllm-ascend allocates ``size + 2MiB``).
    The copy must operate on the tensor's own block-major region instead of the
    whole storage, otherwise the block count no longer divides the storage
    size and the copy assertion fails."""
    num_blocks, page_bytes, alignment = 7, 16, 8
    per_layer = num_blocks * page_bytes
    # alignment chosen so that (num_blocks * page_bytes + alignment) is not a
    # multiple of num_blocks.
    raw = torch.zeros(per_layer + alignment, dtype=torch.uint8)
    raw[alignment : alignment + per_layer] = torch.arange(per_layer, dtype=torch.uint8)
    # bf16 view with a nonzero storage offset, as produced by vllm-ascend.
    tensor = (
        raw[alignment : alignment + per_layer]
        .view(torch.bfloat16)
        .view(num_blocks, page_bytes // 2)
    )
    assert raw.untyped_storage().nbytes() % num_blocks != 0

    expected = raw[alignment : alignment + per_layer].view(num_blocks, page_bytes)
    assert torch.equal(_block_major_bytes(tensor, num_blocks), expected)
    _assert_copy(tensor, [tensor], num_blocks, 2, 5, expected)


def test_copy_kv_cache_blocks_page_padded_strided():
    """Block copy must follow the padded page stride of a strided cache view."""
    num_blocks, inner_bytes, padded_page = 6, 12, 16
    raw = torch.arange(num_blocks * padded_page, dtype=torch.uint8)
    tensor = torch.as_strided(
        raw, size=(num_blocks, inner_bytes), stride=(padded_page, 1)
    )
    assert not tensor.is_contiguous()

    expected = raw.view(num_blocks, padded_page)
    assert torch.equal(_block_major_bytes(tensor, num_blocks), expected)
    _assert_copy(tensor, [tensor], num_blocks, 1, 4, expected)


def test_copy_kv_cache_blocks_virtual_split():
    """Dense caches may expose several kernel blocks per logical block."""
    num_blocks, ratio, inner_bytes = 5, 3, 8
    raw = torch.arange(num_blocks * ratio * inner_bytes, dtype=torch.uint8)
    tensor = raw.view(num_blocks * ratio, inner_bytes)

    expected = raw.view(num_blocks, ratio * inner_bytes)
    assert torch.equal(_block_major_bytes(tensor, num_blocks), expected)
    _assert_copy(tensor, [tensor], num_blocks, 0, 3, expected)


def test_copy_kv_cache_blocks_skips_duplicate_and_empty_entries():
    """Duplicate views of the same data pointer are copied once; empty and
    non-tensor entries are ignored."""
    num_blocks, page_bytes = 4, 8
    tensor = torch.arange(num_blocks * page_bytes, dtype=torch.uint8)
    other = torch.zeros(0, dtype=torch.uint8)
    expected = tensor.view(num_blocks, page_bytes).clone()
    _assert_copy(
        tensor,
        [(tensor, tensor), other, None, (tensor, tensor)],
        num_blocks,
        1,
        2,
        expected,
    )
