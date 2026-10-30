"""MoE HummingLayer on ROCm: indexed / grouped_contiguous / grouped_masked through the public API.

Every case builds an MoE ``HummingLayer`` (load_from_unquantized -> transform), routes tokens
with ``humming.ops.moe_align_block_size`` (indexed) or an expert-sorted copy of the inputs
(grouped), runs ``layer(...)`` (humming_forward -> gfx950 MFMA MoE kernel) and compares with a
per-expert FP32 GEMM on the layer's own dequantized weights (``weight_schema.dequant_tensors``
before transform).
"""

import pytest
import torch

from humming import ops
from humming.layer import HummingLayer
from humming.tune import get_heuristics_config
from tests.rocm.quant_utils import TOLERANCE

DTYPES = [torch.float16, torch.bfloat16]
DTYPE_IDS = ["fp16", "bf16"]
GEMM_TYPES = ["indexed", "grouped_contiguous", "grouped_masked"]


def weight_config(num_bits, group_size, has_zero_point):
    config = {
        "dtype": f"uint{num_bits}" if has_zero_point else f"int{num_bits}",
        "has_zero_point": has_zero_point,
    }
    if group_size:
        config["group_size"] = group_size
    return config


W4G128 = weight_config(4, 128, False)


def build_moe_layer(num_experts, shape_n, shape_k, config, dtype, has_bias=False, seed=0):
    torch.manual_seed(seed)
    layer = HummingLayer(
        shape_n=shape_n,
        shape_k=shape_k,
        weight_config=dict(config),
        torch_dtype=dtype,
        num_experts=num_experts,
        has_bias=has_bias,
    ).cuda()
    layer.load_from_unquantized(torch.randn(num_experts, shape_n, shape_k, dtype=dtype, device="cuda"))
    bias = None
    if has_bias:
        bias = torch.randn(num_experts, shape_n, dtype=dtype, device="cuda")
        layer.bias.data.copy_(bias)
    quantized = {name: tensor.clone() for name, tensor in layer.state_dict().items()}
    reference_weight = layer.weight_schema.dequant_tensors(quantized).float()
    layer.transform()
    return layer, reference_weight, bias


def random_topk_ids(num_tokens, num_experts, top_k, skew=None):
    """Distinct experts per token; ``skew`` = number of experts that receive all tokens."""
    pool = num_experts if skew is None else skew
    scores = torch.rand(num_tokens, pool, device="cuda")
    ids = scores.topk(top_k, dim=1).indices
    if skew is not None:
        ids = torch.randperm(num_experts, device="cuda")[:pool][ids]
    return ids.int()


def reference_rows(inputs, reference_weight, bias, row_experts):
    """FP32 per-expert GEMM; row i uses expert row_experts[i] (-1: zero row)."""
    out = torch.zeros(inputs.size(0), reference_weight.size(1), device="cuda")
    for expert in row_experts.unique().tolist():
        if expert < 0:
            continue
        rows = (row_experts == expert).nonzero().flatten()
        out[rows] = inputs[rows].float() @ reference_weight[expert].T
        if bias is not None:
            out[rows] += bias[expert].float()
    return out


def block_m_for(layer, gemm_type, valid_shape_m):
    for low, high, config in get_heuristics_config(layer.humming_config, gemm_type=gemm_type):
        if low < valid_shape_m <= high:
            return config["block_shape"][0]
    raise AssertionError(valid_shape_m)


def run_moe(
    layer, reference_weight, bias, inputs, topk_ids, gemm_type, layout_dtype=torch.int64, expert_map=None
):
    """Returns (outputs, reference) restricted to the rows the GEMM defines."""
    num_tokens, top_k = topk_ids.shape
    compute_config = {"gemm_type": gemm_type}
    flat = topk_ids.flatten().long()
    local = flat if expert_map is None else expert_map[flat].long()
    valid_shape_m = num_tokens * top_k
    if gemm_type == "indexed":
        block_m = block_m_for(layer, gemm_type, valid_shape_m)
        num_experts = layer.humming_config.num_experts if expert_map is None else expert_map.numel()
        sorted_ids, expert_ids, num_tokens_padded = ops.moe_align_block_size(
            topk_ids,
            block_m,
            num_experts,
            expert_map=expert_map,
            ignore_invalid_experts=expert_map is not None,
        )
        outputs = layer(
            inputs,
            sorted_ids=sorted_ids,
            expert_ids=expert_ids,
            num_tokens_padded=num_tokens_padded,
            top_k=top_k,
            valid_shape_m=valid_shape_m,
            compute_config=compute_config,
        )
        assert outputs.shape == (num_tokens * top_k, reference_weight.size(1))
        reference = reference_rows(inputs.repeat_interleave(top_k, dim=0), reference_weight, bias, local)
        keep = local >= 0
        return outputs[keep], reference[keep]

    num_experts = layer.humming_config.num_experts
    order = flat.argsort(stable=True)
    sorted_inputs = inputs[order // top_k]
    sorted_experts = flat[order]
    counts = flat.bincount(minlength=num_experts)
    if gemm_type == "grouped_contiguous":
        offsets = torch.nn.functional.pad(counts.cumsum(0), (1, 0)).to(layout_dtype)
        outputs = layer(
            sorted_inputs, expert_layout=offsets, valid_shape_m=valid_shape_m, compute_config=compute_config
        )
        assert outputs.shape == (sorted_inputs.size(0), reference_weight.size(1))
        return outputs, reference_rows(sorted_inputs, reference_weight, bias, sorted_experts)

    assert gemm_type == "grouped_masked"
    max_tokens = int(counts.max()) + 5
    masked_inputs = torch.randn(num_experts * max_tokens, inputs.size(1), dtype=inputs.dtype, device="cuda")
    starts = (counts.cumsum(0) - counts).tolist()
    row_index = torch.cat(
        [torch.arange(c, device="cuda") + e * max_tokens for e, c in enumerate(counts.tolist())]
    )
    masked_inputs[row_index] = sorted_inputs
    del starts
    outputs = layer(
        masked_inputs,
        expert_layout=counts.to(layout_dtype),
        valid_shape_m=valid_shape_m,
        compute_config=compute_config,
    )
    assert outputs.shape == (masked_inputs.size(0), reference_weight.size(1))
    return outputs[row_index], reference_rows(sorted_inputs, reference_weight, bias, sorted_experts)


def relative_error(outputs, reference):
    return ((outputs.float() - reference).norm() / reference.norm().clamp_min(1e-6)).item()


# -------------------------------------------------------------------------------------------------
# Layouts x expert counts x token counts


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
@pytest.mark.parametrize("gemm_type", GEMM_TYPES)
@pytest.mark.parametrize(
    ("num_experts", "top_k"), [(8, 2), (64, 8), (128, 1), (256, 8)], ids=lambda v: str(v)
)
def test_layouts(dtype, gemm_type, num_experts, top_k):
    layer, reference_weight, _ = build_moe_layer(num_experts, 256, 512, W4G128, dtype)
    for num_tokens in [1, 4, 16, 64, 512, 4096]:
        inputs = torch.randn(num_tokens, 512, dtype=dtype, device="cuda")
        topk_ids = random_topk_ids(num_tokens, num_experts, top_k)
        outputs, reference = run_moe(layer, reference_weight, None, inputs, topk_ids, gemm_type)
        assert relative_error(outputs, reference) < TOLERANCE[dtype], num_tokens


@pytest.mark.parametrize("gemm_type", ["grouped_contiguous", "grouped_masked"])
def test_int32_expert_layout(gemm_type):
    dtype = torch.float16
    layer, reference_weight, _ = build_moe_layer(16, 128, 256, W4G128, dtype)
    for num_tokens in [3, 100]:
        inputs = torch.randn(num_tokens, 256, dtype=dtype, device="cuda")
        topk_ids = random_topk_ids(num_tokens, 16, 4)
        outputs, reference = run_moe(layer, reference_weight, None, inputs, topk_ids, gemm_type, torch.int32)
        assert relative_error(outputs, reference) < TOLERANCE[dtype]


# -------------------------------------------------------------------------------------------------
# Quantization variants


SCALE_CASES = [(0, False), (0, True), (32, True), (64, False), (128, False), (128, True)]


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
@pytest.mark.parametrize("num_bits", [2, 3, 4, 8])
@pytest.mark.parametrize(("group_size", "has_zero_point"), SCALE_CASES, ids=lambda v: str(v))
def test_bits_and_scales(dtype, num_bits, group_size, has_zero_point):
    config = weight_config(num_bits, group_size, has_zero_point)
    layer, reference_weight, _ = build_moe_layer(8, 384, 1024, config, dtype)
    for gemm_type in GEMM_TYPES:
        for num_tokens in [1, 16, 200]:
            inputs = torch.randn(num_tokens, 1024, dtype=dtype, device="cuda")
            topk_ids = random_topk_ids(num_tokens, 8, 2)
            outputs, reference = run_moe(layer, reference_weight, None, inputs, topk_ids, gemm_type)
            assert relative_error(outputs, reference) < TOLERANCE[dtype], (gemm_type, num_tokens)


# -------------------------------------------------------------------------------------------------
# Model shapes (w13 = gate/up, w2 = down)


MODEL_SHAPES = {
    # name: (experts, top_k, [(N, K), ...])
    "mixtral_8x7b": (8, 2, [(28672, 4096), (4096, 14336)]),
    "qwen3_30b_a3b": (128, 8, [(1536, 2048), (2048, 768)]),
    "deepseek_v3_tp8": (256, 8, [(512, 7168), (7168, 256)]),
}


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
@pytest.mark.parametrize("model", list(MODEL_SHAPES))
@pytest.mark.parametrize("sublayer", [0, 1], ids=["w13", "w2"])
@pytest.mark.parametrize("config", [W4G128, weight_config(8, 0, False)], ids=["w4g128", "w8"])
def test_model_shapes(dtype, model, sublayer, config):
    num_experts, top_k, shapes = MODEL_SHAPES[model]
    shape_n, shape_k = shapes[sublayer]
    layer, reference_weight, _ = build_moe_layer(num_experts, shape_n, shape_k, config, dtype)
    for num_tokens in [1, 4, 64, 512, 4096]:
        inputs = torch.randn(num_tokens, shape_k, dtype=dtype, device="cuda")
        topk_ids = random_topk_ids(num_tokens, num_experts, top_k)
        gemm_type = "indexed" if num_tokens <= 64 else "grouped_contiguous"
        outputs, reference = run_moe(layer, reference_weight, None, inputs, topk_ids, gemm_type)
        assert relative_error(outputs, reference) < TOLERANCE[dtype], num_tokens


def test_w13_w2_chain():
    """vLLM's indexed flow: one alignment, w13 with top_k, w2 with top_k=1 on the [M * top_k, I] rows."""
    dtype = torch.bfloat16
    num_experts, top_k, hidden, inter = 64, 8, 1024, 384
    w13, ref13, _ = build_moe_layer(num_experts, 2 * inter, hidden, W4G128, dtype, seed=1)
    w2, ref2, _ = build_moe_layer(num_experts, hidden, inter, W4G128, dtype, seed=2)
    for num_tokens in [1, 7, 64, 1000]:
        inputs = torch.randn(num_tokens, hidden, dtype=dtype, device="cuda")
        topk_ids = random_topk_ids(num_tokens, num_experts, top_k)
        valid_shape_m = num_tokens * top_k
        block_m = block_m_for(w13, "indexed", valid_shape_m)
        assert block_m == block_m_for(w2, "indexed", valid_shape_m)
        sorted_ids, expert_ids, num_tokens_padded = ops.moe_align_block_size(topk_ids, block_m, num_experts)
        moe_kwargs = {
            "sorted_ids": sorted_ids,
            "expert_ids": expert_ids,
            "num_tokens_padded": num_tokens_padded,
            "valid_shape_m": valid_shape_m,
            "compute_config": {"gemm_type": "indexed"},
        }
        gate_up = w13(inputs, top_k=top_k, **moe_kwargs)
        activation = torch.nn.functional.silu(gate_up[:, :inter]) * gate_up[:, inter:]
        down = w2(activation, top_k=1, **moe_kwargs)

        flat = topk_ids.flatten().long()
        reference = reference_rows(activation, ref2, None, flat)
        assert relative_error(down, reference) < TOLERANCE[dtype], num_tokens
        reference13 = reference_rows(inputs.repeat_interleave(top_k, 0), ref13, None, flat)
        assert relative_error(gate_up, reference13) < TOLERANCE[dtype], num_tokens


# -------------------------------------------------------------------------------------------------
# Routing corner cases


@pytest.mark.parametrize("gemm_type", GEMM_TYPES)
@pytest.mark.parametrize("skew", [1, 3], ids=["one_expert", "three_experts"])
def test_uneven_and_empty_experts(gemm_type, skew):
    dtype = torch.float16
    num_experts = 32
    layer, reference_weight, _ = build_moe_layer(num_experts, 256, 512, W4G128, dtype)
    for num_tokens in [1, 5, 300, 3000]:
        inputs = torch.randn(num_tokens, 512, dtype=dtype, device="cuda")
        topk_ids = random_topk_ids(num_tokens, num_experts, 1, skew=skew)
        outputs, reference = run_moe(layer, reference_weight, None, inputs, topk_ids, gemm_type)
        assert relative_error(outputs, reference) < TOLERANCE[dtype], num_tokens


@pytest.mark.parametrize("gemm_type", GEMM_TYPES)
def test_all_experts_empty(gemm_type):
    dtype = torch.float16
    layer, _, _ = build_moe_layer(8, 128, 256, W4G128, dtype)
    if gemm_type == "indexed":
        topk_ids = torch.full((4, 2), 99, dtype=torch.int32, device="cuda")  # out of range: dropped
        sorted_ids, expert_ids, num_tokens_padded = ops.moe_align_block_size(topk_ids, 16, 8)
        assert int(num_tokens_padded) == 0
        outputs = torch.full((8, 128), 7.0, dtype=dtype, device="cuda")
        layer(
            torch.randn(4, 256, dtype=dtype, device="cuda"),
            outputs=outputs,
            sorted_ids=sorted_ids,
            expert_ids=expert_ids,
            num_tokens_padded=num_tokens_padded,
            top_k=2,
            compute_config={"gemm_type": gemm_type},
        )
    else:
        layout = torch.zeros(9 if gemm_type == "grouped_contiguous" else 8, dtype=torch.int64, device="cuda")
        outputs = torch.full((16, 128), 7.0, dtype=dtype, device="cuda")
        layer(
            torch.randn(16, 256, dtype=dtype, device="cuda"),
            outputs=outputs,
            expert_layout=layout,
            compute_config={"gemm_type": gemm_type},
        )
    torch.cuda.synchronize()
    assert (outputs == 7.0).all()


def test_expert_parallel_map():
    """Indexed GEMM with an expert_map: rows routed to non-local experts are skipped."""
    dtype = torch.bfloat16
    global_experts, local_experts = 16, 8
    layer, reference_weight, _ = build_moe_layer(local_experts, 256, 512, W4G128, dtype)
    expert_map = torch.full((global_experts,), -1, dtype=torch.int32, device="cuda")
    expert_map[4:12] = torch.arange(local_experts, dtype=torch.int32, device="cuda")
    for num_tokens in [1, 9, 500]:
        inputs = torch.randn(num_tokens, 512, dtype=dtype, device="cuda")
        topk_ids = random_topk_ids(num_tokens, global_experts, 4)
        if (expert_map[topk_ids.long()] < 0).all():
            continue
        outputs, reference = run_moe(
            layer, reference_weight, None, inputs, topk_ids, "indexed", expert_map=expert_map
        )
        assert relative_error(outputs, reference) < TOLERANCE[dtype], num_tokens


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
@pytest.mark.parametrize("gemm_type", GEMM_TYPES)
def test_padding_and_bias(dtype, gemm_type):
    """N / K not multiples of 64 / 128 (padded by transform), per-expert bias added in the kernel."""
    layer, reference_weight, bias = build_moe_layer(
        8, 96, 224, weight_config(4, 32, True), dtype, has_bias=True
    )
    for num_tokens in [1, 33, 700]:
        inputs = torch.randn(num_tokens, 224, dtype=dtype, device="cuda")
        topk_ids = random_topk_ids(num_tokens, 8, 2)
        outputs, reference = run_moe(layer, reference_weight, bias, inputs, topk_ids, gemm_type)
        assert outputs.size(-1) == 96
        assert relative_error(outputs, reference) < TOLERANCE[dtype], num_tokens


def test_strided_outputs():
    """Outputs may be a column slice of a wider buffer (row stride != N)."""
    dtype = torch.float16
    layer, reference_weight, _ = build_moe_layer(8, 256, 512, W4G128, dtype)
    inputs = torch.randn(20, 512, dtype=dtype, device="cuda")
    topk_ids = random_topk_ids(20, 8, 2)
    sorted_ids, expert_ids, num_tokens_padded = ops.moe_align_block_size(topk_ids, 16, 8)
    wide = torch.zeros(40, 300, dtype=dtype, device="cuda")
    outputs = wide[:, 10:266]
    layer(
        inputs,
        outputs=outputs,
        sorted_ids=sorted_ids,
        expert_ids=expert_ids,
        num_tokens_padded=num_tokens_padded,
        top_k=2,
        valid_shape_m=40,
        compute_config={"gemm_type": "indexed"},
    )
    reference = reference_rows(
        inputs.repeat_interleave(2, 0), reference_weight, None, topk_ids.flatten().long()
    )
    assert relative_error(outputs, reference) < TOLERANCE[dtype]
    assert (wide[:, :10] == 0).all() and (wide[:, 266:] == 0).all()


# -------------------------------------------------------------------------------------------------
# CUDA graphs


@pytest.mark.parametrize("gemm_type", GEMM_TYPES)
@pytest.mark.parametrize("num_tokens", [1, 16, 300])
def test_cuda_graph_capture(gemm_type, num_tokens):
    """Capture routing + MoE GEMM (split-K for small M) and replay with new inputs and routing."""
    dtype = torch.bfloat16
    num_experts, top_k, shape_k = 8, 2, 4096
    layer, reference_weight, _ = build_moe_layer(num_experts, 512, shape_k, W4G128, dtype)
    inputs = torch.randn(num_tokens, shape_k, dtype=dtype, device="cuda")
    topk_ids = random_topk_ids(num_tokens, num_experts, top_k)
    compute_config = {"gemm_type": gemm_type}
    rows = num_tokens * top_k
    max_tokens = rows  # grouped_masked capacity per expert

    def forward():
        if gemm_type == "indexed":
            block_m = block_m_for(layer, gemm_type, rows)
            sorted_ids, expert_ids, num_tokens_padded = ops.moe_align_block_size(
                topk_ids, block_m, num_experts
            )
            return layer(
                inputs,
                sorted_ids=sorted_ids,
                expert_ids=expert_ids,
                num_tokens_padded=num_tokens_padded,
                top_k=top_k,
                valid_shape_m=rows,
                compute_config=compute_config,
            )
        flat = topk_ids.flatten().long()
        order = flat.argsort(stable=True)
        counts = torch.zeros(num_experts, dtype=torch.int64, device="cuda").scatter_add_(
            0, flat, torch.ones_like(flat)
        )
        sorted_inputs = inputs[order // top_k]
        if gemm_type == "grouped_contiguous":
            offsets = torch.nn.functional.pad(counts.cumsum(0), (1, 0))
            return layer(
                sorted_inputs, expert_layout=offsets, valid_shape_m=rows, compute_config=compute_config
            )
        starts = counts.cumsum(0) - counts
        sorted_experts = flat[order]
        dest = sorted_experts * max_tokens + torch.arange(rows, device="cuda") - starts[sorted_experts]
        masked = torch.zeros(num_experts * max_tokens, shape_k, dtype=dtype, device="cuda")
        masked.index_copy_(0, dest, sorted_inputs)
        out = layer(masked, expert_layout=counts, valid_shape_m=rows, compute_config=compute_config)
        return out[dest]

    def reference():
        flat = topk_ids.flatten().long()
        if gemm_type == "indexed":
            return reference_rows(inputs.repeat_interleave(top_k, 0), reference_weight, None, flat)
        order = flat.argsort(stable=True)
        return reference_rows(inputs[order // top_k], reference_weight, None, flat[order])

    side_stream = torch.cuda.Stream()
    side_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side_stream):
        forward()
    torch.cuda.current_stream().wait_stream(side_stream)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = forward()
    for _ in range(3):
        inputs.copy_(torch.randn_like(inputs))
        topk_ids.copy_(random_topk_ids(num_tokens, num_experts, top_k))
        graph.replay()
        torch.cuda.synchronize()
        assert relative_error(outputs, reference()) < TOLERANCE[dtype]


# -------------------------------------------------------------------------------------------------
# Helpers


ALIGN_IMPLS = [ops.moe_align_block_size, ops.moe_align_block_size_torch]
ALIGN_IDS = ["hip", "torch"]


@pytest.mark.parametrize("align", ALIGN_IMPLS, ids=ALIGN_IDS)
@pytest.mark.parametrize("num_tokens", [0, 1, 5, 64, 1000, 8192])
@pytest.mark.parametrize(("num_experts", "top_k", "block_size"), [(8, 2, 16), (64, 8, 32), (256, 8, 128)])
def test_moe_align_block_size(align, num_tokens, num_experts, top_k, block_size):
    topk_ids = (
        random_topk_ids(num_tokens, num_experts, top_k)
        if num_tokens
        else torch.empty(0, top_k, dtype=torch.int32, device="cuda")
    )
    sorted_ids, expert_ids, num_tokens_padded = align(topk_ids, block_size, num_experts)
    numel = topk_ids.numel()
    # Same buffer sizes as vLLM's moe_align_block_size.
    max_padded = numel + num_experts * (block_size - 1)
    if numel < num_experts:
        max_padded = min(numel * block_size, max_padded)
    assert sorted_ids.numel() == -(-max_padded // block_size) * block_size
    assert expert_ids.numel() == -(-max_padded // block_size)
    flat = topk_ids.flatten().long().cpu()
    sorted_ids, expert_ids, used = sorted_ids.cpu(), expert_ids.cpu(), int(num_tokens_padded)
    counts = flat.bincount(minlength=num_experts)
    assert used == int(((counts + block_size - 1) // block_size * block_size).sum())
    seen = []
    for block in range(used // block_size):
        expert = int(expert_ids[block])
        ids = sorted_ids[block * block_size : (block + 1) * block_size]
        real = ids[ids < numel]
        assert (flat[real] == expert).all()
        seen.append(real)
    assert (expert_ids[used // block_size :] == -1).all()
    assert (sorted_ids[used:] == numel).all()
    seen = torch.cat(seen) if seen else torch.empty(0, dtype=torch.int32)
    assert sorted(seen.tolist()) == list(range(numel))


def canonical_alignment(sorted_ids, expert_ids, num_tokens_padded, block_size):
    """Sort the row ids inside every run of same-expert blocks (the HIP kernel does not fix the
    order of rows inside an expert; padding sorts last since it equals numel)."""
    sorted_ids, expert_ids = sorted_ids.long().cpu(), expert_ids.cpu()
    if expert_ids.numel():
        run = torch.cat(
            [torch.zeros(1, dtype=torch.long), (expert_ids[1:] != expert_ids[:-1]).long()]
        ).cumsum(0)
        key = run.repeat_interleave(block_size) * (1 << 32) + sorted_ids
        sorted_ids = sorted_ids[key.argsort()]
    return sorted_ids, expert_ids, int(num_tokens_padded)


@pytest.mark.parametrize("ids_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("map_mode", ["none", "keep", "ignore"])
@pytest.mark.parametrize(
    ("num_tokens", "num_experts", "top_k", "block_size"),
    [(1, 8, 2, 16), (3, 256, 8, 16), (77, 64, 6, 32), (4096, 128, 8, 64), (513, 1024, 4, 16)],
)
def test_moe_align_hip_matches_torch(ids_dtype, map_mode, num_tokens, num_experts, top_k, block_size):
    """HIP align == torch align (up to the row order inside an expert): expert_map, int64, invalid ids."""
    topk_ids = random_topk_ids(num_tokens, num_experts, top_k).to(ids_dtype)
    topk_ids.view(-1)[::7] = -1  # invalid (e.g. masked-out) routes are dropped
    expert_map = None
    if map_mode != "none":
        expert_map = torch.full((num_experts,), -1, dtype=torch.int32, device="cuda")
        local = num_experts // 2
        expert_map[num_experts // 4 : num_experts // 4 + local] = torch.arange(
            local, dtype=torch.int32, device="cuda"
        )
    ignore = map_mode == "ignore"
    hip = ops.moe_align_block_size(topk_ids, block_size, num_experts, expert_map, ignore)
    ref = ops.moe_align_block_size_torch(topk_ids, block_size, num_experts, expert_map, ignore)
    hip, ref = canonical_alignment(*hip, block_size), canonical_alignment(*ref, block_size)
    assert hip[2] == ref[2]
    assert torch.equal(hip[1], ref[1])
    assert torch.equal(hip[0], ref[0])


def test_moe_align_cuda_graph():
    """The HIP align replays inside a CUDA graph on new routing."""
    num_tokens, num_experts, top_k, block_size = 300, 64, 8, 32
    topk_ids = random_topk_ids(num_tokens, num_experts, top_k)
    ops.moe_align_block_size(topk_ids, block_size, num_experts)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = ops.moe_align_block_size(topk_ids, block_size, num_experts)
    for _ in range(3):
        topk_ids.copy_(random_topk_ids(num_tokens, num_experts, top_k))
        graph.replay()
        ref = ops.moe_align_block_size_torch(topk_ids, block_size, num_experts)
        torch.cuda.synchronize()
        got, want = canonical_alignment(*outputs, block_size), canonical_alignment(*ref, block_size)
        assert got[2] == want[2] and torch.equal(got[1], want[1]) and torch.equal(got[0], want[0])


def test_heuristics_block_m_matches_between_sublayers():
    """w13 and w2 of one MoE layer must align to the same block_m for any valid_shape_m."""
    dtype = torch.float16
    w13, _, _ = build_moe_layer(64, 256, 512, W4G128, dtype)
    w2, _, _ = build_moe_layer(64, 512, 128, weight_config(8, 0, False), dtype)
    configs13 = get_heuristics_config(w13.humming_config, gemm_type="indexed")
    configs2 = get_heuristics_config(w2.humming_config, gemm_type="indexed")
    assert [(lo, hi, c["block_shape"][0]) for lo, hi, c in configs13] == [
        (lo, hi, c["block_shape"][0]) for lo, hi, c in configs2
    ]


def test_moe_layer_needs_moe_gemm_type():
    layer, _, _ = build_moe_layer(8, 128, 256, W4G128, torch.float16)
    with pytest.raises(Exception, match="gemm_type"):
        layer(torch.randn(4, 256, dtype=torch.float16, device="cuda"))
