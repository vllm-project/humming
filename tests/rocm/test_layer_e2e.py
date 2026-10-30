"""End-to-end HummingLayer on ROCm: load_from_unquantized -> transform -> layer(inputs).

Every case goes through the public API (gfx950 MFMA GEMM JIT-compiled behind humming_forward)
and is checked against an FP32 GEMM with the layer's own dequantized weight
(``weight_schema.dequant_tensors`` of the quantized tensors, before transform).
"""

import pytest
import torch

from humming.layer import HummingLayer
from tests.rocm.quant_utils import TOLERANCE

M_VALUES = [1, 7, 64, 255, 1024]
DTYPES = [torch.float16, torch.bfloat16]
DTYPE_IDS = ["fp16", "bf16"]


def build_layer(shape_n, shape_k, weight_config, dtype, has_bias=False, seed=0):
    torch.manual_seed(seed)
    layer = HummingLayer(
        shape_n=shape_n,
        shape_k=shape_k,
        weight_config=dict(weight_config),
        torch_dtype=dtype,
        has_bias=has_bias,
    ).cuda()
    layer.load_from_unquantized(torch.randn(shape_n, shape_k, dtype=dtype, device="cuda"))
    bias = None
    if has_bias:
        bias = torch.randn(shape_n, dtype=dtype, device="cuda")
        layer.bias.data.copy_(bias)
    quantized = {name: tensor.clone() for name, tensor in layer.state_dict().items()}
    reference_weight = layer.weight_schema.dequant_tensors(quantized).float()
    layer.transform()
    return layer, reference_weight, bias


def relative_error(layer, reference_weight, bias, shape_m, dtype):
    inputs = torch.randn(shape_m, reference_weight.size(1), dtype=dtype, device="cuda")
    outputs = layer(inputs)
    assert outputs.shape == (shape_m, reference_weight.size(0))
    assert outputs.dtype == dtype
    reference = inputs.float() @ reference_weight.T
    if bias is not None:
        reference += bias.float()
    return ((outputs.float() - reference).norm() / reference.norm()).item()


def weight_config(num_bits, group_size, has_zero_point):
    # "intN" is symmetric (implicit zero point 2^(N-1)); "uintN" + has_zero_point is asymmetric.
    config = {
        "dtype": f"uint{num_bits}" if has_zero_point else f"int{num_bits}",
        "has_zero_point": has_zero_point,
    }
    if group_size:
        config["group_size"] = group_size
    return config


SCALE_CASES = [(0, False), (32, True), (64, False), (128, False), (128, True)]


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
@pytest.mark.parametrize("num_bits", [1, 2, 3, 4, 5, 6, 7, 8])
@pytest.mark.parametrize(("group_size", "has_zero_point"), SCALE_CASES, ids=lambda v: str(v))
def test_bits_and_scales(dtype, num_bits, group_size, has_zero_point):
    layer, reference_weight, bias = build_layer(
        512, 1024, weight_config(num_bits, group_size, has_zero_point), dtype
    )
    for shape_m in M_VALUES:
        assert relative_error(layer, reference_weight, bias, shape_m, dtype) < TOLERANCE[dtype], shape_m


REAL_SHAPES = [(4096, 4096), (14336, 4096), (4096, 14336), (28672, 8192)]


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
@pytest.mark.parametrize(("shape_n", "shape_k"), REAL_SHAPES, ids=lambda v: str(v))
@pytest.mark.parametrize(
    "config",
    [weight_config(4, 128, False), weight_config(4, 128, True), weight_config(8, 0, False)],
    ids=["w4g128", "w4g128zp", "w8"],
)
def test_real_shapes(dtype, shape_n, shape_k, config):
    layer, reference_weight, bias = build_layer(shape_n, shape_k, config, dtype)
    for shape_m in M_VALUES:
        assert relative_error(layer, reference_weight, bias, shape_m, dtype) < TOLERANCE[dtype], shape_m


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
@pytest.mark.parametrize(
    ("shape_n", "shape_k"), [(96, 224), (1056, 1056), (4128, 3008)], ids=lambda v: str(v)
)
def test_padding_and_bias(dtype, shape_n, shape_k):
    """N / K that are not multiples of 64 / 128 are padded by transform(); bias is added after."""
    layer, reference_weight, bias = build_layer(
        shape_n, shape_k, weight_config(4, 32, True), dtype, has_bias=True
    )
    for shape_m in M_VALUES:
        assert relative_error(layer, reference_weight, bias, shape_m, dtype) < TOLERANCE[dtype], shape_m


def test_outputs_argument():
    dtype = torch.float16
    layer, reference_weight, _ = build_layer(1024, 2048, weight_config(4, 128, False), dtype)
    inputs = torch.randn(33, 2048, dtype=dtype, device="cuda")
    outputs = torch.empty(33, 1024, dtype=dtype, device="cuda")
    result = layer(inputs, outputs=outputs)
    assert result.data_ptr() == outputs.data_ptr()
    reference = inputs.float() @ reference_weight.T
    assert ((outputs.float() - reference).norm() / reference.norm()).item() < TOLERANCE[dtype]


def test_readme_int6_example():
    layer, reference_weight, _ = build_layer(8192, 8192, {"dtype": "int6"}, torch.float16)
    assert relative_error(layer, reference_weight, None, 128, torch.float16) < TOLERANCE[torch.float16]


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
@pytest.mark.parametrize("shape_m", [1, 16, 200])
@pytest.mark.parametrize("warmup", [True, False], ids=["warm", "cold"])
def test_cuda_graph_capture(dtype, shape_m, warmup):
    """Capture layer(x) in a graph (split-K active for small M) and replay it with new inputs.

    "cold" captures the very first forward of a fresh layer: kernel preparation, registration and
    module load all happen inside the capture.
    """
    shape_n = 4096 if warmup else 4160
    layer, reference_weight, _ = build_layer(shape_n, 4096, weight_config(4, 128, False), dtype)
    inputs = torch.randn(shape_m, 4096, dtype=dtype, device="cuda")
    if warmup:
        side_stream = torch.cuda.Stream()
        side_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side_stream):
            layer(inputs)
        torch.cuda.current_stream().wait_stream(side_stream)
        torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = layer(inputs)
    for _ in range(3):
        inputs.copy_(torch.randn_like(inputs))
        graph.replay()
        torch.cuda.synchronize()
        reference = inputs.float() @ reference_weight.T
        assert ((outputs.float() - reference).norm() / reference.norm()).item() < TOLERANCE[dtype]
