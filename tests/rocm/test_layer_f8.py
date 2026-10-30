"""HummingLayer on ROCm with 8-bit / 4-bit activations (gfx950 low-bit MFMA GEMM).

Every case is checked against a torch reference built from the layer's own dequantized weights
(``weight_schema.dequant_tensors`` before ``transform``) and the activations quantized by the
layer's own ``process_input`` (dequantized with the returned scales), so the only error left is
the kernel's FP32 accumulation and the BF16 / FP16 output rounding.
"""

import functools

import pytest
import torch

from humming import ops
from humming.forward import may_quant_input
from humming.layer import HummingLayer

SHAPE_N = 1024
SHAPE_K = 2048
M_VALUES = [1, 7, 64, 255, 1024]
REAL_SHAPES = [(6144, 4096), (4096, 14336), (2880, 2880)]  # (N, K); 2880 is padded to the tile


def mx(dtype: str) -> dict:
    return {"dtype": dtype, "group_size": 32, "scale_dtype": "float8e8m0"}


F8_TOKEN = {"dtype": "float8e4m3"}
F8_G128 = {"dtype": "float8e4m3", "group_size": 128}
F8_G32 = {"dtype": "float8e4m3", "group_size": 32, "scale_dtype": "float32"}
F8_MX = mx("float8e4m3")
E5M2_TOKEN = {"dtype": "float8e5m2"}
I8_TOKEN = {"dtype": "int8"}
FP4_MX = mx("float4e2m1")

# (id, weight_config, input_config)
CASES = [
    # FP8 activations x FP8 weights
    ("e4m3-channel", {"dtype": "float8e4m3"}, F8_TOKEN),
    ("e4m3-tensor", {"dtype": "float8e4m3", "scale_type": "tensor"}, F8_TOKEN),
    ("e4m3-block128-ag128", {"dtype": "float8e4m3", "group_size": 128, "group_size_n": 128}, F8_G128),
    ("e4m3-block128-token", {"dtype": "float8e4m3", "group_size": 128, "group_size_n": 128}, F8_TOKEN),
    ("e4m3-g128", {"dtype": "float8e4m3", "group_size": 128}, F8_TOKEN),
    ("e5m2-channel-a-e5m2", {"dtype": "float8e5m2"}, E5M2_TOKEN),
    ("e4m3-channel-a-e5m2", {"dtype": "float8e4m3"}, E5M2_TOKEN),
    # W4A8 and friends: integer weights dequantized to e4m3
    ("int4-g128", {"dtype": "int4", "group_size": 128}, F8_TOKEN),
    ("int4-g128-ag128", {"dtype": "int4", "group_size": 128}, F8_G128),
    ("int4-g128-ag32-e8m0", {"dtype": "int4", "group_size": 128}, F8_MX),
    ("int4-g32", {"dtype": "int4", "group_size": 32}, F8_TOKEN),
    ("int4-g64", {"dtype": "int4", "group_size": 64}, F8_TOKEN),
    ("int4-g256", {"dtype": "int4", "group_size": 256}, F8_TOKEN),
    ("int4-channel", {"dtype": "int4"}, F8_TOKEN),
    ("uint4-zp-g128", {"dtype": "uint4", "group_size": 128, "has_zero_point": True}, F8_TOKEN),
    ("uint3-zp-g64", {"dtype": "uint3", "group_size": 64, "has_zero_point": True}, F8_TOKEN),
    ("int5-g128", {"dtype": "int5", "group_size": 128}, F8_TOKEN),
    ("int3-g128", {"dtype": "int3", "group_size": 128}, F8_TOKEN),
    ("int2-g128", {"dtype": "int2", "group_size": 128}, F8_TOKEN),
    # MX / FP weights with FP8 activations
    ("mxfp4", FP4_MX, F8_TOKEN),
    ("mxfp4-ag32-e8m0", FP4_MX, F8_MX),
    ("mxfp4-ag32-fp32", FP4_MX, F8_G32),
    ("mxfp4-ag128", FP4_MX, F8_G128),
    ("mxfp6-e2m3", mx("float6e2m3"), F8_TOKEN),
    ("mxfp6-e3m2", mx("float6e3m2"), F8_TOKEN),
    ("mxfp8", mx("float8e4m3"), F8_TOKEN),
    ("fp4-g128-bf16-scale", {"dtype": "float4e2m1", "group_size": 128}, F8_TOKEN),
    ("fp4-channel", {"dtype": "float4e2m1"}, F8_TOKEN),
    # MXFP4 activations (native mixed formats)
    ("a-mxfp4-mxfp4", FP4_MX, FP4_MX),
    ("a-mxfp4-mxfp6", mx("float6e2m3"), FP4_MX),
    ("a-mxfp4-mxfp8", mx("float8e4m3"), FP4_MX),
    ("a-mxfp4-int4-g128", {"dtype": "int4", "group_size": 128}, FP4_MX),
    # INT8 activations
    ("a-int8-int8", {"dtype": "int8"}, I8_TOKEN),
    ("a-int8-int4-g128", {"dtype": "int4", "group_size": 128}, I8_TOKEN),
    ("a-int8-int4-channel", {"dtype": "int4"}, I8_TOKEN),
    ("a-int8-int4-g256", {"dtype": "int4", "group_size": 256}, I8_TOKEN),
    ("a-int8-uint4-zp-g128", {"dtype": "uint4", "group_size": 128, "has_zero_point": True}, I8_TOKEN),
    ("a-int8-mxfp4", FP4_MX, I8_TOKEN),
]
CASE_BY_ID = {case_id: (weight, inputs) for case_id, weight, inputs in CASES}

# One case per kernel family; M values hit every tile bucket of select_f8_tile (64-row tiles with
# three stages, 128-row tiles, 2 x 2-wave FP4 tiles) with partial M and N tiles (N = 2112).
TILE_BUCKET_CASES = [
    "e4m3-channel",
    "int4-g128",
    "int4-g256",
    "uint3-zp-g64",
    "mxfp4",
    "a-mxfp4-mxfp4",
    "a-int8-int8",
    "a-int8-int4-g128",
]
TILE_BUCKET_M = [100, 300, 2049, 3000]

REAL_SHAPE_CASES = [
    "e4m3-channel",
    "e4m3-block128-ag128",
    "int4-g64",
    "mxfp4",
    "a-mxfp4-mxfp4",
    "a-int8-int8",
]

# Configs that must raise NotImplementedError on ROCm (never run with wrong results).
UNSUPPORTED_CASES = [
    ("nvfp4", {"dtype": "float4e2m1", "group_size": 16, "scale_dtype": "float8e4m3"}, F8_TOKEN),
    ("a-bf16-e4m3-weight", {"dtype": "float8e4m3"}, None),
    ("a-int8-uint8-zp", {"dtype": "uint8", "group_size": 128, "has_zero_point": True}, I8_TOKEN),
]


def dequant_activation(quantized: torch.Tensor, scale: torch.Tensor, shape_k: int) -> torch.Tensor:
    if quantized.dtype == torch.uint8:  # packed FP4 e2m1, low nibble first
        codes = torch.stack([quantized & 0xF, quantized >> 4], dim=-1).reshape(quantized.size(0), shape_k)
        values = ops.dequant_weight(codes.to(torch.int32), 2, 1, True).float()
    else:
        values = quantized.float()
    if scale.numel() == 1:
        return values * scale.float()
    if scale.dtype in (torch.int32, torch.uint8, torch.float8_e8m0fnu):  # packed E8M0 group scales
        exponents = scale.contiguous().view(torch.uint8).float()
        num_groups = exponents.size(1)
        if shape_k % num_groups != 0 or shape_k // num_groups < 32:
            num_groups = shape_k // 32
        factors = torch.exp2(exponents[:, :num_groups] - 127)
        return values * factors.repeat_interleave(shape_k // num_groups, dim=1)
    if scale.size(1) == 1:
        return values * scale.float()
    return values * scale.float().repeat_interleave(shape_k // scale.size(1), dim=1)


@functools.cache
def build_layer(
    case_id: str, shape_n: int, shape_k: int, torch_dtype: torch.dtype = torch.bfloat16, bias: bool = False
):
    weight_config, input_config = CASE_BY_ID[case_id]
    torch.manual_seed(2026)
    layer = HummingLayer(
        shape_n=shape_n,
        shape_k=shape_k,
        weight_config=dict(weight_config),
        input_config=dict(input_config),
        torch_dtype=torch_dtype,
        has_bias=bias,
    ).cuda()
    layer.load_from_unquantized(torch.randn(shape_n, shape_k, dtype=torch_dtype, device="cuda") * 0.05)
    if bias:
        layer.bias.data.normal_()
    weight = layer.weight_schema.dequant_tensors(dict(layer.state_dict())).float()
    layer.transform()
    return layer, weight


def reference(layer: HummingLayer, weight: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    quantized, scale = may_quant_input(layer.humming_config, inputs)
    result = dequant_activation(quantized, scale, inputs.size(1)) @ weight.T
    if getattr(layer, "bias", None) is not None:
        result = result + layer.bias.float()
    return result


def relative_error(outputs: torch.Tensor, expected: torch.Tensor) -> float:
    return ((outputs.float() - expected).norm() / expected.norm().clamp_min(1e-30)).item()


def check(case_id: str, shape_m: int, shape_n: int, shape_k: int, torch_dtype=torch.bfloat16, bias=False):
    layer, weight = build_layer(case_id, shape_n, shape_k, torch_dtype, bias)
    inputs = torch.randn(shape_m, shape_k, dtype=torch_dtype, device="cuda")
    outputs = layer(inputs)
    assert outputs.shape == (shape_m, shape_n) and outputs.dtype == torch_dtype
    tolerance = 5e-3 if torch_dtype == torch.bfloat16 else 1e-3
    error = relative_error(outputs, reference(layer, weight, inputs))
    assert error < tolerance, f"{case_id} M={shape_m} N={shape_n} K={shape_k}: relative error {error:.3g}"


@pytest.mark.parametrize("shape_m", M_VALUES)
@pytest.mark.parametrize("case_id", list(CASE_BY_ID))
def test_layer_f8(case_id: str, shape_m: int):
    check(case_id, shape_m, SHAPE_N, SHAPE_K)


@pytest.mark.parametrize("shape_m", TILE_BUCKET_M)
@pytest.mark.parametrize("case_id", TILE_BUCKET_CASES)
def test_layer_f8_tile_buckets(case_id: str, shape_m: int):
    check(case_id, shape_m, 2112, SHAPE_K)


@pytest.mark.parametrize("shape_m", [1, 16, 1024])
@pytest.mark.parametrize(("shape_n", "shape_k"), REAL_SHAPES, ids=str)
@pytest.mark.parametrize("case_id", REAL_SHAPE_CASES)
def test_layer_f8_real_shapes(case_id: str, shape_n: int, shape_k: int, shape_m: int):
    weight_config, input_config = CASE_BY_ID[case_id]
    for group, size in (
        (weight_config.get("group_size", 0), shape_k),
        (weight_config.get("group_size_n", 0), shape_n),
    ):
        if group and size % group:
            pytest.skip(f"the weight quantizer needs {size} to be a multiple of the group size {group}")
    if shape_k % 128 and input_config.get("group_size"):
        with pytest.raises(NotImplementedError, match="Humming on ROCm"):
            check(case_id, shape_m, shape_n, shape_k)
        return
    check(case_id, shape_m, shape_n, shape_k)


@pytest.mark.parametrize(
    "case_id", ["e4m3-channel", "int4-g128", "mxfp4", "a-int8-int4-g128", "a-mxfp4-mxfp4"]
)
def test_layer_f8_float16_output(case_id: str):
    check(case_id, 64, SHAPE_N, SHAPE_K, torch.float16)


@pytest.mark.parametrize("case_id", ["int4-g128", "a-int8-int8"])
def test_layer_f8_bias(case_id: str):
    check(case_id, 7, SHAPE_N, SHAPE_K, bias=True)


@pytest.mark.parametrize(
    "weight_config", [{"dtype": "int4", "group_size": 128}, {"dtype": "float8e4m3"}], ids=str
)
def test_layer_f8_static_tensor_scale(weight_config):
    layer = HummingLayer(
        shape_n=SHAPE_N,
        shape_k=SHAPE_K,
        weight_config=dict(weight_config),
        input_config={"dtype": "float8e4m3", "quant_mode": "static_tensor"},
        torch_dtype=torch.bfloat16,
    ).cuda()
    layer.load_from_unquantized(torch.randn(SHAPE_N, SHAPE_K, dtype=torch.bfloat16, device="cuda") * 0.05)
    layer.input_scale.data.fill_(4.5 / 448)
    weight = layer.weight_schema.dequant_tensors(dict(layer.state_dict())).float()
    layer.transform()
    inputs = torch.randn(64, SHAPE_K, dtype=torch.bfloat16, device="cuda")
    scale = layer.input_scale.float()
    quantized = (inputs.float() / scale).clamp(-448, 448).to(torch.float8_e4m3fn).float() * scale
    assert relative_error(layer(inputs), quantized @ weight.T) < 5e-3


@pytest.mark.parametrize("shape_m", [1, 255])
@pytest.mark.parametrize("case_id", ["int4-g128", "mxfp4-ag32-e8m0", "a-int8-int8"])
def test_layer_f8_graph_capture(case_id: str, shape_m: int):
    """Input quantization + GEMM (including split-K) captured in a graph and replayed on new data."""
    layer, weight = build_layer(case_id, SHAPE_N, SHAPE_K)
    static_inputs = torch.randn(shape_m, SHAPE_K, dtype=torch.bfloat16, device="cuda")
    layer(static_inputs)  # compile + warm up outside the capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_outputs = layer(static_inputs)
    for _ in range(2):
        static_inputs.copy_(torch.randn_like(static_inputs))
        graph.replay()
        torch.cuda.synchronize()
        error = relative_error(static_outputs, reference(layer, weight, static_inputs))
        assert error < 5e-3, f"graph replay relative error {error:.3g}"


@pytest.mark.parametrize(
    ("case_id", "weight_config", "input_config"), UNSUPPORTED_CASES, ids=[c[0] for c in UNSUPPORTED_CASES]
)
def test_layer_f8_unsupported_raises(case_id, weight_config, input_config):
    layer = HummingLayer(
        shape_n=SHAPE_N,
        shape_k=SHAPE_K,
        weight_config=dict(weight_config),
        input_config=input_config and dict(input_config),
        torch_dtype=torch.bfloat16,
    ).cuda()
    layer.load_from_unquantized(torch.randn(SHAPE_N, SHAPE_K, dtype=torch.bfloat16, device="cuda"))
    inputs = torch.randn(16, SHAPE_K, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(NotImplementedError, match="Humming on ROCm"):
        layer.transform()
        layer(inputs)
