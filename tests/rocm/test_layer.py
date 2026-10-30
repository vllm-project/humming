"""HummingLayer on ROCm: integer weight-only layers run the MFMA GEMM; unsupported configs raise.

8-bit / 4-bit activation layers are covered by test_layer_f8.py.
"""

import pytest
import torch

from humming.layer import HummingLayer

SHAPE_N = 1024
SHAPE_K = 2048

SUPPORTED_CASES = [{"dtype": "int4"}, {"dtype": "int8"}]
UNSUPPORTED_CASES = [
    ({"dtype": "float8e4m3"}, None),
    ({"dtype": "int4", "group_size": 16}, None),
]


def make_layer(weight_config, input_config=None):
    layer = HummingLayer(
        shape_n=SHAPE_N,
        shape_k=SHAPE_K,
        weight_config=weight_config,
        torch_dtype=torch.bfloat16,
        input_config=input_config,
    ).cuda()
    layer.load_from_unquantized(torch.randn(SHAPE_N, SHAPE_K, dtype=torch.bfloat16, device="cuda"))
    return layer


@pytest.mark.parametrize("weight_config", SUPPORTED_CASES, ids=str)
def test_layer_transforms_and_runs(weight_config):
    layer = make_layer(dict(weight_config))
    reference_weight = layer.weight_schema.dequant_tensors(dict(layer.state_dict())).float()
    layer.transform()
    for parameter_name, parameter in layer.named_parameters():
        assert parameter.is_cuda, parameter_name

    inputs = torch.randn(16, SHAPE_K, dtype=torch.bfloat16, device="cuda")
    outputs = layer(inputs)
    reference = inputs.float() @ reference_weight.T
    assert outputs.shape == (16, SHAPE_N) and outputs.dtype == torch.bfloat16
    assert ((outputs.float() - reference).norm() / reference.norm()).item() < 1e-2


@pytest.mark.parametrize(("weight_config", "input_config"), UNSUPPORTED_CASES, ids=str)
def test_unsupported_layer_raises(weight_config, input_config):
    layer = make_layer(dict(weight_config), input_config and dict(input_config))
    inputs = torch.randn(16, SHAPE_K, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(NotImplementedError, match="Humming on ROCm"):
        layer.transform()
        layer(inputs)
