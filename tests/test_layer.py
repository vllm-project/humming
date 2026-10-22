import json

import pytest
import torch
from safetensors.torch import save_file

from humming.layer import HummingLayer

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="HummingLayer requires CUDA")

LAYER_PREFIX = "model.layers.0.self_attn.q_proj"
SHAPE_N = 128
SHAPE_K = 128


def _write_gptq_checkpoint(tmp_path, dynamic, bits, group_size):
    config = {
        "torch_dtype": "float16",
        "quantization_config": {
            "quant_method": "gptq",
            "bits": 4,
            "group_size": 64,
            "sym": True,
            "desc_act": False,
            "dynamic": dynamic,
        },
    }
    (tmp_path / "config.json").write_text(json.dumps(config))

    input_indices = torch.arange(SHAPE_K)[:, None]
    output_indices = torch.arange(SHAPE_N)[None, :]
    quantized = (input_indices * 5 + output_indices * 3) % (1 << bits)
    pack_factor = 32 // bits
    shifts = torch.arange(pack_factor) * bits
    packed = quantized.reshape(SHAPE_K // pack_factor, pack_factor, SHAPE_N)
    qweight = (packed << shifts[None, :, None]).sum(dim=1).to(torch.int32)

    group_indices = torch.arange(SHAPE_K // group_size)[:, None]
    scales = ((group_indices + output_indices % 5 + 1) / 128).to(torch.float16)
    midpoint = 1 << (bits - 1)
    packed_zero = sum(midpoint << shift for shift in range(0, 32, bits))
    qzeros = torch.full((SHAPE_K // group_size, SHAPE_N // pack_factor), packed_zero, dtype=torch.int64).to(
        torch.int32
    )
    tensors = {
        "qweight": qweight,
        "qzeros": qzeros,
        "scales": scales,
        "g_idx": torch.arange(SHAPE_K, dtype=torch.int32) // group_size,
    }
    save_file(
        {f"{LAYER_PREFIX}.{name}": value for name, value in tensors.items()}, tmp_path / "model.safetensors"
    )

    # GPTQ symmetric codes represent (code - 2**(bits - 1)) * group scale.
    weight = (quantized.float() - midpoint) * scales.float().repeat_interleave(group_size, dim=0)
    return tensors, weight.T.contiguous()


@pytest.mark.parametrize(
    ("dynamic", "bits", "group_size"),
    [
        pytest.param({r"+:^model\.layers\.0\.": {"bits": 8, "group_size": 32}}, 8, 32, id="positive-prefix"),
        pytest.param({r"^model\.layers\.0\.": {"bits": 8, "group_size": 32}}, 8, 32, id="bare-regex"),
        pytest.param({r"+:^model\.": {"bits": 8}}, 8, 64, id="bits-preserves-default-group-size"),
        pytest.param({r"+:^model\.": {"group_size": 32}}, 4, 32, id="group-size-preserves-default-bits"),
        pytest.param(
            {r"+:^model\.layers\.1\.": {"bits": 8, "group_size": 32}}, 4, 64, id="unmatched-defaults"
        ),
        pytest.param({}, 4, 64, id="no-dynamic-rules"),
        pytest.param(
            {
                r"+:^model\.layers\.0\.": {"bits": 8, "group_size": 32},
                r"+:^model\.": {"bits": 4, "group_size": 64},
            },
            8,
            32,
            id="first-matching-positive-wins",
        ),
        pytest.param(
            {r"+:^model\.": {"bits": 8, "group_size": 32}, r"-:^model\.layers\.0\.": {}},
            8,
            32,
            id="stored-rule-order-is-preserved",
        ),
    ],
)
def test_from_safetensors_applies_gptq_dynamic_overrides(tmp_path, dynamic, bits, group_size):
    tensors, _ = _write_gptq_checkpoint(tmp_path, dynamic, bits, group_size)

    layer = HummingLayer.from_safetensors(str(tmp_path), prefix=LAYER_PREFIX)

    assert layer.weight_schema.bits == bits
    assert layer.weight_schema.group_size == group_size
    assert (layer.shape_n, layer.shape_k) == (SHAPE_N, SHAPE_K)
    for name, tensor in tensors.items():
        torch.testing.assert_close(getattr(layer, name), tensor, rtol=0, atol=0)


@pytest.mark.parametrize(
    "dynamic",
    [
        pytest.param({r"-:^model\.layers\.0\.": {}}, id="excluded-layer"),
        pytest.param(
            {r"-:^model\.layers\.0\.": {}, r"+:^model\.": {"bits": 8}},
            id="exclusion-before-positive-match",
        ),
    ],
)
def test_from_safetensors_rejects_gptq_dynamic_exclusion(tmp_path, dynamic):
    _write_gptq_checkpoint(tmp_path, dynamic, bits=4, group_size=64)
    save_file(
        {f"{LAYER_PREFIX}.weight": torch.ones((SHAPE_N, SHAPE_K), dtype=torch.float16)},
        tmp_path / "model.safetensors",
    )

    with pytest.raises(AssertionError, match="is unquantized"):
        HummingLayer.from_safetensors(str(tmp_path), prefix=LAYER_PREFIX)


@pytest.mark.parametrize(
    ("regex", "bits", "group_size"),
    [(r"+:^model\.layers\.0\.", 8, 32), (r"^model\.layers\.0\.", 4, 32)],
    ids=["positive-prefix", "bare-regex"],
)
def test_gptq_dynamic_checkpoint_forward_matches_dequantized_weight(tmp_path, regex, bits, group_size):
    dynamic = {regex: {"bits": bits, "group_size": group_size}}
    _, weight = _write_gptq_checkpoint(tmp_path, dynamic, bits, group_size)
    layer = HummingLayer.from_safetensors(str(tmp_path), prefix=LAYER_PREFIX).cuda()
    layer.transform()
    inputs = ((torch.arange(3 * SHAPE_K).reshape(3, SHAPE_K) * 3 % 17 - 8) / 8).to(torch.float16)
    expected = torch.nn.functional.linear(inputs.float(), weight)

    actual = layer(inputs.cuda())

    torch.testing.assert_close(actual.cpu().float(), expected, rtol=0.01, atol=0.02)
