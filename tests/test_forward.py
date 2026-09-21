import json

import pytest
import torch

import humming.forward as forward_module
from humming import dtypes
from humming.config import LayerConfig, MmaType


@pytest.mark.parametrize(
    "compute_config,tuning_config",
    [
        pytest.param(None, None, id="defaults"),
        pytest.param("", "", id="empty"),
        pytest.param(
            '{"use_batch_invariant": false, "use_f16_accum": false, "gemm_type": "dense"}',
            None,
            id="vllm-defaults",
        ),
        pytest.param('{"use_batch_invariant": true}', None, id="batch-invariant"),
        pytest.param('{"use_f16_accum": true}', None, id="f16-accum"),
        pytest.param('{"use_m_major_input_scale": true}', None, id="m-major-string"),
        pytest.param({"use_m_major_input_scale": True}, None, id="m-major-dict"),
        pytest.param({"gemm_type": "dense"}, {"block_shape_m": 16}, id="dicts"),
        pytest.param(None, [[0, 128, {"block_shape_m": 16}]], id="tuning-list"),
    ],
)
def test_forward_fullgraph_preserves_configs(monkeypatch, compute_config, tuning_config):
    """Keep JSON handling traceable without requiring CUDA kernel compilation."""
    torch._dynamo.reset()
    config = LayerConfig(
        sm_version=100,
        shape_n=64,
        shape_k=32,
        a_dtype=dtypes.bfloat16,
        b_dtype=dtypes.float8e4m3,
        c_dtype=dtypes.bfloat16,
        bs_dtype=dtypes.bfloat16,
        mma_type=MmaType.MMA,
    )
    expected_compute = json.dumps(compute_config) if isinstance(compute_config, dict) else compute_config
    expected_tuning = json.dumps(tuning_config) if isinstance(tuning_config, (dict, list)) else tuning_config
    parsed_compute = json.loads(expected_compute or "{}")
    expected_m_major = bool(parsed_compute.get("use_m_major_input_scale", False))

    def process_input(config, inputs, *, m_major_scale, **kwargs):
        assert m_major_scale == expected_m_major
        return inputs, None, None

    def gemm(*, inputs, weight, compute_config, tuning_config, **kwargs):
        assert compute_config == expected_compute
        assert tuning_config == expected_tuning
        return inputs @ weight.T

    monkeypatch.setattr(forward_module, "may_process_input", process_input)
    monkeypatch.setattr(forward_module.ops, "humming_gemm", gemm)

    def forward(inputs, weight):
        return forward_module.humming_forward(
            config,
            inputs,
            weight,
            compute_config=compute_config,
            tuning_config=tuning_config,
            hadamard_block_size=16,
        )

    inputs = torch.randn(2, 32)
    weight = torch.randn(64, 32)
    expected = forward(inputs, weight)
    compiled = torch.compile(forward, backend="eager", fullgraph=True)
    torch.testing.assert_close(compiled(inputs, weight), expected)
