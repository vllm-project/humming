import pytest
import torch

from humming.config import ProcessInputConfig, ProcessInputTuningConfig
from humming.kernel.process_input import ProcessInputKernel
from humming.ops.input import process_input

from ._reference import _hadamard_reference, _quantize_int8_reference, _round_positive_m3_rne


def launch_config(config, tuning, inputs):
    configs = ProcessInputKernel.prepare_kernels(config, [(0, 1 << 30, tuning)], inputs.device)
    output = torch.empty(
        (inputs.size(0), config.output_row_size), dtype=config.output_torch_dtype, device="cuda"
    )
    groups = None
    if config.quant_mode.has_group_scale:
        from humming import dtypes

        groups = torch.empty(
            config.get_group_scale_shape(inputs.size(0)),
            dtype=dtypes.torch_dtype_map[config.group_scale_dtype],
            device="cuda",
        )
    tokens = torch.empty(inputs.size(0), dtype=torch.float32, device="cuda")
    torch.ops.humming.launch_process_input(configs, inputs, output, groups, tokens, None, None)
    return output, groups, tokens


def test_explicit_dynamic_token_two_stage():
    torch.manual_seed(42)
    x = torch.randn(3, 4096, device="cuda")
    config = ProcessInputConfig(
        input_dtype="float32",
        hidden_size=4096,
        quant_mode="dynamic_token",
        quant_dtype="int8",
        hadamard_block_size=128,
    )
    tuning = ProcessInputTuningConfig(threads_per_task=64, values_per_thread=64, two_stage=True)
    output, _, tokens = launch_config(config, tuning, x)
    source = _hadamard_reference(x, 128)
    expected_scale = source.abs().amax(-1) / 127
    expected = _quantize_int8_reference(source, expected_scale, 4096)
    torch.testing.assert_close(tokens, expected_scale, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(output, expected, rtol=0, atol=1)


@pytest.mark.parametrize("staged", [False, True])
@pytest.mark.parametrize("scale_dtype", ["float32", "float8e4m3"])
def test_explicit_group_token_stages(staged, scale_dtype):
    torch.manual_seed(43)
    x = torch.randn(5, 768, device="cuda")
    config = ProcessInputConfig(
        input_dtype="float32",
        hidden_size=768,
        quant_mode="dynamic_group_token",
        quant_dtype="int8",
        quant_group_size=128,
        group_scale_dtype=scale_dtype,
        use_m_major_input_scale=True,
    )
    tuning = ProcessInputTuningConfig(
        threads_per_task=32 if staged else 128,
        values_per_thread=8,
        use_tile_partition=staged,
        finalize_tokens_per_block=2,
    )
    output, groups, tokens = launch_config(config, tuning, x)
    grouped = x.reshape(5, 6, 128)
    m3 = _round_positive_m3_rne(grouped.abs().amax(-1) / 127)
    expected_token = torch.exp2(torch.ceil(torch.log2(m3.amax(-1) / 448)))
    expected_group = (m3 / expected_token[:, None]).to(groups.dtype)
    if scale_dtype == "float32":
        unpacked = groups[:, :5].T
    else:
        unpacked = groups[:, :5].permute(1, 0, 2).reshape(5, -1)[:, :6]
    torch.testing.assert_close(tokens, expected_token, rtol=0, atol=0)
    torch.testing.assert_close(unpacked, expected_group, rtol=0, atol=0)
    torch.testing.assert_close(output, _quantize_int8_reference(x, m3, 128), rtol=0, atol=1)


def test_launcher_checks_buffers_on_cache_hit():
    x = torch.randn(3, 128, device="cuda")
    process_input(x)
    with pytest.raises(RuntimeError, match="invalid output shape"):
        process_input(x, outputs=torch.empty(3, 256, device="cuda"))
    with pytest.raises(RuntimeError, match="contiguous"):
        process_input(x, outputs=torch.empty(128, 3, device="cuda").T)
    with pytest.raises(RuntimeError, match="invalid dtype"):
        process_input(x, outputs=torch.empty_like(x, dtype=torch.float16))


def test_compile_dynamic_rows_and_inplace():
    compiled = torch.compile(process_input, fullgraph=True, dynamic=True, backend="eager")
    for rows in (3, 129):
        x = torch.randn(rows, 256, device="cuda")
        expected = _hadamard_reference(x, 128)
        output = compiled(x, hadamard_block_size=128)[0]
        torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)
        output = compiled(x, outputs=x, hadamard_block_size=128)[0]
        assert output is x
        torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)


def test_cuda_graph_replay():
    x = torch.randn(5, 256, device="cuda")
    output = torch.empty_like(x)
    process_input(x, outputs=output, hadamard_block_size=128)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        process_input(x, outputs=output, hadamard_block_size=128)
    x.normal_()
    graph.replay()
    torch.testing.assert_close(output, _hadamard_reference(x, 128), rtol=1e-5, atol=1e-5)


def test_launcher_selects_row_interval():
    config = ProcessInputConfig(
        input_dtype="float32", hidden_size=4096, quant_mode="dynamic_token", quant_dtype="int8"
    )
    fused = ProcessInputTuningConfig(threads_per_task=128, values_per_thread=32)
    staged = ProcessInputTuningConfig(threads_per_task=64, values_per_thread=64, two_stage=True)
    configs = ProcessInputKernel.prepare_kernels(
        config, [(0, 3, fused), (3, 1 << 30, staged)], torch.device("cuda", 0)
    )
    for rows in (3, 4):
        x = torch.randn(rows, 4096, device="cuda")
        output = torch.empty_like(x, dtype=torch.int8)
        tokens = torch.empty(rows, device="cuda")
        torch.ops.humming.launch_process_input(configs, x, output, None, tokens, None, None)
        expected_scale = x.abs().amax(-1) / 127
        torch.testing.assert_close(tokens, expected_scale, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(output, _quantize_int8_reference(x, expected_scale, 4096), rtol=0, atol=1)
