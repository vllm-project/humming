import dataclasses

import pytest
import torch

from humming import dtypes
from humming.config import GemmType
from humming.config.ldmatrix_s4 import assert_layout_compatible
from humming.kernel.humming import HummingKernel
from humming.testing import skip_if_unsupported
from humming.testing.ldmatrix_s4 import (
    PreparedDense,
    assert_output,
    compare_outputs,
    input_data,
    make_config,
    record_rejection,
    record_result,
    record_selection,
    weight_data,
)
from humming.transform import transform_humming_weight
from humming.tune import get_heuristics_config


@pytest.fixture(autouse=True)
def eligible_environment():
    skip_if_unsupported(mma_type="wgmma", min_cuda_version=(13, 4))
    config = make_config()
    if not config.can_use_ldmatrix_s4:
        pytest.skip(str(config.ldmatrix_s4_rejection_reasons))
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = previous


@pytest.mark.parametrize(
    "m,n,k,random,nonuniform",
    [
        (1, 128, 128, False, False),
        (127, 128, 128, False, False),
        (128, 128, 128, False, True),
        (256, 4096, 4096, True, True),
    ],
)
def test_dense_production_parity(m, n, k, random, nonuniform):
    data, reference = weight_data(n, k, random=random, nonuniform=nonuniform)
    new = PreparedDense.create(make_config(n, k), data, reference)
    legacy = PreparedDense.create(make_config(n, k, legacy=True), data, reference)
    inputs = input_data(m, k)
    tuning = get_heuristics_config(new.config, shape_m=m, gemm_type="dense")
    outputs, kernels = [], []
    for prepared in (new, legacy):
        output, kernel = prepared.run(inputs, tuning=tuning)
        assert_output(output, prepared.oracle(inputs), kernel)
        outputs.append(output)
        kernels.append(kernel)
    torch.testing.assert_close(*outputs, rtol=0.01, atol=0.05)
    for output, kernel in zip(outputs, kernels, strict=True):
        report = compare_outputs(
            output,
            outputs[1],
            new.oracle(inputs),
            kernel,
            m=m,
            n=n,
            k=k,
            scale_pattern="nonuniform" if nonuniform else "unit",
        )
        record_result(f"m{m}-n{n}-k{k}-{report['loader_mode']}", report)


def test_layout_cache_and_mode_switching():
    data, reference = weight_data(128, 256, nonuniform=True)
    inputs = input_data(16, 256)
    layouts, specializations, repacked = {}, {}, {}
    for index, force_legacy in enumerate((True, False, False, True)):
        config = make_config(128, 256, legacy=force_legacy)
        prepared = PreparedDense.create(config, data, reference)
        # Match the exact new-path tuning for an attributable comparison.
        tuning = get_heuristics_config(make_config(128, 256), shape_m=16, gemm_type="dense")
        output, kernel = prepared.run(inputs, tuning=tuning)
        assert_output(output, prepared.oracle(inputs), kernel)
        if force_legacy in specializations:
            assert specializations[force_legacy] == kernel.specialization_identity
        repacked[force_legacy] = prepared.tensors["weight"]
        record_selection(f"switch-{index}", kernel, "force_legacy" if force_legacy else "automatic")
        layouts[force_legacy] = prepared.layout_identity
        specializations[force_legacy] = kernel.specialization_identity
    assert not torch.equal(repacked[True], repacked[False])
    assert layouts[True] != layouts[False]
    assert specializations[True] != specializations[False]
    with pytest.raises(ValueError, match="does not match"):
        assert_layout_compatible(layouts[True], layouts[False])


def test_compact_batch_and_alternation_stability():
    data, reference = weight_data(4096, 4096, random=True, nonuniform=True)
    target = input_data(8, 4096)
    surrounding = input_data(16, 4096, seed=1)
    other = input_data(8, 4096, seed=2)
    # Share automatic tuning to isolate loader/repack stability.
    auto_config = make_config(4096, 4096)

    def tuning_for(rows):
        return get_heuristics_config(auto_config, shape_m=rows, gemm_type="dense")

    baseline = PreparedDense.create(make_config(4096, 4096, legacy=True), data, reference)
    legacy_outputs = {
        0: baseline.run(target, tuning=tuning_for(8))[0],
        1: baseline.run(other, tuning=tuning_for(8))[0],
    }
    for legacy in (False, True):
        prepared = PreparedDense.create(make_config(4096, 4096, legacy=legacy), data, reference)
        alone, kernel = prepared.run(target, tuning=tuning_for(8))
        assert_output(alone, prepared.oracle(target), kernel)
        for batch, start in ((torch.cat((target, surrounding)), 0), (torch.cat((surrounding, target)), 16)):
            output, _ = prepared.run(batch, tuning=tuning_for(batch.shape[0]))
            torch.testing.assert_close(output[start : start + 8], alone, rtol=0.01, atol=0.05)
        for index in range(12):
            inputs = target if index % 2 == 0 else other
            output, kernel = prepared.run(inputs, tuning=tuning_for(inputs.shape[0]))
            assert_output(output, prepared.oracle(inputs), kernel)
        report = compare_outputs(
            output,
            legacy_outputs[index % 2],
            prepared.oracle(inputs),
            kernel,
            m=8,
            n=4096,
            k=4096,
            scale_pattern="nonuniform",
        )
        report.update(alternating_invocations=12, positions=["alone", "beginning", "tail"])
        record_result(f"stability-{report['loader_mode']}", report)


@pytest.mark.parametrize("mode", [GemmType.INDEXED, GemmType.GROUPED_CONTIGUOUS, GemmType.GROUPED_MASKED])
def test_moe_excluded_and_dense_layout_cannot_be_reinterpreted(mode):
    assert make_config(num_experts=4).use_ldmatrix_s4 is False
    config = make_config()
    with pytest.raises(ValueError, match="dense execution"):
        get_heuristics_config(config, shape_m=16, gemm_type=mode)
    record_rejection(f"dense-to-{mode.value}", "prepared dense layout cannot use MoE")


@pytest.mark.parametrize(
    "change",
    [
        {"use_tma_b": False},
        {"block_shape": (16, 128, 128)},
        {"warp_shape": (16, 32, 128)},
        {"use_stream_k": True},
    ],
)
def test_incompatible_specialization_rejected_before_compilation(change, monkeypatch):
    config = make_config()
    tuning = get_heuristics_config(config, shape_m=16, gemm_type="dense") | change

    def unexpected_compile(*args, **kwargs):
        pytest.fail("incompatible specialization reached compilation")

    monkeypatch.setattr(HummingKernel, "prepare", unexpected_compile)
    with pytest.raises((ValueError, AssertionError)):
        HummingKernel.prepare_kernels(config.to_str(), {"gemm_type": "dense"}, tuning)
    record_rejection(
        "tuning-" + next(iter(change)), "incompatible specialization rejected before compilation"
    )


def test_wrong_orientation_and_moe_repack_rejected():
    data, _ = weight_data(128, 128)
    for index, weights in enumerate((data["weight"].T, data["weight"].unsqueeze(0))):
        with pytest.raises((ValueError, AssertionError)):
            transform_humming_weight(
                weights,
                make_config().b_dtype,
                make_config().a_dtype,
                packed=True,
                use_wgmma=True,
                use_packed_k_layout=True,
                use_ldmatrix_s4=True,
            )

        record_rejection(f"physical-input-{index}", "non-NK or MoE weights rejected")


def test_graph_capture_rejected_for_new_loader():
    data, reference = weight_data(128, 128)
    prepared = PreparedDense.create(make_config(), data, reference)
    inputs = input_data(16, 128)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        prepared.run(inputs)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with pytest.raises(RuntimeError, match="does not support CUDA Graph capture"):
        with torch.cuda.graph(graph, stream=stream):
            prepared.run(inputs)
    record_rejection("graph-capture", "new loader capture rejected at native launch")


def test_force_legacy_does_not_mutate_prepared_config():
    new = make_config()
    old = dataclasses.replace(new, test_force_packed_k_legacy=True)
    assert new.use_ldmatrix_s4 and not old.use_ldmatrix_s4
    assert new.b_layout_identity != old.b_layout_identity


def test_nonuniform_scale_oracle_detects_wrong_group():
    data, reference = weight_data(128, 256, random=True, nonuniform=True)
    inputs = input_data(16, 256)
    codes = data["weight"].long().unsqueeze(-1)
    shifts = torch.arange(8, device="cuda") * 4
    logical = ((codes >> shifts) & 15).reshape(128, 256).float() - 8
    wrong = logical * data["weight_scale"].flip(-1).float().repeat_interleave(128, dim=1)
    expected = ((inputs.float() / 128) @ reference.T).to(torch.bfloat16)
    corrupted = ((inputs.float() / 128) @ wrong.T).to(torch.bfloat16)
    assert not torch.allclose(expected, corrupted, rtol=0.01, atol=0.05)


@pytest.mark.parametrize("mode", ["dense", "indexed", "grouped_contiguous", "grouped_masked"])
def test_selection_metadata(mode):
    config = make_config(num_experts=0 if mode == "dense" else 4)
    tuning = get_heuristics_config(make_config(), shape_m=16, gemm_type="dense")
    if mode == "indexed":
        tuning.update(use_tma_a=False, use_tma_c=False, use_tma_as=False, use_tma_as2=False)
    configs = HummingKernel.prepare_kernels(config.to_str(), {"gemm_type": mode}, tuning).reshape(-1, 4)
    kernel = HummingKernel._id2kernel[int(configs[0, 2])]
    assert kernel.selected_loader_variant == ("ldmatrix_s8_s4" if mode == "dense" else "packed_k_legacy")
    record_selection(f"select-{mode}", kernel, "automatic")
    if mode == "dense":
        legacy = make_config(legacy=True)
        configs = HummingKernel.prepare_kernels(legacy.to_str(), {"gemm_type": mode}, tuning).reshape(-1, 4)
        kernel = HummingKernel._id2kernel[int(configs[0, 2])]
        assert kernel.selected_loader_variant == "packed_k_legacy"
        record_selection("select-force-legacy", kernel, "force_legacy")


@pytest.mark.parametrize(
    "name,changes",
    [
        ("target", {"sm_version": 100}),
        ("packing", {"use_packed_k_layout": False}),
        ("zero-point", {"has_zero_point": True}),
        ("orientation-padding", {"pad_shape_n": 8}),
        ("n127", {"shape_n": 127}),
        ("n129", {"shape_n": 129}),
        ("k127", {"shape_k": 127}),
        ("k129", {"shape_k": 129}),
        ("fp4", {"b_dtype": dtypes.float4e2m1}),
        ("fp8", {"b_dtype": dtypes.float8e4m3}),
        ("int8", {"b_dtype": dtypes.int8}),
        # input_quant_mode=None lets LayerConfig re-resolve to Disabled for the
        # 16-bit activation (int8's dynamic-token mode would violate the
        # has_input_scale invariant), yielding a valid ineligible A16 config.
        ("wna16", {"a_dtype": dtypes.bfloat16, "use_packed_k_layout": False, "input_quant_mode": None}),
        ("preprocess", {"use_int_weight_scale": True, "weight_scale_group_size_n": 1}),
    ],
)
def test_ineligible_layer_cannot_force_new(name, changes):
    config = dataclasses.replace(make_config(), use_ldmatrix_s4=None, **changes)
    assert not config.use_ldmatrix_s4
    with pytest.raises(ValueError, match="ineligible") as caught:
        dataclasses.replace(config, use_ldmatrix_s4=True)
    record_rejection(f"layer-{name}", caught.value)


@pytest.mark.parametrize("compiler,version", [("NVCCCompiler", (13, 3)), ("NVRTCCompiler", (13, 4))])
def test_ineligible_toolchain_cannot_force_new(compiler, version, monkeypatch):
    import humming.config.config as config_module
    from humming.jit.runtime import KernelRuntime

    monkeypatch.setattr(KernelRuntime, "_get_compiler", staticmethod(lambda: type(compiler, (), {})))
    monkeypatch.setattr(config_module, "_cuda_compiler_version", lambda cls: version)
    config = make_config()
    assert not config.use_ldmatrix_s4
    with pytest.raises(ValueError, match="toolchain_ptx") as caught:
        dataclasses.replace(config, use_ldmatrix_s4=True)
    record_rejection(f"toolchain-{compiler}", caught.value)


def test_misaligned_repacked_weight_rejected():
    data, reference = weight_data(128, 128)
    prepared = PreparedDense.create(make_config(), data, reference)
    original = prepared.tensors["weight"]
    storage = torch.empty(original.numel() + 1, device="cuda", dtype=torch.int32)
    misaligned = storage[1:].reshape(original.shape)
    misaligned.copy_(original)
    assert misaligned.is_contiguous() and misaligned.data_ptr() % 16 != 0
    prepared.tensors["weight"] = misaligned
    with pytest.raises(RuntimeError, match="16-byte alignment"):
        prepared.run(input_data(16, 128))
    record_rejection("repacked-alignment", "native launch rejected misaligned B before execution")


@pytest.mark.parametrize("routing", ["sorted_ids", "expert_ids", "num_tokens_padded", "expert_layout"])
def test_native_moe_routing_rejected_for_dense_loader(routing):
    from humming import ops

    data, reference = weight_data(128, 128)
    prepared = PreparedDense.create(make_config(), data, reference)
    inputs = input_data(16, 128)
    _, kernel = prepared.run(inputs)
    with pytest.raises(RuntimeError, match="does not support MoE/EP routing metadata"):
        ops.launch_kernel(
            configs=torch.tensor([kernel.kernel_id], dtype=torch.int64),
            inputs=inputs,
            **prepared.tensors,
            input_scale=torch.full((16, 1), 1 / 128, device="cuda"),
            **{routing: torch.zeros(1, device="cuda", dtype=torch.int32)},
        )
    record_rejection(f"routing-{routing}", "native launch rejected MoE/EP routing metadata")
