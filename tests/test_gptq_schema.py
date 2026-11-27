import json

import pytest
import safetensors.torch
import torch

from humming import ops
from humming.layer import HummingLayer
from humming.schema import BaseWeightSchema
from humming.schema.autoround import AutoRoundWeightSchema
from humming.schema.awq import AWQWeightSchema
from humming.schema.gptq import GPTQWeightSchema


def _pack_codes_cpu(codes: torch.Tensor, num_bits: int) -> torch.Tensor:
    """Bit-pack unsigned integer codes along the last dimension into int32 words on CPU."""
    mask = (1 << num_bits) - 1
    flat_rows = codes.reshape(-1, codes.size(-1)).to(torch.int64).tolist()
    packed_rows = []
    for row in flat_rows:
        words = []
        buffer = 0
        buffered_bits = 0
        for code in row:
            buffer |= (code & mask) << buffered_bits
            buffered_bits += num_bits
            while buffered_bits >= 32:
                word = buffer & 0xFFFFFFFF
                words.append(word - (1 << 32) if word >= 1 << 31 else word)
                buffer >>= 32
                buffered_bits -= 32
        assert buffered_bits == 0
        packed_rows.append(words)
    output_shape = (*codes.shape[:-1], codes.size(-1) * num_bits // 32)
    return torch.tensor(packed_rows, dtype=torch.int32).reshape(output_shape)


def _build_gptq_checkpoint_tensors(
    shape_n: int,
    shape_k: int,
    bits: int,
    group_size: int,
    param_dtype: torch.dtype,
    checkpoint_format: str,
    sym: bool = False,
    num_experts: int | None = None,
    seed: int = 2026,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """Build GPTQ-format tensors plus true unsigned zero points and reference FP32 weights."""
    generator = torch.Generator().manual_seed(seed)
    max_code = (1 << bits) - 1
    num_groups = shape_k // group_size
    leading = (num_experts,) if num_experts is not None else ()

    qweight_codes = torch.randint(
        0,
        max_code + 1,
        (*leading, shape_k, shape_n),
        dtype=torch.int32,
        generator=generator,
    )
    scales = (
        torch.rand((*leading, num_groups, shape_n), dtype=torch.float32, generator=generator) * 0.08 + 0.02
    ).to(param_dtype)

    if sym:
        true_zp = torch.full((*leading, num_groups, shape_n), 1 << (bits - 1), dtype=torch.int32)
    else:
        true_zp = torch.randint(
            0,
            max_code + 1,
            (*leading, num_groups, shape_n),
            dtype=torch.int32,
            generator=generator,
        )
        # Ensure boundary codes 0, 1, and max_code are always exercised.
        flat_zp = true_zp.reshape(-1)
        boundary_values = [0, 1, max_code - 1, max_code]
        for idx, value in enumerate(boundary_values):
            if idx < flat_zp.numel():
                flat_zp[idx] = value

    stored_zp = (true_zp - 1) & max_code if checkpoint_format == "gptq" else true_zp

    qweight_nk = qweight_codes.transpose(-1, -2).contiguous()
    packed_qweight = _pack_codes_cpu(qweight_nk, bits).transpose(-1, -2).contiguous()
    packed_qzeros = _pack_codes_cpu(stored_zp, bits)
    g_idx = torch.arange(shape_k, dtype=torch.int32) // group_size
    if num_experts is not None:
        g_idx = g_idx.unsqueeze(0).expand(num_experts, -1).contiguous()

    expanded_zp = true_zp.repeat_interleave(group_size, dim=-2)
    expanded_scales = scales.float().repeat_interleave(group_size, dim=-2)
    ref_weight_kn = (qweight_codes.float() - expanded_zp.float()) * expanded_scales
    ref_weight_nk = ref_weight_kn.transpose(-1, -2).contiguous()

    tensors = {
        "qweight": packed_qweight,
        "scales": scales,
        "qzeros": packed_qzeros,
        "g_idx": g_idx,
    }
    return tensors, true_zp, ref_weight_nk


def test_gptq_from_config_checkpoint_format() -> None:
    default_schema = BaseWeightSchema.from_config(
        {"quant_method": "gptq", "bits": 4, "group_size": 128, "sym": False}
    )
    assert isinstance(default_schema, GPTQWeightSchema)
    assert default_schema.checkpoint_format == "gptq"

    v1_schema = BaseWeightSchema.from_config(
        {
            "quant_method": "gptq",
            "bits": 4,
            "group_size": 128,
            "sym": False,
            "checkpoint_format": "gptq",
        }
    )
    assert isinstance(v1_schema, GPTQWeightSchema)
    assert v1_schema.checkpoint_format == "gptq"

    v2_schema = BaseWeightSchema.from_config(
        {
            "quant_method": "gptq",
            "bits": 4,
            "group_size": 128,
            "sym": False,
            "checkpoint_format": "GPTQ_V2",
        }
    )
    assert isinstance(v2_schema, GPTQWeightSchema)
    assert v2_schema.checkpoint_format == "gptq_v2"

    v2_alias_schema = BaseWeightSchema.from_config(
        {
            "quant_method": "gptq",
            "bits": 4,
            "group_size": 128,
            "sym": False,
            "format": "gptq_v2",
        }
    )
    assert isinstance(v2_alias_schema, GPTQWeightSchema)
    assert v2_alias_schema.checkpoint_format == "gptq_v2"

    empty_format_schema = BaseWeightSchema.from_config(
        {
            "quant_method": "gptq",
            "bits": 4,
            "group_size": 128,
            "sym": False,
            "checkpoint_format": "",
        }
    )
    assert isinstance(empty_format_schema, GPTQWeightSchema)
    assert empty_format_schema.checkpoint_format == "gptq"

    with pytest.raises(ValueError, match="unsupported GPTQ checkpoint_format"):
        BaseWeightSchema.from_config(
            {
                "quant_method": "gptq",
                "bits": 4,
                "group_size": 128,
                "sym": False,
                "checkpoint_format": "marlin",
            }
        )


def test_autoround_delegate_checkpoint_format() -> None:
    auto_gptq = BaseWeightSchema.from_config(
        {
            "quant_method": "auto-round",
            "bits": 4,
            "group_size": 128,
            "sym": False,
            "packing_format": "auto_round:auto_gptq",
        }
    )
    assert isinstance(auto_gptq, AutoRoundWeightSchema)
    assert isinstance(auto_gptq._delegate, GPTQWeightSchema)
    assert auto_gptq._delegate.checkpoint_format == "gptq"

    for fmt in ("auto_round:gptq_v2", "auto_round:gptqv2"):
        v2_schema = BaseWeightSchema.from_config(
            {
                "quant_method": "auto-round",
                "bits": 4,
                "group_size": 128,
                "sym": False,
                "packing_format": fmt,
            }
        )
        assert isinstance(v2_schema, AutoRoundWeightSchema)
        assert isinstance(v2_schema._delegate, GPTQWeightSchema)
        assert v2_schema._delegate.checkpoint_format == "gptq_v2"

    explicit_v2 = BaseWeightSchema.from_config(
        {
            "quant_method": "auto-round",
            "bits": 4,
            "group_size": 128,
            "sym": False,
            "packing_format": "auto_round:gptqmodel",
            "checkpoint_format": "gptq_v2",
        }
    )
    assert isinstance(explicit_v2, AutoRoundWeightSchema)
    assert isinstance(explicit_v2._delegate, GPTQWeightSchema)
    assert explicit_v2._delegate.checkpoint_format == "gptq_v2"

    awq_schema = BaseWeightSchema.from_config(
        {
            "quant_method": "auto-round",
            "bits": 4,
            "group_size": 128,
            "sym": False,
            "packing_format": "auto_round:auto_awq",
        }
    )
    assert isinstance(awq_schema, AutoRoundWeightSchema)
    assert isinstance(awq_schema._delegate, AWQWeightSchema)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("bits", [2, 3, 4, 5, 6, 7, 8])
@pytest.mark.parametrize("checkpoint_format", ["gptq", "gptq_v2"])
def test_gptq_convert_humming_asymmetric_zero_point(bits: int, checkpoint_format: str) -> None:
    shape_n = 64
    shape_k = 128
    group_size = 64
    param_dtype = torch.float16

    cpu_tensors, true_zp, ref_weight_nk = _build_gptq_checkpoint_tensors(
        shape_n=shape_n,
        shape_k=shape_k,
        bits=bits,
        group_size=group_size,
        param_dtype=param_dtype,
        checkpoint_format=checkpoint_format,
        sym=False,
    )
    cuda_tensors = {name: tensor.cuda() for name, tensor in cpu_tensors.items()}
    original_qzeros = cuda_tensors["qzeros"].clone()

    schema = GPTQWeightSchema(
        bits=bits,
        group_size=group_size,
        sym=False,
        checkpoint_format=checkpoint_format,
    )
    humming_schema, output_tensors = schema.convert_humming(
        tensors=cuda_tensors,
        shape_n_stacks=[shape_n],
        shape_k_stacks=[shape_k],
        param_dtype=param_dtype,
    )

    assert humming_schema.has_zero_point is True
    assert torch.equal(cuda_tensors["qzeros"], original_qzeros)

    packed_zp = output_tensors["zero_point"].transpose(-1, -2).contiguous()
    actual_zp = ops.unpack_weight(packed_zp, bits, dtype=torch.int32)
    torch.testing.assert_close(actual_zp, true_zp.cuda(), rtol=0, atol=0)

    dequantized = humming_schema.dequant_tensors(output_tensors)
    torch.testing.assert_close(dequantized, ref_weight_nk.cuda(), rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("checkpoint_format", ["gptq", "gptq_v2"])
def test_gptq_convert_humming_moe_and_cpu_and_unaligned_n(checkpoint_format: str) -> None:
    bits = 4
    group_size = 64
    param_dtype = torch.bfloat16

    # 3D MoE tensors on CUDA.
    moe_tensors, true_zp_moe, ref_moe = _build_gptq_checkpoint_tensors(
        shape_n=64,
        shape_k=128,
        bits=bits,
        group_size=group_size,
        param_dtype=param_dtype,
        checkpoint_format=checkpoint_format,
        sym=False,
        num_experts=3,
    )
    schema = GPTQWeightSchema(
        bits=bits,
        group_size=group_size,
        sym=False,
        checkpoint_format=checkpoint_format,
    )
    humming_schema, moe_out = schema.convert_humming(
        tensors={k: v.cuda() for k, v in moe_tensors.items()},
        shape_n_stacks=[64],
        shape_k_stacks=[128],
        param_dtype=param_dtype,
        num_experts=3,
    )
    actual_zp_moe = ops.unpack_weight(moe_out["zero_point"].transpose(-1, -2).contiguous(), bits)
    torch.testing.assert_close(actual_zp_moe, true_zp_moe.cuda(), rtol=0, atol=0)
    torch.testing.assert_close(humming_schema.dequant_tensors(moe_out), ref_moe.cuda(), rtol=1e-5, atol=1e-5)

    # Non-multiple-of-32 N (shape_n=16 -> 2 int32 words per group, not divisible by bits=4)
    # and CPU tensor preservation.
    small_cpu_tensors, true_zp_small, _ = _build_gptq_checkpoint_tensors(
        shape_n=16,
        shape_k=128,
        bits=bits,
        group_size=group_size,
        param_dtype=param_dtype,
        checkpoint_format=checkpoint_format,
        sym=False,
    )
    _, small_out = schema.convert_humming(
        tensors=small_cpu_tensors,
        shape_n_stacks=[16],
        shape_k_stacks=[128],
        param_dtype=param_dtype,
    )
    assert small_out["zero_point"].device.type == "cpu"
    expected_packed_zp = _pack_codes_cpu(true_zp_small, bits).transpose(-1, -2).contiguous()
    torch.testing.assert_close(small_out["zero_point"], expected_packed_zp, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("checkpoint_format", ["gptq", "gptq_v2"])
def test_gptq_symmetric_ignores_qzeros(checkpoint_format: str) -> None:
    shape_n = 64
    shape_k = 128
    group_size = 64
    bits = 4
    param_dtype = torch.float16

    cpu_tensors, _, ref_weight_nk = _build_gptq_checkpoint_tensors(
        shape_n=shape_n,
        shape_k=shape_k,
        bits=bits,
        group_size=group_size,
        param_dtype=param_dtype,
        checkpoint_format=checkpoint_format,
        sym=True,
    )
    schema = GPTQWeightSchema(
        bits=bits,
        group_size=group_size,
        sym=True,
        checkpoint_format=checkpoint_format,
    )
    humming_schema, output_tensors = schema.convert_humming(
        tensors={k: v.cuda() for k, v in cpu_tensors.items()},
        shape_n_stacks=[shape_n],
        shape_k_stacks=[shape_k],
        param_dtype=param_dtype,
        convert_symmetric_integer=False,
    )
    assert humming_schema.has_zero_point is False
    assert "zero_point" not in output_tensors
    torch.testing.assert_close(
        humming_schema.dequant_tensors(output_tensors),
        ref_weight_nk.cuda(),
        rtol=1e-5,
        atol=1e-5,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    ("quant_config", "checkpoint_format", "bits", "param_dtype"),
    [
        (
            {"quant_method": "gptq", "bits": 4, "group_size": 64, "desc_act": False, "sym": False},
            "gptq",
            4,
            torch.float16,
        ),
        (
            {
                "quant_method": "gptq",
                "bits": 4,
                "group_size": 64,
                "desc_act": False,
                "sym": False,
                "checkpoint_format": "gptq",
            },
            "gptq",
            4,
            torch.bfloat16,
        ),
        (
            {
                "quant_method": "gptq",
                "bits": 3,
                "group_size": 64,
                "desc_act": False,
                "sym": False,
                "checkpoint_format": "gptq",
            },
            "gptq",
            3,
            torch.float16,
        ),
        (
            {
                "quant_method": "gptq",
                "bits": 8,
                "group_size": 64,
                "desc_act": False,
                "sym": False,
                "checkpoint_format": "gptq",
            },
            "gptq",
            8,
            torch.float16,
        ),
        (
            {
                "quant_method": "gptq",
                "bits": 4,
                "group_size": 64,
                "desc_act": False,
                "sym": False,
                "checkpoint_format": "gptq_v2",
            },
            "gptq_v2",
            4,
            torch.bfloat16,
        ),
        (
            {
                "quant_method": "gptq",
                "bits": 4,
                "group_size": 64,
                "desc_act": False,
                "sym": False,
                "format": "gptq_v2",
            },
            "gptq_v2",
            4,
            torch.float16,
        ),
        (
            {
                "quant_method": "auto-round",
                "bits": 4,
                "group_size": 64,
                "sym": False,
                "packing_format": "auto_round:auto_gptq",
            },
            "gptq",
            4,
            torch.bfloat16,
        ),
        (
            {
                "quant_method": "auto-round",
                "bits": 4,
                "group_size": 64,
                "sym": False,
                "packing_format": "auto_round:gptq_v2",
            },
            "gptq_v2",
            4,
            torch.bfloat16,
        ),
        (
            {
                "quant_method": "gptq",
                "bits": 4,
                "group_size": 64,
                "desc_act": False,
                "sym": True,
                "checkpoint_format": "gptq",
            },
            "gptq",
            4,
            torch.bfloat16,
        ),
    ],
)
def test_humming_layer_gptq_end_to_end_forward(
    tmp_path,
    quant_config: dict[str, object],
    checkpoint_format: str,
    bits: int,
    param_dtype: torch.dtype,
) -> None:
    shape_n = 128
    shape_k = 256
    group_size = 64
    is_sym = bool(quant_config["sym"])

    cpu_tensors, _, ref_weight_nk = _build_gptq_checkpoint_tensors(
        shape_n=shape_n,
        shape_k=shape_k,
        bits=bits,
        group_size=group_size,
        param_dtype=param_dtype,
        checkpoint_format=checkpoint_format,
        sym=is_sym,
    )
    prefixed = {f"proj.{name}": tensor for name, tensor in cpu_tensors.items()}
    dtype_str = "float16" if param_dtype == torch.float16 else "bfloat16"
    (tmp_path / "config.json").write_text(
        json.dumps({"torch_dtype": dtype_str, "quantization_config": quant_config})
    )
    safetensors.torch.save_file(prefixed, str(tmp_path / "model.safetensors"))

    layer = HummingLayer.from_safetensors(str(tmp_path), prefix="proj", torch_dtype=param_dtype)
    layer = layer.cuda()
    layer.transform()

    identity = torch.eye(shape_k, dtype=param_dtype, device="cuda")
    kernel_dequant = layer.forward(identity)
    torch.testing.assert_close(kernel_dequant, ref_weight_nk.cuda().t().to(param_dtype), rtol=1e-2, atol=1e-2)

    generator = torch.Generator(device="cuda").manual_seed(1234)
    # Use positive-biased activations so a +1*scale weight shift produces a large, unmistakable bias.
    inputs = torch.rand((16, shape_k), dtype=param_dtype, device="cuda", generator=generator) + 0.5
    actual = layer.forward(inputs)
    expected = torch.matmul(inputs.float(), ref_weight_nk.cuda().t()).to(param_dtype)

    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=5e-2)

    if not is_sym and checkpoint_format == "gptq":
        scales_nk = cpu_tensors["scales"].float().repeat_interleave(group_size, dim=0).t().cuda()
        uncorrected = torch.matmul(inputs.float(), (ref_weight_nk.cuda() + scales_nk).t()).to(param_dtype)
        max_diff_uncorrected = (actual.float() - uncorrected.float()).abs().max().item()
        assert max_diff_uncorrected > 1.0
