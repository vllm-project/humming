"""Random weight-only quantized problems + torch references for the ROCm MFMA tests."""

import dataclasses

import torch

from humming.rocm import layout


@dataclasses.dataclass
class QuantizedLinear:
    num_bits: int
    group_size: int
    codes: torch.Tensor  # [N, K] int32
    packed_weight: torch.Tensor  # Humming standard packed [N, K * bits / 32]
    weight_scale: torch.Tensor  # [N, num_groups], activation dtype
    zero_point: torch.Tensor | None  # [N, num_groups] int32 or None
    mfma_weight: torch.Tensor  # ROCm MFMA layout, [N, K * bits / 32]
    mfma_scale: torch.Tensor  # [num_groups, N]
    mfma_zero_point: torch.Tensor | None  # [num_groups, N] uint8

    def dequantized(self) -> torch.Tensor:
        return layout.dequantize_reference(
            self.codes, self.weight_scale, self.zero_point, self.num_bits, self.group_size
        )


def make_quantized_linear(
    shape_n: int,
    shape_k: int,
    num_bits: int,
    group_size: int,
    has_zero_point: bool,
    dtype: torch.dtype,
    device: str = "cuda",
    seed: int = 0,
) -> QuantizedLinear:
    generator = torch.Generator(device=device).manual_seed(seed)
    codes = torch.randint(
        0, 1 << num_bits, (shape_n, shape_k), dtype=torch.int32, device=device, generator=generator
    )
    num_groups = shape_k // group_size if group_size else 1
    # Scales like max-abs quantization of N(0, 1) weights: |w| <~ 4 -> 4 / 2^(bits-1).
    base_scale = 4.0 / (1 << (num_bits - 1))
    scale_noise = torch.rand(shape_n, num_groups, device=device, generator=generator)
    weight_scale = (base_scale * (0.5 + scale_noise)).to(dtype)
    zero_point = None
    if has_zero_point:
        zero_point = torch.randint(
            0, 1 << num_bits, (shape_n, num_groups), dtype=torch.int32, device=device, generator=generator
        )
    packed_weight = layout.pack_weight(codes, num_bits)
    return QuantizedLinear(
        num_bits=num_bits,
        group_size=group_size,
        codes=codes,
        packed_weight=packed_weight,
        weight_scale=weight_scale,
        zero_point=zero_point,
        mfma_weight=layout.repack_weight_for_mfma(packed_weight, num_bits),
        mfma_scale=layout.transform_weight_scale(weight_scale, dtype),
        mfma_zero_point=layout.transform_zero_point(zero_point) if zero_point is not None else None,
    )


def error_stats(outputs: torch.Tensor, reference: torch.Tensor) -> tuple[float, float]:
    """(max abs error, max abs error normalized by max |reference|)."""
    difference = (outputs.float() - reference.float()).abs()
    max_abs = difference.max().item()
    return max_abs, max_abs / max(reference.abs().max().item(), 1e-6)


# Normalized-error budget: output rounding (2^-11 / 2^-8) plus FP32 summation order.
TOLERANCE = {torch.float16: 2e-3, torch.bfloat16: 1e-2}
