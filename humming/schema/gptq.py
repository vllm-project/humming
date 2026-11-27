import dataclasses
from typing import Any, ClassVar

import torch

from humming import dtypes, ops
from humming.schema.base import BaseWeightSchema
from humming.schema.humming import HummingWeightSchema


@dataclasses.dataclass(kw_only=True)
class GPTQWeightSchema(BaseWeightSchema):
    quant_method: str = "gptq"
    bits: int
    group_size: int
    desc_act: bool = False
    sym: bool = True
    checkpoint_format: str = "gptq"

    KWARGS_ALIAS: ClassVar[dict[str, list[str]]] = {"checkpoint_format": ["format"]}

    def __post_init__(self):
        checkpoint_format = (self.checkpoint_format or "gptq").lower()
        if checkpoint_format not in ("gptq", "gptq_v2"):
            raise ValueError(f"unsupported GPTQ checkpoint_format: {self.checkpoint_format!r}")
        self.checkpoint_format = checkpoint_format

    def get_tensors_attrs(
        self,
        shape_n: int,
        shape_k: int,
        param_dtype: torch.dtype,
        num_experts: int | None = None,
        has_bias: bool = False,
        stack_size: int = 1,
    ) -> dict[str, dict[str, Any]]:
        num_bits = self.bits
        group_size = self.group_size
        tensor_meta = {
            "qweight": {
                "shape": (shape_k * num_bits // 32, shape_n),
                "dtype": torch.int32,
                "extra_attrs": {
                    "output_dim": 1,
                    "input_dim": 0,
                    "packed_dim": 0,
                    "packed_factor": 32 / num_bits,
                },
            },
            "scales": {
                "shape": (shape_k // group_size, shape_n),
                "dtype": param_dtype,
                "extra_attrs": {"output_dim": 1, "input_dim": 0, "scale_type": "group"},
            },
            "qzeros": {
                "shape": (shape_k // group_size, shape_n * num_bits // 32),
                "dtype": torch.int32,
                "extra_attrs": {
                    "output_dim": 1,
                    "input_dim": 0,
                    "packed_dim": 1,
                    "packed_factor": 32 / num_bits,
                    "scale_type": "group",
                },
            },
            "g_idx": {
                "shape": (shape_k,),
                "dtype": torch.int32,
                "extra_attrs": {"input_dim": 0},
            },
        }

        if has_bias:
            tensor_meta["bias"] = {
                "shape": (shape_n,),
                "dtype": param_dtype,
                "extra_attrs": {"output_dim": 0},
            }

        self.may_add_expert_dim(tensor_meta, num_experts)
        return tensor_meta

    def infer_shape(self, tensors: dict[str, torch.Tensor]) -> tuple[int, int, int | None, bool]:
        shape_n = tensors["qweight"].size(-1)
        shape_k = tensors["qweight"].size(-2) * 32 // self.bits
        has_bias = "bias" in tensors
        return shape_n, shape_k, None, has_bias

    def to_humming_schema(self, param_dtype: torch.dtype) -> HummingWeightSchema:
        assert not self.desc_act
        return HummingWeightSchema(
            b_dtype=dtypes.DataType.from_str(f"uint{self.bits}"),
            bs_dtype=dtypes.DataType.from_torch_dtype(param_dtype),
            weight_scale_group_size=self.group_size,
            has_zero_point=not self.sym,
        )

    def _add_zero_point_offset(self, qzeros: torch.Tensor) -> torch.Tensor:
        orig_device = qzeros.device
        orig_last_dim = qzeros.size(-1)
        pad_words = -orig_last_dim % self.bits

        packed = qzeros.contiguous()
        if not packed.is_cuda:
            packed = packed.cuda()
        if pad_words > 0:
            packed = torch.nn.functional.pad(packed, (0, pad_words))

        unpack_dtype = torch.uint8 if self.bits <= 8 else torch.int32
        unpacked = ops.unpack_weight(packed, self.bits, dtype=unpack_dtype)
        # GPTQ v1 checkpoints store packed (zp - 1); add 1 modulo 2^bits to recover true zp.
        max_code = (1 << self.bits) - 1
        adjusted = (unpacked + 1) & max_code

        repacked = ops.pack_weight(adjusted, self.bits)
        if pad_words > 0:
            repacked = repacked[..., :orig_last_dim]
        return repacked.to(orig_device)

    def _convert_humming(
        self,
        tensors: dict[str, torch.Tensor],
        shape_n_stacks: list[int],
        shape_k_stacks: list[int],
        param_dtype: torch.dtype,
        num_experts: int | None = None,
        allow_inplace: bool = False,
    ) -> tuple[HummingWeightSchema, dict[str, torch.Tensor]]:
        schema = self.to_humming_schema(param_dtype)

        weight = tensors["qweight"]
        weight = weight.transpose(-1, -2).contiguous()
        weight_scale = tensors["scales"].to(param_dtype)
        weight_scale = weight_scale.transpose(-1, -2).contiguous()

        output_tensors = {"weight": weight, "weight_scale": weight_scale}

        if not self.sym:
            zero_point = tensors["qzeros"]
            if self.checkpoint_format == "gptq":
                zero_point = self._add_zero_point_offset(zero_point)
            zero_point = zero_point.transpose(-1, -2).contiguous()
            output_tensors["zero_point"] = zero_point

        if "bias" in tensors:
            output_tensors["bias"] = tensors["bias"]

        return schema, output_tensors
