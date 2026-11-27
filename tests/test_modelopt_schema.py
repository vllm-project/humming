import json
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import safetensors.torch
import torch

from humming import dtypes
from humming.config import LayerConfig
from humming.config.enum import InputQuantizationMode, WeightScale2Type, WeightScaleType
from humming.layer import HummingLayer
from humming.schema import BaseInputSchema, BaseWeightSchema, is_humming_schema_compatible
from humming.schema.modelopt import (
    ModeloptInputSchema,
    ModeloptMxfp8InputSchema,
    ModeloptMxfp8WeightSchema,
    ModeloptNvfp4InputSchema,
    ModeloptNvfp4WeightSchema,
    ModeloptWeightSchema,
)


def _mock_cpu_cuda_if_needed(monkeypatch: pytest.MonkeyPatch) -> None:
    if not torch.cuda.is_available():
        monkeypatch.setattr(torch.cuda, "device", lambda _device: nullcontext())
        monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")


def test_mxfp8_weight_infer_shape_and_convert(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_cpu_cuda_if_needed(monkeypatch)
    schema = ModeloptMxfp8WeightSchema()

    weight_2d = torch.zeros((64, 128), dtype=torch.float8_e4m3fn)
    weight_3d = torch.zeros((4, 64, 128), dtype=torch.float8_e4m3fn)
    bias = torch.ones((64,), dtype=torch.bfloat16)

    assert schema.infer_shape({"weight": weight_2d}) == (64, 128, None, False)
    assert schema.infer_shape({"weight": weight_3d, "bias": bias}) == (64, 128, None, True)

    weight_scale = torch.full((64, 4), 127, dtype=torch.uint8)
    humming_schema, tensors = schema.convert_humming(
        tensors={"weight": weight_2d, "weight_scale": weight_scale, "bias": bias},
        shape_n_stacks=[64],
        shape_k_stacks=[128],
        param_dtype=torch.bfloat16,
    )

    assert humming_schema.b_dtype == dtypes.float8e4m3
    assert humming_schema.bs_dtype == dtypes.float8e8m0
    assert humming_schema.weight_scale_group_size == 32
    assert humming_schema.weight_scale_type == WeightScaleType.GROUP
    assert humming_schema.weight_scale_2_type == WeightScale2Type.NONE
    assert tensors["weight"].dtype == torch.int32
    assert tensors["weight"].shape == (64, 32)
    assert tensors["weight_scale"].dtype == torch.float8_e8m0fnu
    assert tensors["weight_scale"].shape == (64, 4)
    assert torch.equal(tensors["bias"], bias)


def test_modelopt_from_config_dispatch_and_subclass_fields() -> None:
    nvfp4_config = {
        "quant_method": "modelopt",
        "quant_algo": "NVFP4",
        "dynamic": True,
        "num_bits": 4,
        "type": "float",
        "group_size": 16,
    }
    nvfp4_input = BaseInputSchema.from_config(nvfp4_config)
    assert isinstance(nvfp4_input, ModeloptNvfp4InputSchema)
    assert nvfp4_input.dynamic is True
    assert nvfp4_input.num_bits == 4
    assert nvfp4_input.type == "float"
    assert nvfp4_input.group_size == 16
    assert nvfp4_input.get_tensors_attrs(shape_k=128, param_dtype=torch.bfloat16) == {}
    assert nvfp4_input.to_humming_schema(torch.bfloat16).input_quant_mode == (
        InputQuantizationMode.DynamicGroupToken
    )

    mxfp8_config = {"quant_method": "modelopt", "quant_algo": "MXFP8"}
    mxfp8_weight = BaseWeightSchema.from_config(mxfp8_config)
    mxfp8_input = BaseInputSchema.from_config(mxfp8_config)
    assert isinstance(mxfp8_weight, ModeloptMxfp8WeightSchema)
    assert isinstance(mxfp8_input, ModeloptMxfp8InputSchema)
    assert mxfp8_input.dynamic is True
    assert mxfp8_input.group_size == 32

    direct_nvfp4_input = ModeloptNvfp4InputSchema.from_config({"dynamic": True})
    direct_mxfp8_input = ModeloptMxfp8InputSchema.from_config({})
    direct_nvfp4_weight = ModeloptNvfp4WeightSchema.from_config({})
    direct_mxfp8_weight = ModeloptMxfp8WeightSchema.from_config({})
    assert isinstance(direct_nvfp4_input, ModeloptNvfp4InputSchema)
    assert direct_nvfp4_input.quant_algo == "nvfp4"
    assert direct_nvfp4_input.dynamic is True
    assert isinstance(direct_mxfp8_input, ModeloptMxfp8InputSchema)
    assert direct_mxfp8_input.quant_algo == "mxfp8"
    assert isinstance(direct_nvfp4_weight, ModeloptNvfp4WeightSchema)
    assert direct_nvfp4_weight.quant_algo == "nvfp4"
    assert isinstance(direct_mxfp8_weight, ModeloptMxfp8WeightSchema)
    assert direct_mxfp8_weight.quant_algo == "mxfp8"

    with pytest.raises(ValueError, match="unsupported modelopt algo"):
        ModeloptWeightSchema.from_config({"quant_algo": "fp8"})
    with pytest.raises(ValueError, match="unsupported modelopt algo"):
        ModeloptInputSchema.from_config({"quant_algo": "fp8"})


def test_mxfp8_input_schema_humming_conversion_and_compatibility(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_cpu_cuda_if_needed(monkeypatch)
    input_schema = ModeloptMxfp8InputSchema()
    weight_humming = ModeloptMxfp8WeightSchema().to_humming_schema(torch.bfloat16)

    assert input_schema.get_activation_bits() == 8
    assert input_schema.get_tensors_attrs(shape_k=128, param_dtype=torch.bfloat16) == {}

    input_humming, output_tensors = input_schema.convert_humming(
        tensors={},
        shape_n_stacks=[64],
        shape_k_stacks=[128],
        param_dtype=torch.bfloat16,
    )
    assert output_tensors == {}
    assert input_humming.a_dtype == dtypes.float8e4m3
    assert input_humming.input_scale_group_size == 32
    assert input_humming.input_scale_dtype is None
    assert input_humming.input_quant_mode == InputQuantizationMode.DynamicGroup

    # Fallback config_groups entries without input_activations copy weight dynamic=False;
    # MXFP8 activations stay dynamic with no static input_scale tensor.
    fallback_input = ModeloptMxfp8InputSchema(dynamic=False)
    assert fallback_input.get_tensors_attrs(shape_k=128, param_dtype=torch.bfloat16) == {}
    assert fallback_input.to_humming_schema(torch.bfloat16) == input_humming

    for sm_version, expected in ((80, False), (90, True), (100, True), (120, True)):
        monkeypatch.setattr("humming.schema.humming.current_device", SimpleNamespace(sm_version=sm_version))
        assert is_humming_schema_compatible(input_humming, weight_humming, torch.bfloat16) is expected

    # Unspecified input_scale_dtype resolves to float32 on SM90 WGMMA and float8e8m0 on block-scaled SMs.
    assert isinstance(weight_humming.weight_scale_type, WeightScaleType)
    for sm_version, expected_as_dtype in (
        (90, dtypes.float32),
        (100, dtypes.float8e8m0),
        (120, dtypes.float8e8m0),
    ):
        monkeypatch.setattr(
            "humming.config.config.current_device",
            SimpleNamespace(sm_version=sm_version, is_ppu=False),
        )
        layer_config = LayerConfig(
            sm_version=sm_version,
            shape_n=64,
            shape_k=128,
            a_dtype=input_humming.a_dtype,
            as_dtype=input_humming.input_scale_dtype,
            input_quant_mode=input_humming.input_quant_mode,
            input_scale_group_size=input_humming.input_scale_group_size,
            b_dtype=weight_humming.b_dtype,
            bs_dtype=weight_humming.bs_dtype,
            c_dtype=dtypes.bfloat16,
            weight_scale_group_size=weight_humming.weight_scale_group_size,
            weight_scale_type=weight_humming.weight_scale_type,
        )
        assert layer_config.as_dtype == expected_as_dtype


@pytest.mark.parametrize(
    "case",
    ["mxfp8_flat", "mxfp8_config_groups", "nvfp4_dynamic_config_groups"],
)
def test_from_safetensors_modelopt_checkpoints(case: str, tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_cpu_cuda_if_needed(monkeypatch)

    if case == "mxfp8_flat":
        quant_config: dict[str, object] = {
            "quant_method": "modelopt",
            "quant_algo": "MXFP8",
            "group_size": 32,
        }
    elif case == "mxfp8_config_groups":
        quant_config = {
            "quant_method": "modelopt",
            "quant_algo": "MXFP8",
            "config_groups": {
                "group_0": {
                    "targets": ["Linear"],
                    "weights": {"dynamic": False, "num_bits": 8, "type": "float", "group_size": 32},
                    "input_activations": {"dynamic": True, "num_bits": 8, "type": "float", "group_size": 32},
                }
            },
        }
    else:
        quant_config = {
            "quant_method": "modelopt",
            "quant_algo": "NVFP4",
            "config_groups": {
                "group_0": {
                    "targets": ["Linear"],
                    "weights": {"dynamic": False, "num_bits": 4, "type": "float", "group_size": 16},
                    "input_activations": {"dynamic": True, "num_bits": 4, "type": "float", "group_size": 16},
                }
            },
        }

    if case.startswith("mxfp8"):
        tensors = {
            "proj.weight": torch.ones((64, 128), dtype=torch.bfloat16).to(torch.float8_e4m3fn),
            "proj.weight_scale": torch.full((64, 4), 127, dtype=torch.uint8),
        }
    else:
        tensors = {
            "proj.weight": torch.full((64, 64), 3, dtype=torch.uint8),
            "proj.weight_scale": torch.ones((64, 8), dtype=torch.bfloat16).to(torch.float8_e4m3fn),
            "proj.weight_scale_2": torch.full((1,), 0.5, dtype=torch.float32),
        }

    (tmp_path / "config.json").write_text(
        json.dumps({"torch_dtype": "bfloat16", "quantization_config": quant_config})
    )
    safetensors.torch.save_file(tensors, str(tmp_path / "model.safetensors"))

    layer = HummingLayer.from_safetensors(str(tmp_path), prefix="proj", torch_dtype=torch.bfloat16)

    assert layer.shape_n == 64
    assert layer.shape_k == 128
    assert not hasattr(layer, "input_scale")
    assert torch.equal(layer.weight.data.view(torch.uint8), tensors["proj.weight"].view(torch.uint8))
    if case.startswith("mxfp8"):
        assert isinstance(layer.weight_schema, ModeloptMxfp8WeightSchema)
        assert isinstance(layer.input_schema, ModeloptMxfp8InputSchema)
        assert torch.equal(layer.weight_scale.data, tensors["proj.weight_scale"])
    else:
        assert isinstance(layer.weight_schema, ModeloptNvfp4WeightSchema)
        assert isinstance(layer.input_schema, ModeloptNvfp4InputSchema)
        assert layer.input_schema.dynamic is True
        assert torch.equal(layer.weight_scale_2.data, tensors["proj.weight_scale_2"])
