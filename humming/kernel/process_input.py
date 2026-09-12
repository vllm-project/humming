import dataclasses
from typing import ClassVar

import jinja2
import torch

from humming import dtypes
from humming.config import (
    InputQuantizationMode,
    ProcessInputProblemConfig,
    ProcessInputTuningConfig,
)
from humming.config import (
    ProcessInputQuantizationPhase as QuantizationPhase,
)
from humming.jit.runtime import KernelRuntime

_SOURCE_TYPE_CPP = {
    dtypes.float16: "__half",
    dtypes.bfloat16: "__nv_bfloat16",
    dtypes.float32: "float",
}

_SCALE_TYPE_CPP = {
    "float32": "Float32",
    "float8e4m3": "Float8E4M3",
    "float8e8m0": "Float8E8M0",
    "m3bfloat16": "M3BFloat16",
}


CODE_TEMPLATE = jinja2.Template("""
#include <humming/kernel/process_input.cuh>

struct ProcessInputActivation {
  static constexpr ActivationType kType = ActivationType::{{ activation_type }};
{% if activation_type == "Unary" %}
  CUDA_INLINE static float apply(float a) {
    return {{ activation_impl }};
  }
{% elif activation_type != "None" %}
  CUDA_INLINE static float apply(float a, float b) {
    return {{ activation_impl }};
  }
{% endif %}
};

class Config {
public:
  using SourceType = {{ source_dtype }};
  using TargetType = {{ target_dtype }};
  using GroupScaleType = {{ group_scale_dtype }};
  using Activation = ProcessInputActivation;
{{ process_input_config }}
};

class TuningConfig {
public:
{{ process_input_tuning }}
};

using Context = ProcessInputContext<
    Config, TuningConfig, ProcessInputQuantizationPhase::{{ quantization_phase }}>;

{{ process_input_extern }}
{{ tuning_extern }}
extern "C" __constant__ uint32_t NUM_THREADS = Context::kThreads;
extern "C" __constant__ uint32_t INPUT_ROW_SIZE = Context::kInputRowSize;
extern "C" __constant__ uint32_t COLUMNS_PER_TASK = Context::kColumnsPerTask;
extern "C" __constant__ uint32_t OUTPUT_PACKING = Context::kOutputPacking;
extern "C" __constant__ uint32_t SOURCE_DTYPE_ID = {{ source_dtype_config }}::kId;
extern "C" __constant__ uint32_t TARGET_DTYPE_ID = Config::TargetType::kId;
extern "C" __constant__ uint32_t GROUP_SCALE_DTYPE_ID = {{ group_scale_data_type }}::kId;
extern "C" __constant__ uint32_t LAYOUT = static_cast<uint32_t>(Config::kLayout);
extern "C" __constant__ uint32_t QUANT_MODE = static_cast<uint32_t>(Context::kQuantization);
extern "C" __constant__ uint32_t QUANTIZATION_PHASE = static_cast<uint32_t>(Context::kPhase);
""")


@dataclasses.dataclass(kw_only=True)
class ProcessInputKernel(KernelRuntime, ProcessInputProblemConfig, ProcessInputTuningConfig):
    name: ClassVar[str] = "process_input_kernel"
    _str2kernel_cache: ClassVar[dict[tuple[object, ...], torch.Tensor]] = {}
    quantization_phase: QuantizationPhase = QuantizationPhase.Fused

    def __post_init__(self):
        self.quantization_phase = QuantizationPhase(self.quantization_phase)
        ProcessInputProblemConfig.__post_init__(self)
        ProcessInputTuningConfig.__post_init__(self)
        KernelRuntime.__post_init__(self)

    def register_kernel(self):
        from humming import ops

        kernel_id, kernel_name = ops.register_process_input_kernel(self.kernel_filename)
        assert self.name in kernel_name
        self.kernel_id = kernel_id
        self.kernel_name = kernel_name

    def init_kernel(self):
        is_finalizer = isinstance(self, ProcessInputScaleKernel)
        if is_finalizer:
            assert self.quant_mode == InputQuantizationMode.DynamicGroupToken
            assert self.use_tile_partition
            assert self.quantization_phase == QuantizationPhase.Fused
            assert 1 <= self.finalize_tokens_per_block <= 32
            finalize_tokens = self.finalize_tokens_per_block
            kernel_expr = f"finalize_group_token_scales_kernel<Config, TuningConfig, {finalize_tokens}>"
        else:
            kernel_expr = "process_input_kernel<Config, TuningConfig, Context::kPhase>"

        group_scale_data_type = self.group_scale_dtype
        template_args = self.to_template_args()
        template_args.update(
            process_input_config=self.to_cpp_str(ProcessInputProblemConfig),
            process_input_extern=self.to_extern_cpp_str(ProcessInputProblemConfig),
            process_input_tuning=self.to_cpp_str(ProcessInputTuningConfig),
            tuning_extern=self.to_extern_cpp_str(ProcessInputTuningConfig),
            source_dtype=_SOURCE_TYPE_CPP[self.input_dtype],
            target_dtype=(self.quant_dtype or dtypes.float32).to_cpp_str(),
            source_dtype_config=self.input_dtype.to_cpp_str(),
            group_scale_dtype=_SCALE_TYPE_CPP[str(self.group_scale_dtype)],
            group_scale_data_type=group_scale_data_type.to_cpp_str(),
            activation_type=self.activation_type.cpp_name,
        )
        self.code = CODE_TEMPLATE.render(**template_args)
        self.kernel_expr = kernel_expr
        self.prepare()
        self.register_kernel()

    def postprocess_cubin(self, cubin_path: str):
        from humming.utils.cubin import patch_cubin

        mode = None
        if self.quant_dtype == dtypes.float8e3m4:
            mode = "cvt_e3m4"
        elif self.quant_dtype == dtypes.float4e0m3:
            mode = "cvt_e0m3"
        if mode:
            patch_cubin(cubin_path=cubin_path, mode=mode)

    @classmethod
    def prepare_kernels(cls, config, tuning_intervals, device, cache_key=None):
        if cache_key is not None and cache_key in cls._str2kernel_cache:
            return cls._str2kernel_cache[cache_key]

        config_dict = config.to_dict()
        quant_mode = config.quant_mode
        kernel_specs_by_tuning = {}
        for _, _, tuning_config in tuning_intervals:
            if tuning_config in kernel_specs_by_tuning:
                continue
            kernel_config = config_dict | tuning_config.to_dict()
            primary_spec = (cls, kernel_config | {"quantization_phase": QuantizationPhase.Fused})
            secondary_spec = None
            if quant_mode.dynamic_scale_mode == "token" and tuning_config.two_stage:
                primary_spec = (cls, kernel_config | {"quantization_phase": QuantizationPhase.CollectAbsmax})
                secondary_spec = (cls, kernel_config | {"quantization_phase": QuantizationPhase.Quantize})
            elif quant_mode.dynamic_scale_mode == "group_token" and tuning_config.use_tile_partition:
                finalizer_config = kernel_config | {"quantization_phase": QuantizationPhase.Fused}
                finalizer_config["finalize_tokens_per_block"] = tuning_config.finalize_tokens_per_block
                secondary_spec = (ProcessInputScaleKernel, finalizer_config)
            kernel_specs_by_tuning[tuning_config] = primary_spec, secondary_spec

        def get_kernel_spec_key(kernel_spec):
            kernel_type, kernel_config = kernel_spec
            return kernel_type, tuple(sorted(kernel_config.items()))

        unique_kernel_specs = {}
        for primary_spec, secondary_spec in kernel_specs_by_tuning.values():
            unique_kernel_specs.setdefault(get_kernel_spec_key(primary_spec), primary_spec)
            if secondary_spec is not None:
                unique_kernel_specs.setdefault(get_kernel_spec_key(secondary_spec), secondary_spec)

        kernel_specs = list(unique_kernel_specs.values())
        compiled_kernels = cls.compile_many(kernel_specs, device)
        kernels_by_spec = dict(zip(unique_kernel_specs, compiled_kernels, strict=True))
        kernel_ids_by_tuning = {}
        for tuning_config, (primary_spec, secondary_spec) in kernel_specs_by_tuning.items():
            primary_id = kernels_by_spec[get_kernel_spec_key(primary_spec)].kernel_id
            secondary_id = -1
            if secondary_spec is not None:
                secondary_id = kernels_by_spec[get_kernel_spec_key(secondary_spec)].kernel_id
            kernel_ids_by_tuning[tuning_config] = primary_id, secondary_id

        launch_configs = []
        for min_shape_m, max_shape_m, tuning_config in tuning_intervals:
            primary_id, secondary_id = kernel_ids_by_tuning[tuning_config]
            launch_configs.extend((min_shape_m, max_shape_m, primary_id, secondary_id))
        result = torch.tensor(launch_configs, dtype=torch.int64, device="cpu")
        if cache_key is not None:
            cls._str2kernel_cache[cache_key] = result
        return result


@dataclasses.dataclass(kw_only=True)
class ProcessInputScaleKernel(ProcessInputKernel):
    name: ClassVar[str] = "finalize_group_token_scales_kernel"
    finalize_tokens_per_block: int = 4

    def postprocess_cubin(self, cubin_path: str):
        pass
