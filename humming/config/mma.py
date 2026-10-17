import math
import re

import humming.dtypes as dtypes
from humming.config.enum import MmaType
from humming.device import current_device
from humming.utils.cubin import is_cubin_patch_enabled

DTYPE_BIT_WIDTH_MAP = {
    "f32": 32,
    "s32": 32,
    "f16": 16,
    "bf16": 16,
    "e4m3": 8,
    "e5m2": 8,
    "e8m0": 8,
    "s8": 8,
    "e3m2": 6,
    "e2m3": 6,
    "e2m1": 4,
    "s4": 4,
}

DTYPE_MAP = {
    dtypes.float32: "f32",
    dtypes.int32: "s32",
    dtypes.float16: "f16",
    dtypes.bfloat16: "bf16",
    dtypes.float8e4m3: "e4m3",
    dtypes.float8e5m2: "e5m2",
    dtypes.float8e8m0: "e8m0",
    dtypes.int8: "s8",
    dtypes.float6e3m2: "e3m2",
    dtypes.float6e2m3: "e2m3",
    dtypes.float4e2m1: "e2m1",
    dtypes.int4: "s4",
    # Hardware supports these dtypes (e3m4/e0m3) but exposes no matching PTX types.
    # MMA/MXMMA substitute PTX types and patch the cubin; UMMA selects descriptor formats.
    dtypes.float8e3m4: "e5m2",
    dtypes.float4e0m3: "e2m1",
}

UNDOCUMENTED_FP_DTYPES = (dtypes.float8e3m4, dtypes.float4e0m3)
# Architectures verified to execute the undocumented E3M4/E0M3 encodings. Add an architecture only after
# testing on hardware: SM107 raises an illegal instruction for them.
UNDOCUMENTED_FP_SM_VERSIONS = (100, 103, 120, 121)


def supports_undocumented_fp_dtypes(sm_version: int) -> bool:
    return is_cubin_patch_enabled() and sm_version in UNDOCUMENTED_FP_SM_VERSIONS


def uses_undocumented_fp_operand(a_dtype: dtypes.DataType, b_dtype: dtypes.DataType | None = None) -> bool:
    # Weights narrower than the activation are dequantized in software instead of fed to the tensor core.
    has_raw_undocumented_weight = b_dtype in UNDOCUMENTED_FP_DTYPES and b_dtype.num_bits == a_dtype.num_bits
    return a_dtype in UNDOCUMENTED_FP_DTYPES or has_raw_undocumented_weight


def get_mma_b_dtype(
    a_dtype: dtypes.DataType,
    b_dtype: dtypes.DataType,
    has_zero_point: bool,
    sm_version: int,
) -> dtypes.DataType:
    fp8_dtypes = (dtypes.float8e4m3, dtypes.float8e5m2, dtypes.float8e3m4)
    fp4_dtypes = (dtypes.float4e2m1, dtypes.float4e0m3)
    if a_dtype in fp8_dtypes and b_dtype in fp8_dtypes:
        return b_dtype
    if a_dtype in fp4_dtypes and b_dtype in fp4_dtypes:
        return b_dtype
    if a_dtype not in fp8_dtypes or b_dtype.num_bits >= a_dtype.num_bits:
        return a_dtype

    candidates = (a_dtype, dtypes.float8e4m3)
    if supports_undocumented_fp_dtypes(sm_version):
        candidates += (dtypes.float8e3m4,)
    for target_dtype in candidates:
        if b_dtype.is_integer_type:
            if b_dtype.num_bits <= target_dtype.mantissa_bits + (1 if has_zero_point else 2):
                return target_dtype
        else:
            required_mantissa = b_dtype.mantissa_bits - (b_dtype.exponent_bits == 0)
            fits_mantissa = required_mantissa <= target_dtype.mantissa_bits
            fits_exponent = b_dtype.exponent_bits <= target_dtype.exponent_bits
            if fits_mantissa and fits_exponent:
                return target_dtype
    # Keep unsupported conversions visible to schema/kernel validation.
    return a_dtype


SF_DTYPE_MAP = {
    "e8m0": "ue8m0",
    "e4m3": "ue4m3",
}


def get_default_mma_type(layer_config):
    if layer_config.sm_version // 10 == 9 and layer_config.a_dtype != dtypes.int4:
        return MmaType.WGMMA
    if layer_config.use_block_scaled_mma and layer_config.sm_version // 10 == 12:
        return MmaType.MXMMA
    if layer_config.is_umma_supported:
        return MmaType.UMMA
    return MmaType.MMA


def get_mxmma_scale_config(layer_config):
    """Return the scale vector size and dtype used by the generated MXMMA instruction."""
    mma_shape_k = 256 // layer_config.a_dtype.num_bits
    group_size = (
        layer_config.weight_scale_group_size
        or layer_config.input_scale_group_size
        or (layer_config.mma_scale_group_size if layer_config.a_dtype.num_bits == 4 else mma_shape_k)
    )
    assert mma_shape_k % group_size == 0
    scale_dtype = (
        layer_config.bs_dtype
        if layer_config.is_group_weight_scale or layer_config.is_block_weight_scale
        else layer_config.as_dtype
        if layer_config.input_scale_group_size > 0
        else dtypes.float8e8m0
    )
    return mma_shape_k // group_size, scale_dtype


def get_mxmma_compiler_error(layer_config, compiler_version):
    scale_vec, scale_dtype = get_mxmma_scale_config(layer_config)
    uses_fp4_4x = layer_config.a_dtype.num_bits == 4 and scale_vec == 4
    if uses_fp4_4x and scale_dtype == dtypes.float8e8m0 and compiler_version < (13, 1):
        return (
            "MXMMA kind::mxf4nvf4 with scale_vec::4X and UE8M0 scales "
            "requires CUDA 13.1 or newer (PTX ISA 9.1)"
        )
    return None


def get_warp_spec_registers(layer_config, tuning_config, registers_per_sm=None):
    if registers_per_sm is None:
        registers_per_sm = current_device.max_registers_per_sm
    num_math_threads = math.prod(tuning_config.block_shape) // math.prod(tuning_config.warp_shape) * 32
    num_threads = num_math_threads + 128
    launch_budget = registers_per_sm // (num_threads * tuning_config.num_ctas_per_sm) // 8 * 8
    producer_registers = tuning_config.producer_registers
    if producer_registers is None:
        needs_more_load_registers = tuning_config.num_ctas_per_sm == 1 and layer_config.a_dtype.num_bits != 16
        needs_more_load_registers |= num_math_threads > 256
        producer_registers = 40 if needs_more_load_registers else 24
    available_registers = launch_budget * num_threads - producer_registers * 128
    consumer_registers = min(248, available_registers // num_math_threads // 8 * 8)
    if producer_registers > launch_budget or consumer_registers < producer_registers:
        raise ValueError("register budget exceeded: producer allocation cannot fit the launch budget")
    return producer_registers, consumer_registers


def get_register_budget_error(layer_config, tuning_config, use_f16_accum=False, registers_per_sm=None):
    mma_type = tuning_config.mma_type or get_default_mma_type(layer_config)
    if registers_per_sm is None:
        registers_per_sm = current_device.max_registers_per_sm
    warp_m, warp_n, warp_k = tuning_config.warp_shape
    num_math_threads = math.prod(tuning_config.block_shape) // math.prod(tuning_config.warp_shape) * 32
    num_threads = num_math_threads + (128 if tuning_config.use_warp_spec else 0)
    if mma_type == MmaType.UMMA:
        if layer_config.use_raw_weight:
            dequant_threads = 0
        else:
            dequant_threads = 128 * tuning_config.umma_num_dequant_warpgroups
        num_threads = 256 + dequant_threads
    launch_budget = registers_per_sm // (num_threads * tuning_config.num_ctas_per_sm) // 8 * 8
    if launch_budget < 56:
        return (
            "register budget exceeded: requires at least 56 registers per thread; "
            f"launch budget {launch_budget:g}"
        )
    if mma_type == MmaType.UMMA:
        return None  # UMMA accumulators use TMEM and have separate resource checks.

    if mma_type == MmaType.WGMMA:
        use_wgmma_ss = layer_config.a_dtype.num_bits == layer_config.b_dtype.num_bits
        # ptxas needs 26 (SS) or 32 (RS) registers beyond one N=64 accumulator tuple.
        # setmaxnreg cannot relax this per-instruction compile limit.
        instruction_accumulators = warp_m // (4 if use_f16_accum else 2)
        instruction_overhead = 26 if use_wgmma_ss else 32
        instruction_registers = instruction_accumulators + instruction_overhead
        if layer_config.use_packed_k_layout and tuning_config.wgmma_split_issue_wait:
            # Split packed-K keeps all RS K slabs live across prefetch. The
            # per-instruction overhead already includes four operand registers.
            packed_operand_registers = warp_k * layer_config.a_dtype.num_bits // 64
            instruction_registers += packed_operand_registers - 4
        if instruction_registers > launch_budget:
            return (
                "register budget exceeded: WGMMA issue requires at least "
                f"{instruction_registers} registers per thread; launch budget {launch_budget:g}"
            )

    math_budget = min(255, launch_budget)
    if tuning_config.use_warp_spec:
        try:
            _, math_budget = get_warp_spec_registers(layer_config, tuning_config, registers_per_sm)
        except ValueError as error:
            return str(error)

    if math_budget < 56:
        return (
            "register budget exceeded: requires at least 56 registers per math thread; "
            f"launch budget {launch_budget:g}, math-thread budget {math_budget:g}"
        )

    if mma_type == MmaType.WGMMA:
        return None

    accumulator_registers = warp_m * warp_n / (64 if use_f16_accum else 32)
    has_group_accumulator = False
    if mma_type != MmaType.MXMMA and layer_config.a_dtype.num_bits < 16:
        has_group_scale = layer_config.input_scale_group_size > 0
        if not has_group_scale and not layer_config.use_fused_e8m0_scale:
            has_group_scale = layer_config.is_group_weight_scale or layer_config.is_block_weight_scale
        has_group_accumulator = has_group_scale
    buffer_registers = (warp_m + warp_n) / 4
    if has_group_accumulator:
        accumulator_registers *= 1.5
    demand = accumulator_registers + buffer_registers
    if demand >= math_budget - 8:
        return (
            f"register budget exceeded: accumulator {accumulator_registers:g} + "
            f"single-buffer {buffer_registers:g} = {demand:g} must be < "
            f"math-thread budget {math_budget:g} - 8 ({math_budget - 8:g})"
        )
    return None


def calc_reg_count(rows, cols, ptx_dtype):
    total_bits = rows * cols * DTYPE_BIT_WIDTH_MAP[ptx_dtype]
    assert total_bits % (32 * 32) == 0
    reg_count = total_bits // (32 * 32)
    return reg_count


class MmaOpClassImpl:
    mma_type = MmaType.MMA

    def __init__(self, m, n, k, a_dtype, b_dtype, cd_dtype):
        self.shape = (m, n, k)
        self.a_dtype = a_dtype if isinstance(a_dtype, str) else DTYPE_MAP[a_dtype]
        self.b_dtype = b_dtype if isinstance(b_dtype, str) else DTYPE_MAP[b_dtype]
        self.cd_dtype = cd_dtype if isinstance(cd_dtype, str) else DTYPE_MAP[cd_dtype]

        f6f4_types = ("e3m2", "e2m3", "e2m1")
        self.use_f8f6f4 = self.a_dtype in ("e4m3", "e5m2") and self.b_dtype in f6f4_types
        self.native_mixed = self.use_f8f6f4

        b_reg_dtype = "e4m3" if self.use_f8f6f4 else self.b_dtype
        self.reg_a_count = calc_reg_count(m, k, self.a_dtype)
        self.reg_b_count = calc_reg_count(k, n, b_reg_dtype)
        self.reg_cd_count = calc_reg_count(m, n, self.cd_dtype)
        if self.cd_dtype == "f16":
            self.val_type_cd = "half"
            self.reg_cd_type = "uint32_t"
        elif self.cd_dtype == "bf16":
            self.val_type_cd = "nv_bfloat16"
            self.reg_cd_type = "uint32_t"
        elif self.cd_dtype == "f32":
            self.val_type_cd = "float"
            self.reg_cd_type = "float"
        elif self.cd_dtype == "s32":
            self.val_type_cd = "int32_t"
            self.reg_cd_type = "uint32_t"
        else:
            raise ValueError(f"Invalid cd_dtype: {cd_dtype}")

    def to_cpp_str(self, include_class_name=False):
        reg_cd_type = self.reg_cd_type
        lines = [
            f"static constexpr MmaType kMmaType = MmaType::{self.mma_type.name};",
            f"using MmaShape = Shape<{self.shape[0]}, {self.shape[1]}, {self.shape[2]}>;",
            "",
            f"using ValTypeC = {self.val_type_cd};",
            f"using ValTypeD = {self.val_type_cd};",
            "",
            f"static constexpr uint32_t kATypeBits = {DTYPE_BIT_WIDTH_MAP[self.a_dtype]};",
            f"static constexpr uint32_t kBTypeBits = {DTYPE_BIT_WIDTH_MAP[self.b_dtype]};",
            f"static constexpr uint32_t kCTypeBits = {DTYPE_BIT_WIDTH_MAP[self.cd_dtype]};",
            f"static constexpr uint32_t kDTypeBits = {DTYPE_BIT_WIDTH_MAP[self.cd_dtype]};",
            f"static constexpr bool kNativeMixed = {'true' if self.native_mixed else 'false'};",
            "",
            f"using ARegisters = uint32_t[{self.reg_a_count}];",
            f"using BRegisters = uint32_t[{self.reg_b_count}];",
            f"using CRegisters = {self.reg_cd_type}[{self.reg_cd_count}];",
            f"using DRegisters = {self.reg_cd_type}[{self.reg_cd_count}];",
            "",
            "CUDA_INLINE",
            f"static void fma(uint32_t *a, uint32_t *b, {reg_cd_type} *c, {reg_cd_type} *d) {{",
            *[f"  d[{i}] = c[{i}];" for i in range(self.reg_cd_count)],
            *self.generate_ptx(indent=2).strip("\n").split("\n"),
            "};",
        ]

        code = "\n".join("  " + x if x else x for x in lines)
        if include_class_name:
            code = f"class MmaOpClass {{\n{code}\n}};"

        return code

    def generate_ptx(self, indent=0):
        uses_half_m_float = self.shape[0] == 8 and self.a_dtype in ("f16", "bf16")
        if uses_half_m_float and not current_device.is_ppu:
            return self.generate_half_m_ptx(indent)

        a_dtype = self.a_dtype
        b_dtype = self.b_dtype
        cd_dtype = self.cd_dtype
        shape = self.shape

        asm_op = "mma.sync.aligned"
        if self.use_f8f6f4:
            asm_op += ".kind::f8f6f4"
        asm_op += f".m{shape[0]}n{shape[1]}k{shape[2]}.row.col"
        if current_device.is_ppu:
            asm_op = "ppu." + asm_op
        asm_op += f".{cd_dtype}.{a_dtype}.{b_dtype}.{cd_dtype}"
        if "s" in a_dtype:
            asm_op += ".satfinite"

        start = 0
        end = 0
        param_placeholders_list = []
        counts = [self.reg_cd_count, self.reg_a_count, self.reg_b_count]
        for i in range(len(counts)):
            end += counts[i]
            placeholder_str = ", ".join(f"%{x}" for x in range(start, end))
            param_placeholders_list.append("{" + placeholder_str + "}")
            start += counts[i]
        # Keep C tied to D; merging equal C inputs can create partial register
        # overlap that is miscompiled when MMA expands to multiple instructions.
        param_placeholders_list.append(param_placeholders_list[0])

        a_params = []
        b_params = []
        d_params = []
        for i in range(self.reg_a_count):
            a_params.append(f' "r"(a[{i}])')
        for i in range(self.reg_b_count):
            b_params.append(f' "r"(b[{i}])')
        for i in range(self.reg_cd_count):
            t = "f" if cd_dtype == "f32" else "r"
            d_params.append(f'"+{t}"(d[{i}])')

        asm_code = f"""
        asm volatile(
          "{asm_op} "
          "{", ".join(param_placeholders_list)};\\n"
          : {", ".join(d_params)}
          : {", ".join(a_params)},
            {", ".join(b_params)}
        );
        """

        space_count = len(re.findall("^\n( +)", asm_code)[0])
        asm_code = asm_code.replace("\n" + " " * space_count, "\n").strip()
        asm_code = "".join("\n" + " " * indent + x for x in asm_code.split("\n"))

        return asm_code

    def generate_half_m_ptx(self, indent=0):
        assert self.shape[1] == 16 and self.shape[2] in (8, 16)
        assert self.cd_dtype in ("f16", "f32")
        constraint = "f" if self.cd_dtype == "f32" else "r"
        output_count = self.reg_cd_count
        output = [f"%{index}" for index in range(output_count)]
        a_start = output_count
        b_start = a_start + self.reg_a_count
        # Compute B.T @ A.T with a native 16-row MMA; transpose C at the epilogue.
        b_order = (0, 2, 1, 3) if self.shape[2] == 16 else (0, 1)
        a_operands = [f"%{b_start + index}" for index in b_order]
        b_operands = [f"%{a_start + index}" for index in range(self.reg_a_count)]
        groups = [output, a_operands, b_operands, output]
        input_types = f"{self.b_dtype}.{self.a_dtype}"
        instruction = f"mma.sync.aligned.m16n8k{self.shape[2]}.row.col"
        instruction += f".{self.cd_dtype}.{input_types}.{self.cd_dtype}"
        operands = ", ".join("{" + ", ".join(group) + "}" for group in groups)
        outputs = ", ".join(f'"+{constraint}"(d[{index}])' for index in range(output_count))
        inputs = [f'"r"(a[{index}])' for index in range(self.reg_a_count)]
        inputs += [f'"r"(b[{index}])' for index in range(self.reg_b_count)]
        lines = [
            "asm volatile(",
            '  "{ "',
            f'  "{instruction} {operands}; }}\\n"',
            f"  : {outputs}",
            f"  : {', '.join(inputs)}",
            ");",
        ]
        return "\n".join(" " * indent + line for line in lines)


class UmmaOpClassImpl(MmaOpClassImpl):
    mma_type = MmaType.UMMA

    def __init__(self, m, n, k, a_dtype, b_dtype, cd_dtype, sf_dtype):
        super().__init__(m, n, k, a_dtype, b_dtype, cd_dtype)
        self.sf_is_e4m3 = sf_dtype == dtypes.float8e4m3

    def to_cpp_str(self, include_class_name=False):
        code = super().to_cpp_str()
        code += f"\n  static constexpr bool kSFIsE4M3 = {str(self.sf_is_e4m3).lower()};"
        if include_class_name:
            code = f"class MmaOpClass {{\n{code}\n}};"
        return code

    def generate_ptx(self, indent=0):
        return ""


class WgmmaOpClassImpl:
    def __init__(self, m, n, k, a_dtype, b_dtype, cd_dtype):
        self.shape = (m, n, k)
        self.a_dtype = a_dtype if isinstance(a_dtype, str) else DTYPE_MAP[a_dtype]
        self.b_dtype = b_dtype if isinstance(b_dtype, str) else DTYPE_MAP[b_dtype]
        self.cd_dtype = cd_dtype if isinstance(cd_dtype, str) else DTYPE_MAP[cd_dtype]

        self.reg_b_count = calc_reg_count(n, k, self.b_dtype) // 4
        self.reg_cd_count = calc_reg_count(m, n, self.cd_dtype) // 4
        if self.cd_dtype == "f16":
            self.val_type_cd = "half"
            self.reg_cd_type = "uint32_t"
        elif self.cd_dtype == "bf16":
            self.val_type_cd = "nv_bfloat16"
            self.reg_cd_type = "uint32_t"
        elif self.cd_dtype == "f32":
            self.val_type_cd = "float"
            self.reg_cd_type = "float"
        elif self.cd_dtype == "s32":
            self.val_type_cd = "int32_t"
            self.reg_cd_type = "uint32_t"
        else:
            raise ValueError(f"Invalid cd_dtype: {cd_dtype}")

    def to_cpp_str(self, include_class_name=False):
        reg_cd_type = self.reg_cd_type
        lines = [
            "static constexpr MmaType kMmaType = MmaType::WGMMA;",
            f"using MmaShape = Shape<{self.shape[0]}, {self.shape[1]}, {self.shape[2]}>;",
            "",
            f"using ValTypeC = {self.val_type_cd};",
            f"using ValTypeD = {self.val_type_cd};",
            "",
            f"static constexpr uint32_t kATypeBits = {DTYPE_BIT_WIDTH_MAP[self.a_dtype]};",
            f"static constexpr uint32_t kBTypeBits = {DTYPE_BIT_WIDTH_MAP[self.b_dtype]};",
            f"static constexpr uint32_t kCTypeBits = {DTYPE_BIT_WIDTH_MAP[self.cd_dtype]};",
            f"static constexpr uint32_t kDTypeBits = {DTYPE_BIT_WIDTH_MAP[self.cd_dtype]};",
            "static constexpr bool kNativeMixed = false;",
            "",
            f"using BRegisters = uint32_t[{self.reg_b_count}];",
            f"using CRegisters = {self.reg_cd_type}[{self.reg_cd_count}];",
            f"using DRegisters = {self.reg_cd_type}[{self.reg_cd_count}];",
            "",
            "CUDA_INLINE",
            f"static void fma(uint64_t &desc, uint32_t *b, {reg_cd_type} *d, bool pred = true) {{",
            *self.generate_ptx(indent=2, has_scale_d=True).strip("\n").split("\n"),
            "};",
            "CUDA_INLINE",
            f"static void fma(uint64_t &desc, uint64_t &b_desc, {reg_cd_type} *d, bool pred = true) {{",
            *self.generate_ptx(indent=2, has_scale_d=True, use_ss=True).strip("\n").split("\n"),
            "};",
        ]

        code = "\n".join("  " + x if x else x for x in lines)
        if include_class_name:
            code = f"class MmaOpClass {{\n{code}\n}};"

        return code

    def generate_ptx(self, indent=2, has_scale_d=True, use_ss=False):
        a_dtype = self.a_dtype
        b_dtype = self.b_dtype
        cd_dtype = self.cd_dtype
        m, n, k = self.shape

        # Swap M<->N and A-dtype<->B-dtype in PTX: project's A becomes wgmma's B and
        # project's B becomes wgmma's A. The PTX dtype suffix order is .cd.a.b, so
        # the wgmma A slot takes project's b_dtype and the wgmma B slot takes a_dtype.
        asm_op = f"wgmma.mma_async.sync.aligned.m{n}n{m}k{k}"
        asm_op += f".{cd_dtype}.{b_dtype}.{a_dtype}"
        if "s" in b_dtype:
            asm_op += ".satfinite"

        start = 0
        end = 0
        param_placeholders_list = []
        counts = [self.reg_cd_count, 1 if use_ss else self.reg_b_count]
        for i in range(len(counts)):
            end += counts[i]
            placeholder_str = ", ".join(f"%{x}" for x in range(start, end))
            param_placeholders_list.append("{" + placeholder_str + "}")
            start += counts[i]
        if use_ss:
            param_placeholders_list[1] = f"%{self.reg_cd_count}"
        param_placeholders_list.append(f"%{sum(counts)}")

        other_ptx_args = ", p" if has_scale_d else ", 1"
        # The dtype-specific PTX tail args (scale/trans flags) gate on the wgmma-A
        # operand dtype, which after the swap is project's b_dtype.
        if self.b_dtype in ["f16", "bf16"]:
            other_ptx_args += ", 1, 1, 0, 0" if use_ss else ", 1, 1, 0"
        elif self.b_dtype in ["e4m3", "e5m2", "e2m1"]:
            other_ptx_args += ", 1, 1"

        # Project A's smem descriptor fills the wgmma B operand.
        a_desc_param = ' "l"(desc)'
        # Project B's registers fill the wgmma A operand.
        b_params = []
        cd_params = []
        for i in range(self.reg_b_count):
            b_params.append(f' "r"(b[{i}])')
        if use_ss:
            b_params = [' "l"(b_desc)']
        for i in range(self.reg_cd_count):
            t = "f" if cd_dtype == "f32" else "r"
            cd_params.append(f'"+{t}"(d[{i}])')

        cd_param_str = ""
        for i in range(math.ceil(len(cd_params) / 4)):
            cd_params_part = cd_params[i * 4 : (i + 1) * 4]
            cd_params_part_str = ", ".join(cd_params_part) + ",\n"
            if cd_param_str:
                cd_params_part_str = "    " + cd_params_part_str

            cd_param_str += cd_params_part_str

        cd_param_str = cd_param_str.strip().strip(",")

        if has_scale_d:
            asm_code = f"""
            asm volatile(
              "{{\\n"
                ".reg .pred p;\\n"
                "setp.ne.b32 p, %{sum(counts) + 1}, 0;\\n"
                "{asm_op} "
                "{", ".join(param_placeholders_list)}{other_ptx_args};\\n"
              "}}\\n"
              : {cd_param_str}
              : {", ".join(b_params)},
                {a_desc_param}, "r"((uint32_t)pred)
            );
            """
        else:
            asm_code = f"""
            asm volatile(
            "{asm_op} "
            "{", ".join(param_placeholders_list)}{other_ptx_args};\\n"
            : {cd_param_str}
            : {", ".join(b_params)},
                {a_desc_param}
            );
            """

        space_count = len(re.findall("^\n( +)", asm_code)[0])
        asm_code = asm_code.replace("\n" + " " * space_count, "\n").strip()
        asm_code = "".join("\n" + " " * indent + x for x in asm_code.split("\n"))

        return asm_code


class MxMmaOpClassImpl:
    """Microscale (block-scaled) warp-level ``mma.sync`` for SM120.

    Emits ``mma.sync.aligned.<kind>.block_scale.<scale_vec>...`` which consumes a
    per-block scale factor for each of A and B in addition to the operand
    registers. Three formats are supported, selected by operand/scale dtype:

    * mxfp4  : e2m1 x e2m1, m16n8k64, ``kind::mxf4``,     scale_vec::2X, ue8m0
    * nvfp4  : e2m1 x e2m1, m16n8k64, ``kind::mxf4nvf4``, scale_vec::4X, ue4m3
    * mxfp8  : e4m3 x e4m3, m16n8k32, ``kind::mxf8f6f4``, scale_vec::1X, ue8m0
    """

    def __init__(self, m, n, k, a_dtype, b_dtype, cd_dtype, sf_dtype, scale_vec=None):
        self.scale_vec_int = scale_vec
        self.shape = (m, n, k)
        self.a_dtype = a_dtype if isinstance(a_dtype, str) else DTYPE_MAP[a_dtype]
        self.b_dtype = b_dtype if isinstance(b_dtype, str) else DTYPE_MAP[b_dtype]
        self.cd_dtype = cd_dtype if isinstance(cd_dtype, str) else DTYPE_MAP[cd_dtype]
        sf_dtype = sf_dtype if isinstance(sf_dtype, str) else DTYPE_MAP[sf_dtype]
        if sf_dtype not in SF_DTYPE_MAP:
            raise ValueError(f"Invalid scale-factor dtype for MXMMA: {sf_dtype}")
        self.sf_dtype = sf_dtype
        self.sf_ptx = SF_DTYPE_MAP[sf_dtype]

        if scale_vec is None:
            if self.a_dtype == "e2m1":
                scale_vec = 4 if self.sf_ptx == "ue4m3" else 2
            else:
                scale_vec = 1
        self.scale_vec_size = scale_vec

        if self.a_dtype == "e2m1":
            self.kind = "kind::mxf4nvf4"
            if scale_vec == 4:
                self.scale_vec = "scale_vec::4X"
            elif scale_vec == 2:
                self.scale_vec = "scale_vec::2X"
            else:
                raise ValueError(f"unsupported fp4 scale_vec: {scale_vec} (expected 2 or 4)")
        else:
            assert scale_vec == 1, "fp8/f6f4 microscale uses scale_vec::1X (group == mma K-tile)"
            self.kind = "kind::mxf8f6f4"
            self.scale_vec = "scale_vec::1X"

        self.native_mixed = self.kind == "kind::mxf8f6f4" and DTYPE_BIT_WIDTH_MAP[self.b_dtype] < 8
        b_reg_dtype = self.b_dtype
        if self.kind == "kind::mxf8f6f4" and DTYPE_BIT_WIDTH_MAP[self.b_dtype] < 8:
            b_reg_dtype = "e4m3"
        self.reg_a_count = calc_reg_count(m, k, self.a_dtype)
        self.reg_b_count = calc_reg_count(k, n, b_reg_dtype)
        self.reg_cd_count = calc_reg_count(m, n, self.cd_dtype)
        if self.cd_dtype == "f32":
            self.val_type_cd = "float"
            self.reg_cd_type = "float"
        elif self.cd_dtype == "f16":
            self.val_type_cd = "half"
            self.reg_cd_type = "uint32_t"
        elif self.cd_dtype == "bf16":
            self.val_type_cd = "nv_bfloat16"
            self.reg_cd_type = "uint32_t"
        else:
            raise ValueError(f"Invalid cd_dtype for MXMMA: {cd_dtype}")

    def to_cpp_str(self, include_class_name=False):
        reg_cd_type = self.reg_cd_type
        lines = [
            "static constexpr MmaType kMmaType = MmaType::MXMMA;",
            f"using MmaShape = Shape<{self.shape[0]}, {self.shape[1]}, {self.shape[2]}>;",
            "",
            f"using ValTypeC = {self.val_type_cd};",
            f"using ValTypeD = {self.val_type_cd};",
            "",
            f"static constexpr uint32_t kScaleVec = {self.scale_vec_int};",
            f"static constexpr uint32_t kATypeBits = {DTYPE_BIT_WIDTH_MAP[self.a_dtype]};",
            f"static constexpr uint32_t kBTypeBits = {DTYPE_BIT_WIDTH_MAP[self.b_dtype]};",
            f"static constexpr uint32_t kCTypeBits = {DTYPE_BIT_WIDTH_MAP[self.cd_dtype]};",
            f"static constexpr uint32_t kDTypeBits = {DTYPE_BIT_WIDTH_MAP[self.cd_dtype]};",
            f"static constexpr uint32_t kSFTypeBits = {DTYPE_BIT_WIDTH_MAP[self.sf_dtype]};",
            f"static constexpr bool kSFIsE4M3 = {'true' if self.sf_ptx == 'ue4m3' else 'false'};",
            f"static constexpr bool kNativeMixed = {'true' if self.native_mixed else 'false'};",
            "",
            f"using ARegisters = uint32_t[{self.reg_a_count}];",
            f"using BRegisters = uint32_t[{self.reg_b_count}];",
            f"using CRegisters = {self.reg_cd_type}[{self.reg_cd_count}];",
            f"using DRegisters = {self.reg_cd_type}[{self.reg_cd_count}];",
            "",
            "CUDA_INLINE",
            f"static void fma(uint32_t *a, uint32_t *b, uint32_t sfa, uint32_t sfb, "
            f"{reg_cd_type} *c, {reg_cd_type} *d, "
            f"uint32_t byte_id_a, uint32_t thread_id_a, uint32_t byte_id_b, uint32_t thread_id_b) {{",  # noqa
            *self.generate_ptx(indent=2).strip("\n").split("\n"),
            "};",
        ]

        code = "\n".join("  " + x if x else x for x in lines)
        if include_class_name:
            code = f"class MmaOpClass {{\n{code}\n}};"

        return code

    def generate_ptx(self, indent=0):
        a_dtype = self.a_dtype
        b_dtype = self.b_dtype
        cd_dtype = self.cd_dtype
        shape = self.shape

        asm_op = f"mma.sync.aligned.{self.kind}.block_scale.{self.scale_vec}"
        asm_op += f".m{shape[0]}n{shape[1]}k{shape[2]}.row.col"
        asm_op += f".{cd_dtype}.{a_dtype}.{b_dtype}.{cd_dtype}.{self.sf_ptx}"

        counter = 0

        def take(count):
            nonlocal counter
            group = ", ".join(f"%{counter + i}" for i in range(count))
            counter += count
            return "{" + group + "}"

        d_group = take(self.reg_cd_count)
        a_group = take(self.reg_a_count)
        b_group = take(self.reg_b_count)
        c_group = take(self.reg_cd_count)
        sfa_group = take(1)
        sel_a_group = take(2)
        sfb_group = take(1)
        sel_b_group = take(2)

        t = "f" if cd_dtype == "f32" else "r"
        d_params = ", ".join(f'"+{t}"(d[{i}])' for i in range(self.reg_cd_count))
        a_params = ", ".join(f'"r"(a[{i}])' for i in range(self.reg_a_count))
        b_params = ", ".join(f'"r"(b[{i}])' for i in range(self.reg_b_count))
        c_params = ", ".join(f'"{t}"(c[{i}])' for i in range(self.reg_cd_count))

        asm_code = f"""
        asm volatile(
          "{asm_op} "
          "{d_group}, "
          "{a_group}, "
          "{b_group}, "
          "{c_group}, "
          "{sfa_group}, {sel_a_group}, "
          "{sfb_group}, {sel_b_group};\\n"
          : {d_params}
          : {a_params},
            {b_params},
            {c_params},
            "r"(sfa), "h"((uint16_t)byte_id_a), "h"((uint16_t)thread_id_a),
            "r"(sfb), "h"((uint16_t)byte_id_b), "h"((uint16_t)thread_id_b)
        );
        """

        space_count = len(re.findall("^\n( +)", asm_code)[0])
        asm_code = asm_code.replace("\n" + " " * space_count, "\n").strip()
        asm_code = "".join("\n" + " " * indent + x for x in asm_code.split("\n"))

        return asm_code


class MmaOpClass:
    @classmethod
    def from_config(cls, mma_type, m, n, k, a_dtype, b_dtype, cd_dtype, sf_dtype=None, scale_vec=None):
        mma_type = mma_type if isinstance(mma_type, MmaType) else getattr(MmaType, mma_type.upper())

        if mma_type == MmaType.MMA:
            return MmaOpClassImpl(m, n, k, a_dtype, b_dtype, cd_dtype)
        elif mma_type == MmaType.UMMA:
            assert m in (8, 16) and (n, k) == (8, 256 // a_dtype.num_bits)
            assert a_dtype in (
                dtypes.int8,
                dtypes.bfloat16,
                dtypes.float16,
                dtypes.float8e4m3,
                dtypes.float8e5m2,
                dtypes.float8e3m4,
                dtypes.float4e2m1,
                dtypes.float4e0m3,
            )
            if a_dtype == dtypes.int8 or a_dtype.num_bits == 16:
                assert a_dtype == b_dtype
            else:
                assert b_dtype in (
                    dtypes.float8e4m3,
                    dtypes.float8e5m2,
                    dtypes.float8e3m4,
                    dtypes.float4e2m1,
                    dtypes.float4e0m3,
                    dtypes.float6e3m2,
                    dtypes.float6e2m3,
                )
            assert cd_dtype == (dtypes.int32 if a_dtype == dtypes.int8 else dtypes.float32)
            return UmmaOpClassImpl(m, n, k, a_dtype, b_dtype, cd_dtype, sf_dtype)
        elif mma_type == MmaType.WGMMA:
            return WgmmaOpClassImpl(m, n, k, a_dtype, b_dtype, cd_dtype)
        elif mma_type == MmaType.MXMMA:
            if sf_dtype is None:
                raise ValueError("MXMMA requires sf_dtype (block scale-factor dtype)")
            return MxMmaOpClassImpl(m, n, k, a_dtype, b_dtype, cd_dtype, sf_dtype, scale_vec=scale_vec)
        else:
            raise ValueError(f"Invalid MMA Type: {mma_type}")
