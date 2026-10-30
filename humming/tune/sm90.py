from enum import Enum

from humming import dtypes
from humming.config import GemmType, LayerConfig, MmaType
from humming.device import current_device
from humming.tune.candidate import (
    DeviceProfile,
    ScheduleCandidate,
    TuningDecision,
    TuningProblem,
    analyze_candidate,
)
from humming.tune.sm8x import Sm80Heuristics
from humming.tune.sm90_policies import (
    Sm90CandidatePolicy,
    build_sm90_seed_config,
    calc_sm90_num_block_list,
    fit_sm90_tile,
    get_compact_moe_config,
    get_sm90_specialized_config,
    get_small_m_packed_fp4_config,
    select_grouped_scale,
    select_moe_a16,
    select_sm90_small_m_schedule,
)
from humming.utils.smem import estimate_smem_size_layer


class Sm90Heuristics(Sm80Heuristics):
    max_smem_size: int = 227 * 1024
    candidate_policy = Sm90CandidatePolicy()
    b16_allowed_dtypes: list[dtypes.DataType] = [dtypes.float16, dtypes.bfloat16]
    b8_allowed_dtypes: list[dtypes.DataType] = [
        dtypes.int8,
        dtypes.float8e4m3,
        dtypes.float8e5m2,
    ]
    b4_allowed_dtypes: list[dtypes.DataType] = []
    sm_version: int = 90

    @classmethod
    def get_device_profile(cls, *, include_grid_size: bool) -> DeviceProfile:
        return DeviceProfile(
            name=f"sm{cls.sm_version}",
            sm_version=cls.sm_version,
            num_sms=current_device.sm_count if include_grid_size else None,
            max_smem_size=cls.max_smem_size,
        )

    @classmethod
    def _make_problem(
        cls,
        layer_config: LayerConfig,
        shape_m: int,
        use_f16_accum: bool,
        use_batch_invariant: bool,
        gemm_type: GemmType,
        *,
        include_grid_size: bool = False,
        use_m_major_input_scale: bool = False,
    ) -> TuningProblem:
        return TuningProblem(
            layer_config=layer_config,
            shape_m=shape_m,
            gemm_type=gemm_type,
            device=cls.get_device_profile(include_grid_size=include_grid_size),
            use_f16_accum=use_f16_accum,
            use_batch_invariant=use_batch_invariant,
            use_m_major_input_scale=use_m_major_input_scale,
            expert_probability_cv=cls.expert_probability_cv,
        )

    @classmethod
    def _uses_moe_a16_policy(
        cls,
        layer_config: LayerConfig,
        use_batch_invariant: bool,
        gemm_type: GemmType,
    ) -> bool:
        return gemm_type != GemmType.DENSE and layer_config.a_dtype.num_bits == 16 and not use_batch_invariant

    @classmethod
    def get_config1(
        cls,
        layer_config: LayerConfig,
        shape_m: int,
        use_f16_accum: bool = False,
        use_batch_invariant: bool = False,
        gemm_type: GemmType = GemmType.DENSE,
    ):
        problem = cls._make_problem(
            layer_config,
            shape_m,
            use_f16_accum,
            use_batch_invariant,
            gemm_type,
        )
        return build_sm90_seed_config(problem)

    @classmethod
    def get_config2(
        cls,
        layer_config: LayerConfig,
        shape_m: int,
        use_f16_accum: bool = False,
        use_batch_invariant: bool = False,
        gemm_type: GemmType = GemmType.DENSE,
    ):
        problem = cls._make_problem(
            layer_config,
            shape_m,
            use_f16_accum,
            use_batch_invariant,
            gemm_type,
        )
        return select_grouped_scale(problem).to_config()

    @classmethod
    def calc_num_block_list(
        cls,
        layer_config: LayerConfig,
        shape_m: int,
        max_block_m: int,
    ):
        return calc_sm90_num_block_list(
            layer_config, shape_m, max_block_m, expert_probability_cv=cls.expert_probability_cv
        )

    @classmethod
    def _uses_grouped_scale_candidates(
        cls,
        layer_config: LayerConfig,
    ) -> bool:
        if layer_config.a_dtype.num_bits == 16:
            return False
        if layer_config.use_packed_k_layout:
            return False
        if layer_config.input_scale_group_size == 0 and layer_config.weight_scale_group_size == 0:
            return False
        return not (layer_config.use_fused_e8m0_scale and layer_config.input_scale_group_size == 0)

    @classmethod
    def _get_candidate_decision(
        cls,
        layer_config: LayerConfig,
        shape_m: int,
        use_f16_accum: bool = False,
        use_batch_invariant: bool = False,
        gemm_type: GemmType = GemmType.DENSE,
        use_m_major_input_scale: bool = False,
    ) -> TuningDecision:
        tune_moe_a16 = cls._uses_moe_a16_policy(
            layer_config,
            use_batch_invariant,
            gemm_type,
        )
        problem = cls._make_problem(
            layer_config,
            shape_m,
            use_f16_accum,
            use_batch_invariant,
            gemm_type,
            include_grid_size=gemm_type != GemmType.DENSE,
            use_m_major_input_scale=use_m_major_input_scale,
        )
        if cls._uses_grouped_scale_candidates(layer_config):
            return select_grouped_scale(problem)
        if not tune_moe_a16:
            raise ValueError("decision traces are only available for migrated SM90 policies")
        return select_moe_a16(
            problem,
            cls.candidate_policy,
        )

    @classmethod
    def _select_base_config(
        cls,
        layer_config: LayerConfig,
        shape_m: int,
        use_f16_accum: bool = False,
        use_batch_invariant: bool = False,
        gemm_type: GemmType = GemmType.DENSE,
        use_m_major_input_scale: bool = False,
    ):
        shape_m = cls.get_shape_m_sample(layer_config, shape_m)
        specialized_config = get_sm90_specialized_config(
            layer_config,
            shape_m,
            gemm_type,
            use_f16_accum,
            use_batch_invariant,
            use_m_major_input_scale=use_m_major_input_scale,
            expert_probability_cv=cls.expert_probability_cv,
        )
        if specialized_config is not None:
            return specialized_config

        is_wna16 = layer_config.a_dtype.num_bits == 16 and layer_config.b_dtype.num_bits < 16
        has_short_reduction = shape_m <= 32 and layer_config.shape_k <= 1024
        is_tile_aligned = layer_config.shape_n % 128 == 0 and layer_config.shape_k % 64 == 0
        can_use_small_mma = gemm_type == GemmType.DENSE and not use_f16_accum and not use_batch_invariant
        if is_wna16 and has_short_reduction and is_tile_aligned and can_use_small_mma:
            # A short reduction cannot amortize a producer warpgroup or Stream-K
            # fixups. Two K warps let MMA dequantize while keeping the M tile small.
            block_m = 8 if shape_m <= 8 else 16
            return {
                "mma_type": MmaType.MMA.value,
                "block_shape": (block_m, 128, 64),
                "warp_shape": (block_m, 32, 32),
                "num_stages": 4,
                "num_ctas_per_sm": 1,
                "use_stream_k": False,
                "use_tma": False,
                "use_warp_spec": False,
            }

        if layer_config.a_dtype == dtypes.int4:
            return super().get_config(
                layer_config, shape_m, use_f16_accum, use_batch_invariant, gemm_type, use_m_major_input_scale
            )

        use_grouped_scale_candidates = cls._uses_grouped_scale_candidates(layer_config)
        use_moe_a16_policy = cls._uses_moe_a16_policy(layer_config, use_batch_invariant, gemm_type)
        use_candidates = use_grouped_scale_candidates or use_moe_a16_policy
        if use_candidates:
            return cls._get_candidate_decision(
                layer_config,
                shape_m,
                use_f16_accum,
                use_batch_invariant,
                gemm_type,
                use_m_major_input_scale,
            )

        config = cls.get_config1(
            layer_config,
            shape_m,
            use_f16_accum,
            use_batch_invariant,
            gemm_type,
        )
        while config["num_stages"] > 3:
            smem_size = estimate_smem_size_layer(
                layer_config,
                config["block_shape"],
                gemm_type,
                config["num_stages"],
                warp_shape=config["warp_shape"],
                smem_reuse_mode=config.get("smem_reuse_mode", "all_stages"),
                use_tma=config.get("use_tma", False),
                use_warp_spec=config.get("use_warp_spec", False),
                output_chunk_rows=config.get("output_chunk_rows", 0),
                mma_accum_bits=16 if use_f16_accum else 32,
            )
            if smem_size <= cls.max_smem_size:
                break
            config["num_stages"] -= 1
        return config

    @classmethod
    def _select_config(
        cls,
        layer_config: LayerConfig,
        shape_m: int,
        use_f16_accum: bool = False,
        use_batch_invariant: bool = False,
        gemm_type: GemmType = GemmType.DENSE,
        use_m_major_input_scale: bool = False,
    ):
        packed_fp4_small_m_config = get_small_m_packed_fp4_config(
            layer_config,
            shape_m,
            gemm_type,
            use_f16_accum,
            use_batch_invariant,
            use_m_major_input_scale=use_m_major_input_scale,
        )
        if packed_fp4_small_m_config is not None:
            return packed_fp4_small_m_config

        selected = cls._select_base_config(
            layer_config, shape_m, use_f16_accum, use_batch_invariant, gemm_type, use_m_major_input_scale
        )
        is_wna16 = layer_config.a_dtype.num_bits == 16 and layer_config.b_dtype.num_bits < 16
        has_aligned_reduction = layer_config.shape_k >= 4096 and layer_config.shape_k % 128 == 0
        can_use_mma = is_wna16 and has_aligned_reduction and layer_config.shape_n % 128 == 0
        can_use_mma &= not use_f16_accum and not use_batch_invariant and not layer_config.use_packed_k_layout
        if not can_use_mma or (gemm_type == GemmType.DENSE and shape_m > 32):
            return selected
        problem = cls._make_problem(
            layer_config,
            cls.get_shape_m_sample(layer_config, shape_m),
            use_f16_accum,
            use_batch_invariant,
            gemm_type,
            include_grid_size=True,
            use_m_major_input_scale=use_m_major_input_scale,
        )
        return select_sm90_small_m_schedule(problem, selected)

    @classmethod
    def get_config(
        cls,
        layer_config: LayerConfig,
        shape_m: int,
        use_f16_accum: bool = False,
        use_batch_invariant: bool = False,
        gemm_type: GemmType = GemmType.DENSE,
        use_m_major_input_scale: bool = False,
    ):
        selected = cls._select_config(
            layer_config, shape_m, use_f16_accum, use_batch_invariant, gemm_type, use_m_major_input_scale
        )
        config = selected.to_config() if isinstance(selected, TuningDecision) else selected
        config = fit_sm90_tile(
            layer_config,
            config,
            use_f16_accum,
            gemm_type=gemm_type,
            use_m_major_input_scale=use_m_major_input_scale,
        )
        config = get_compact_moe_config(
            layer_config,
            config,
            gemm_type,
            use_f16_accum,
            use_batch_invariant,
            use_m_major_input_scale=use_m_major_input_scale,
        )
        return {key: value.value if isinstance(value, Enum) else value for key, value in config.items()}

    @classmethod
    def get_tuning_decision(
        cls,
        layer_config: LayerConfig,
        shape_m: int,
        use_f16_accum: bool = False,
        use_batch_invariant: bool = False,
        gemm_type: GemmType = GemmType.DENSE,
        use_m_major_input_scale: bool = False,
    ) -> TuningDecision:
        selected = cls._select_config(
            layer_config, shape_m, use_f16_accum, use_batch_invariant, gemm_type, use_m_major_input_scale
        )
        decision = selected if isinstance(selected, TuningDecision) else None
        config = decision.to_config() if decision is not None else selected
        fitted = fit_sm90_tile(
            layer_config,
            config,
            use_f16_accum,
            gemm_type=gemm_type,
            use_m_major_input_scale=use_m_major_input_scale,
        )
        fitted = get_compact_moe_config(
            layer_config,
            fitted,
            gemm_type,
            use_f16_accum,
            use_batch_invariant,
            use_m_major_input_scale=use_m_major_input_scale,
        )
        if decision is not None and fitted == config:
            return decision
        problem = cls._make_problem(
            layer_config,
            cls.get_shape_m_sample(layer_config, shape_m),
            use_f16_accum,
            use_batch_invariant,
            gemm_type,
            include_grid_size=True,
            use_m_major_input_scale=use_m_major_input_scale,
        )
        candidate = ScheduleCandidate.from_config("sm90_selected", fitted)
        analysis = analyze_candidate(problem, candidate)
        return TuningDecision(
            problem=problem,
            family="sm90",
            selected=candidate,
            considered=(*decision.considered, analysis) if decision is not None else (analysis,),
            reason="selected by the runtime SM90 configuration policy",
        )
