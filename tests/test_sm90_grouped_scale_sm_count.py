"""SM-count-driven H200 W4A8 MoE calibrations in the grouped-scale policy.

PR#76 replaced the device-name-dispatched Sm90H200Heuristics with three
SM-count-gated calibrations inside the generic sm90 grouped-scale policy:

  K1  2-CTA residency cap (post-adjustment on candidate analysis)
  K2  large per_expert_m >= 48 -> (block_m 128|64, block_n 256, block_k 128)
      single resident CTA with stream-K off
  K3  small-M W4A8 wide tile (block_n 512, block_k 64, warp_n 64) at
      two resident CTAs

The gate is problem.device.num_sms >= 128, so 114-SM H100 PCIe keeps the
generic behavior byte-for-byte.
"""

import pytest

from humming import dtypes
from humming.config import GemmType, LayerConfig, MmaType
from humming.tune.candidate import DeviceProfile, TuningProblem
from humming.tune.sm90_policies import select_grouped_scale


def _w4a8_moe_layer(
    shape_n: int = 4096,
    shape_k: int = 4096,
    num_experts: int = 288,
) -> LayerConfig:
    return LayerConfig(
        shape_n=shape_n,
        shape_k=shape_k,
        num_experts=num_experts,
        a_dtype=dtypes.float8e4m3,
        b_dtype=dtypes.int4,
        c_dtype=dtypes.bfloat16,
        bs_dtype=dtypes.bfloat16,
        weight_scale_group_size=128,
        mma_type=MmaType.WGMMA,
        use_packed_k_layout=False,
    )


def _problem(
    layer_config: LayerConfig,
    shape_m: int,
    *,
    num_sms: int,
    gemm_type: GemmType = GemmType.INDEXED,
) -> TuningProblem:
    return TuningProblem(
        layer_config=layer_config,
        shape_m=shape_m,
        gemm_type=gemm_type,
        device=DeviceProfile(
            name=f"sm90-{num_sms}sms",
            sm_version=90,
            num_sms=num_sms,
            max_smem_size=227 * 1024,
        ),
    )


def _assert_vllm_constraints(config: dict) -> None:
    # vLLM's _fixup_moe_tuning_config rejects K-blocks > 128 and requires
    # warp_n >= 32 whenever block_n % 32 == 0.
    assert config["block_shape"][2] <= 128
    if config["block_shape"][1] % 32 == 0:
        assert config["warp_shape"][1] >= 32


class TestLargePerExpertM:
    def test_num_sms_132_selects_256n_tile_with_stream_k_off(self):
        # M=16384 over 288 experts -> per_expert_m ~57 (>= 48, < 96).
        problem = _problem(_w4a8_moe_layer(), 16384, num_sms=132)
        decision = select_grouped_scale(problem)

        config = decision.to_config()
        assert config["block_shape"] == (64, 256, 128)
        assert config["warp_shape"] == (64, 32, 128)
        assert config["num_ctas_per_sm"] == 1
        assert config["use_stream_k"] is False
        _assert_vllm_constraints(config)
        assert decision.selected.candidate_id == "grouped_scale_large_m_n256_k128"

    def test_per_expert_m_above_96_uses_block_m_128(self):
        # 288 experts * 96 rows = 27648 tokens -> per_expert_m 96.
        problem = _problem(_w4a8_moe_layer(), 288 * 100, num_sms=132)
        decision = select_grouped_scale(problem)

        config = decision.to_config()
        assert config["block_shape"][0] == 128
        assert config["block_shape"][1] == 256
        assert config["num_ctas_per_sm"] == 1
        assert config["use_stream_k"] is False
        _assert_vllm_constraints(config)

    def test_per_expert_m_below_48_keeps_generic_tile(self):
        # M=8192 -> per_expert_m ~28: below the K2 threshold.
        problem = _problem(_w4a8_moe_layer(), 8192, num_sms=132)
        decision = select_grouped_scale(problem)

        config = decision.to_config()
        assert decision.selected.candidate_id != "grouped_scale_large_m_n256_k128"
        assert config["block_shape"][1] != 256


class TestTwoCtaResidencyCap:
    def test_num_sms_132_caps_generic_candidates_at_two_ctas(self):
        # Mid-M generic ladder candidate would otherwise allow higher
        # residency; K1 clamps it to 2 on a >=128-SM grid.
        problem = _problem(_w4a8_moe_layer(), 4096, num_sms=132)
        decision = select_grouped_scale(problem)

        config = decision.to_config()
        assert config["num_ctas_per_sm"] <= 2

    def test_cap_only_applies_when_calibrations_enabled(self):
        # Same shape on a 114-SM grid must not emit the calibration cap.
        problem = _problem(_w4a8_moe_layer(), 4096, num_sms=114)
        decision = select_grouped_scale(problem)

        config = decision.to_config()
        # Generic ladder never sets num_ctas_per_sm explicitly; the field
        # stays absent exactly as before the migration.
        assert "num_ctas_per_sm" not in config


class TestH100PcieGate:
    @pytest.mark.parametrize(
        ("shape_m", "shape_n", "shape_k"),
        [
            (64, 4096, 4096),
            (2048, 4096, 4096),
            (16384, 4096, 4096),
            (64, 4096, 2048),
            (16384, 4096, 2048),
        ],
    )
    def test_num_sms_114_matches_pre_migration_behavior(
        self,
        shape_m,
        shape_n,
        shape_k,
    ):
        # Gate closed: selection is the first legal generic candidate with
        # no calibration candidates generated and no residency override.
        layer = _w4a8_moe_layer(shape_n=shape_n, shape_k=shape_k)
        problem = _problem(layer, shape_m, num_sms=114)
        decision = select_grouped_scale(problem)

        config = decision.to_config()
        assert decision.reason == "selected the first legal measured-priority candidate"
        assert decision.selected.candidate_id.startswith("grouped_scale_n")
        assert "num_ctas_per_sm" not in config
        assert config["use_stream_k"] is True

    def test_num_sms_none_disables_calibrations(self):
        # No SM count available (e.g. decision without a live device):
        # behave exactly like the ungated generic policy.
        problem = TuningProblem(
            layer_config=_w4a8_moe_layer(),
            shape_m=16384,
            gemm_type=GemmType.INDEXED,
            device=DeviceProfile(
                name="sm90",
                sm_version=90,
                num_sms=None,
                max_smem_size=227 * 1024,
            ),
        )
        decision = select_grouped_scale(problem)

        assert decision.reason == "selected the first legal measured-priority candidate"
        assert decision.selected.candidate_id.startswith("grouped_scale_n")


class TestSmallMWideTile:
    @pytest.mark.parametrize(
        ("shape_m", "expected_block_m"),
        [(64, 8), (512, 16), (2048, 24)],
    )
    def test_num_sms_132_selects_wide_tile(self, shape_m, expected_block_m):
        problem = _problem(_w4a8_moe_layer(), shape_m, num_sms=132)
        decision = select_grouped_scale(problem)

        config = decision.to_config()
        assert decision.selected.candidate_id == "grouped_scale_small_m_n512_k64"
        assert config["block_shape"] == (expected_block_m, 512, 64)
        assert config["warp_shape"][1] == 64
        assert config["num_ctas_per_sm"] == 2
        assert config["num_stages"] == 3
        _assert_vllm_constraints(config)

    def test_w2_shape_selects_wide_tile(self):
        problem = _problem(
            _w4a8_moe_layer(shape_n=4096, shape_k=2048),
            2048,
            num_sms=132,
        )
        decision = select_grouped_scale(problem)

        config = decision.to_config()
        assert config["block_shape"][1:] == (512, 64)
        assert config["num_ctas_per_sm"] == 2

    def test_dense_batch_above_threshold_skips_wide_tile(self):
        # Dense (non-MoE) layers never take the small-M MoE wide tile.
        layer = _w4a8_moe_layer(num_experts=0)
        problem = _problem(layer, 64, num_sms=132)
        decision = select_grouped_scale(problem)

        assert decision.selected.candidate_id != "grouped_scale_small_m_n512_k64"

    def test_non_w4a8_mix_skips_wide_tile(self):
        # float4e2m1 (MXFP4) weights follow the generic policy even at
        # small M: the calibration is for int4 W4A8 repacked weights.
        layer = LayerConfig(
            shape_n=4096,
            shape_k=4096,
            num_experts=288,
            a_dtype=dtypes.float8e4m3,
            b_dtype=dtypes.float4e2m1,
            c_dtype=dtypes.bfloat16,
            bs_dtype=dtypes.bfloat16,
            weight_scale_group_size=128,
            mma_type=MmaType.WGMMA,
            use_packed_k_layout=False,
        )
        problem = _problem(layer, 64, num_sms=132)
        decision = select_grouped_scale(problem)

        assert decision.selected.candidate_id != "grouped_scale_small_m_n512_k64"
        assert decision.selected.warp_shape[1] == 32
