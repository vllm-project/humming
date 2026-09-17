"""Direct production-path fixtures shared by tests and the H100 runner."""

import dataclasses

import torch

from humming import dtypes
from humming.config import LayerConfig, MmaType
from humming.config.ldmatrix_s4 import LEGACY_LOADER, NEW_LOADER, assert_layout_compatible
from humming.forward import humming_forward
from humming.kernel.humming import HummingKernel
from humming.transform import transform_humming_tensors
from humming.tune import get_heuristics_config


def make_config(n=128, k=128, *, legacy=False, **overrides):
    values = dict(
        shape_n=n,
        shape_k=k,
        sm_version=90,
        mma_type=MmaType.WGMMA,
        a_dtype=dtypes.int8,
        b_dtype=dtypes.uint4,
        c_dtype=dtypes.bfloat16,
        bs_dtype=dtypes.bfloat16,
        weight_scale_group_size=128,
        use_int_weight_scale=False,
        test_force_packed_k_legacy=legacy,
    )
    return LayerConfig(**(values | overrides))


def weight_data(n, k, *, random=False, nonuniform=False):
    generator = torch.Generator(device="cuda").manual_seed(314159)
    if random:
        codes = torch.randint(0, 16, (n, k), device="cuda", generator=generator, dtype=torch.int32)
    else:
        rows = torch.arange(n, device="cuda")[:, None]
        cols = torch.arange(k, device="cuda")[None, :]
        codes = ((rows * 5 + cols * 3 + cols // 7 + rows // 3) % 16).to(torch.int32)
    shifts = torch.arange(8, device="cuda", dtype=torch.int64) * 4
    packed = (codes.reshape(n, k // 8, 8).long() << shifts).sum(-1).to(torch.int32)
    scales = torch.ones((n, k // 128), device="cuda", dtype=torch.bfloat16)
    if nonuniform:
        scales *= torch.arange(1, k // 128 + 1, device="cuda")[None, :]
        scales *= (torch.arange(n, device="cuda") % 3 + 1)[:, None]
    reference = (codes.float() - 8) * scales.float().repeat_interleave(128, dim=1)
    return {"weight": packed, "weight_scale": scales}, reference


def input_data(m, k, seed=0):
    generator = torch.Generator(device="cuda").manual_seed(2718 + seed)
    return torch.randint(-4, 5, (m, k), device="cuda", dtype=torch.int8, generator=generator)


@dataclasses.dataclass
class PreparedDense:
    config: LayerConfig
    tensors: dict
    reference: torch.Tensor
    layout_identity: str

    @classmethod
    def create(cls, config, tensors, reference):
        prepared = transform_humming_tensors(config, {key: value.clone() for key, value in tensors.items()})
        return cls(config, prepared, reference, config.b_layout_identity)

    def run(self, inputs, *, tuning=None):
        assert_layout_compatible(self.layout_identity, self.config.b_layout_identity)
        if tuning is None:
            tuning = get_heuristics_config(self.config, shape_m=inputs.shape[0], gemm_type="dense")
        configs = HummingKernel.prepare_kernels(
            self.config.to_str(),
            {"gemm_type": "dense"},
            tuning,
        ).reshape(-1, 4)
        kernel = HummingKernel._id2kernel[int(configs[0, 2])]
        expected = NEW_LOADER if self.config.use_ldmatrix_s4 else LEGACY_LOADER
        assert kernel.selected_loader_variant == expected
        assert_layout_compatible(self.layout_identity, kernel.b_layout_identity)
        output = humming_forward(
            self.config,
            inputs,
            **self.tensors,
            input_scale=torch.full((inputs.shape[0], 1), 1 / 128, device="cuda"),
            compute_config={"gemm_type": "dense"},
            tuning_config=tuning,
        )
        return output, kernel

    def oracle(self, inputs):
        # Reference uses logical codes and scales, never the production repack/loader.
        return ((inputs.float() / 128) @ self.reference.T).to(torch.bfloat16)


def assert_output(output, reference, kernel):
    assert output.dtype == reference.dtype == torch.bfloat16
    assert output.shape == reference.shape
    assert output.stride() == reference.stride()
    torch.testing.assert_close(output, reference, rtol=0.01, atol=0.05)
    error = (output.float() - reference.float()).abs()
    return dict(
        selected_loader_variant=kernel.selected_loader_variant,
        layout_identity=kernel.b_layout_identity,
        specialization_identity=kernel.specialization_identity,
        dtype=str(output.dtype),
        shape=list(output.shape),
        stride=list(output.stride()),
        differing_elements=int(torch.count_nonzero(error)),
        max_abs_error=float(error.max()),
        mean_abs_error=float(error.mean()),
    )


def compare_outputs(output, legacy_output, reference, kernel, *, m, n, k, scale_pattern):
    report = assert_output(output, reference, kernel)
    torch.testing.assert_close(output, legacy_output, rtol=0.01, atol=0.05)
    error = (output.float() - legacy_output.float()).abs()
    report.update(
        status="passed",
        loader_mode="automatic" if kernel.use_ldmatrix_s4 else "force_legacy",
        repack_layout_identity=kernel.b_layout_identity,
        fallback_reason=",".join(kernel.loader_selection_reasons),
        output_shape=report["shape"],
        shape={"m": m, "n": n, "k": k},
        scale_pattern=scale_pattern,
        num_differing_elements_vs_legacy=int(torch.count_nonzero(error)),
        max_abs_error_vs_legacy=float(error.max()),
        max_abs_error_vs_reference=report["max_abs_error"],
    )
    return report


VALIDATION_RESULTS: list[dict] = []


def record_result(case_id, report):
    VALIDATION_RESULTS.append(dict(report, case_id=case_id))


def record_selection(case_id, kernel, loader_mode):
    record_result(
        case_id,
        dict(
            status="passed",
            loader_mode=loader_mode,
            selected_loader_variant=kernel.selected_loader_variant,
            repack_layout_identity=kernel.b_layout_identity,
            specialization_identity=kernel.specialization_identity,
            fallback_reason=",".join(kernel.loader_selection_reasons),
        ),
    )


def record_rejection(case_id, reason):
    record_result(
        case_id,
        dict(
            status="passed",
            loader_mode="automatic",
            selected_loader_variant="controlled_rejection",
            fallback_reason=str(reason),
            kernel_launched=False,
        ),
    )
