# gfx950 MFMA weight-only GEMM: fragment and weight layouts

`C[M, N] = A[M, K] @ dequant(B)[N, K]^T`. A and C are fp16 or bf16, and accumulation is
fp32 in `v_mfma_f32_16x16x32_{f16,bf16}`.

- Kernel: `humming/include/humming/kernel/humming_mfma.cuh`
- MFMA wrappers: `humming/include/humming/mma/mfma.cuh`
- Host layout: `humming/rocm/layout.py`
- Standalone launcher: `tests/rocm/mfma_gemm.py`

## MFMA fragment layouts (wave64)

These mappings were checked on MI355X by `tests/rocm/test_mfma_layout.py`. Lane is
`l = 0..63`, slot is `s = 0..7` (the 16-bit values in 4 VGPRs, packed as (even, odd)
pairs), and the C register is `r`.

| instruction | A[m][k] | B[k][n] | C[m][n] |
|---|---|---|---|
| 16x16x32 | m = l%16, k = (l/16)*8 + s | n = l%16, k = (l/16)*8 + s | n = l%16, m = (l/16)*4 + r, r=0..3 |
| 32x32x16 | m = l%32, k = (l/32)*8 + s | n = l%32, k = (l/32)*8 + s | n = l%32, m = (r/4)*8 + (l/32)*4 + r%4, r=0..15 |

The kernel uses 16x16x32 only. Each lane holds 8 consecutive K values of one B column,
which is exactly what one Humming `dequant<ElementQB, Float16>(qb, regs, t)` call
produces (4 registers = 8 halves).

## B (quantized weight) layout

The weight is split into chunks of 16 N x 128 K. Each chunk is 64 lanes x `bits` int32
words:
- the words are contiguous per lane
- lanes are contiguous within the chunk
- chunks are ordered `[N/16][K/128]`, so a wave streams along K

Lane `l` loads its `bits` words with one buffer load (b128/b96/b64/b32 plus a remainder)
and calls `dequant(words, regs, t)` for `t = 0..3`. Call `t` feeds MFMA `t` of the chunk:

    n = 16 * n_tile + l % 16
    k = 128 * chunk + 32 * t + (l / 16) * 8 + s

Within a lane's words the bits follow Humming's own packing. Codes are first packed into
`P = next_pow2(bits)` padded words in the order the dequant reads them, then squeezed to
`bits` words the same way `humming_pack_weight` does it (3/5/6/7 bits). The total size
equals the standard packed weight, so the tensor keeps the shape `[N, K*bits/32]` (int32).

Producing it:

```python
from humming.rocm import layout
w_mfma  = layout.repack_weight_for_mfma(packed_weight, bits)    # from Humming [N, K*bits/32] packing
w_mfma  = layout.repack_codes_for_mfma(codes_int, bits)         # or from raw codes [N, K]
scale   = layout.transform_weight_scale(scale_nk, act_dtype)    # [N, groups] -> [groups, N], act dtype
zp      = layout.transform_zero_point(zp_nk)                    # [N, groups] ints -> [groups, N] uint8
```

Requirements: N must be a multiple of 64 (the kernel; the layout itself needs 16), K a
multiple of 128, and bits 1..8. Symmetric weights use the implicit zero point
`2^(bits-1)`. With an explicit zero point the dequant is `(code - zp) * scale`.

## Scales and zero points

- Scales are `[num_groups, N]` in the activation dtype. `num_groups = K / group` for
  group 32/64/128, or 1 for per-channel (`kGroupSize = 0`).
- Zero points are `uint8 [num_groups, N]`.
- Each lane loads the scale and zp of its column `l % 16` with b16/b8 buffer loads,
  one per group, prefetched with the B chunk.
- Per-channel scales are applied to the fp32 accumulator in the epilogue.
- Group scales go through one of two paths:
  - `kScaleB`: multiply the dequantized B fragment before the MFMA.
  - C-side: rescale the accumulator per group.

## Dequant: Humming's code, used as-is for both dtypes

`MfmaDequant` calls Humming's `dequant<ElementQB, Float16, kHasZeroPoint, false>`, the
magic-number fp16 dequant built on `lop3` / `sub`. It is exact for codes up to 10 bits
(9 with a zero point). On HIP:
- `lop3` maps to gfx950 `v_bitop3_b32`, which uses the same LUT convention as PTX.
- Packed fp16 math maps to `v_pk_add_f16` / `v_pk_mul_f16`.

For bf16 activations the B fragment is still dequantized in fp16, because gfx950 has no
packed bf16 add or multiply and Humming's bf16 path would emulate them in fp32. Then:
1. Each fp16 pair is widened with `v_cvt_f32_f16`.
2. When `kScaleB` is set, the pair is multiplied by the group scale with `v_pk_mul_f32`.
3. The pair is narrowed to bf16 with `v_cvt_pk_bf16_f32`.

That is the only fork from the CUDA dequant: a 2-value conversion helper. bf16 errors
(~4e-3 relative) come from bf16 rounding of A, B and C, not from the dequant.

## LDS A staging (`kLdsA`)

The block's A chunk (`block_m x 128` values) is loaded once by all threads and stored
to a double-buffered LDS ring, with one `__syncthreads` per chunk. 16-byte unit `u`
(8 K values) of row `r` sits at slot `u ^ (r % 16)`. Both the stores and the
`ds_read_b128` fragment reads are therefore free of bank conflicts.

## Kernel API

```cpp
template <uint32_t kWarpsM, uint32_t kWarpsN, uint32_t kWarpTilesM, uint32_t kWarpTilesN,
          uint32_t kStages, bool kLdsA, bool kScaleB>
struct MfmaTileConfig;   // kBlockM = kWarpsM*kWarpTilesM*16, kBlockN = kWarpsN*kWarpTilesN*16

template <class ElementA /* Float16 | BFloat16 */, class ElementQB /* UInt1..UInt8 */,
          class TileConfig, uint32_t kGroupSize /* 0 = per channel */, bool kHasZeroPoint>
__global__ void humming_mfma(const uint16_t *A, const uint32_t *B, const uint16_t *BS,
                             const uint8_t *BZP, uint16_t *C, float *workspace,
                             uint32_t *tile_counters, uint32_t M, uint32_t N, uint32_t K);
```

- Launch: grid `(ceil(N / kBlockN), ceil(M / kBlockM), num_splits)`, block
  `kNumThreads = kWarpsM * kWarpsN * 64`, no dynamic LDS.
- `__launch_bounds__(kNumThreads, 2)` caps VGPR + AGPR at 256.
- Split-K (`gridDim.z > 1`):
  - `workspace` needs `num_splits * num_tiles * kBlockM * kBlockN` floats, with
    `num_tiles = gridDim.x * gridDim.y`.
  - `tile_counters` needs `num_tiles` uint32 values, zeroed once. The kernel resets
    them, so they can be reused across launches and graph replays.
  - Partials are stored in accumulator-fragment order with sc1 (device-coherent)
    dwordx4 stores. The last block of a tile (relaxed atomic counter) reduces them.
- Tile selection used by `tests/rocm/mfma_gemm.py::select_tile`:

  | M | tile (wm,wn,tm,tn,stages) | LDS A | scale_b |
  |---|---|---|---|
  | <= 8 | 1,4,1,{1,2},4 | no | no |
  | <= 32 | 1,4,2,2,2 | yes | yes |
  | <= 64 | 1,4,4,2,2 | yes | yes |
  | > 64 | 1,4,8,2,2 | yes | yes |

  Split-K doubles (up to 8) while there are fewer blocks than CUs and each split keeps
  at least 4 K chunks.
