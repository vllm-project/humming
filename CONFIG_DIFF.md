# Old Sm90H200Heuristics vs new Sm90Heuristics policy config diff

- Device: fake H200, num_sms=132, max_smem=227 KiB
- Shapes: w13 (N=4096, K=4096), w2 (N=4096, K=2048); 288 experts, fp8e4m3 a / int4 group-128 b / bf16 scales
- M sweep: 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384
- Gemm types: INDEXED, GROUPED_MASKED

## Per-shape diff

| shape | gemm | M | per_expert_m | field | old | new | attribution |
|---|---|---|---|---|---|---|---|
| w13 N4096 K4096 | INDEXED | 64 | 0.22 | num_sms | 132 | None | GENERIC (policy infra field) |
| w13 N4096 K4096 | INDEXED | 128 | 0.44 | num_sms | 132 | None | GENERIC (policy infra field) |
| w13 N4096 K4096 | INDEXED | 256 | 0.89 | num_sms | 132 | None | GENERIC (policy infra field) |
| w13 N4096 K4096 | INDEXED | 512 | 1.78 | block_shape | (8, 512, 64) | (16, 512, 64) | UNATTRIBUTED |
| w13 N4096 K4096 | INDEXED | 512 | 1.78 | warp_shape | (8, 64, 64) | (16, 64, 64) | UNATTRIBUTED |
| w13 N4096 K4096 | INDEXED | 512 | 1.78 | num_sms | 132 | None | GENERIC (policy infra field) |
| w13 N4096 K4096 | INDEXED | 1024 | 3.56 | block_shape | (8, 512, 64) | (16, 512, 64) | UNATTRIBUTED |
| w13 N4096 K4096 | INDEXED | 1024 | 3.56 | warp_shape | (8, 64, 64) | (16, 64, 64) | UNATTRIBUTED |
| w13 N4096 K4096 | INDEXED | 1024 | 3.56 | num_sms | 132 | None | GENERIC (policy infra field) |
| w13 N4096 K4096 | INDEXED | 2048 | 7.11 | block_shape | (16, 512, 64) | (24, 512, 64) | UNATTRIBUTED |
| w13 N4096 K4096 | INDEXED | 2048 | 7.11 | warp_shape | (16, 64, 64) | (24, 64, 64) | UNATTRIBUTED |
| w13 N4096 K4096 | INDEXED | 2048 | 7.11 | num_sms | 132 | None | GENERIC (policy infra field) |
| w13 N4096 K4096 | INDEXED | 4096 | 14.22 | block_shape | (32, 128, 128) | (32, 128, 256) | UNATTRIBUTED |
| w13 N4096 K4096 | INDEXED | 4096 | 14.22 | warp_shape | (32, 32, 64) | (32, 32, 128) | UNATTRIBUTED |
| w13 N4096 K4096 | INDEXED | 4096 | 14.22 | num_sms | 132 | None | GENERIC (policy infra field) |
| w13 N4096 K4096 | INDEXED | 8192 | 28.44 | block_shape | (32, 128, 128) | (48, 128, 128) | UNATTRIBUTED |
| w13 N4096 K4096 | INDEXED | 8192 | 28.44 | warp_shape | (32, 32, 64) | (48, 32, 128) | UNATTRIBUTED |
| w13 N4096 K4096 | INDEXED | 8192 | 28.44 | num_sms | 132 | None | GENERIC (policy infra field) |
| w13 N4096 K4096 | INDEXED | 16384 | 56.89 | num_sms | 132 | None | GENERIC (policy infra field) |
| w13 N4096 K4096 | GROUPED_MASKED | 64 | 0.22 | use_tma | False | True | GENERIC (policy infra field) |
| w13 N4096 K4096 | GROUPED_MASKED | 64 | 0.22 | use_mbarrier | False | True | GENERIC (policy infra field) |
| w13 N4096 K4096 | GROUPED_MASKED | 64 | 0.22 | use_warp_spec | False | True | GENERIC (policy infra field) |
| w13 N4096 K4096 | GROUPED_MASKED | 64 | 0.22 | num_sms | 132 | None | GENERIC (policy infra field) |
| w13 N4096 K4096 | GROUPED_MASKED | 128 | 0.44 | use_tma | False | True | GENERIC (policy infra field) |
| w13 N4096 K4096 | GROUPED_MASKED | 128 | 0.44 | use_mbarrier | False | True | GENERIC (policy infra field) |
| w13 N4096 K4096 | GROUPED_MASKED | 128 | 0.44 | use_warp_spec | False | True | GENERIC (policy infra field) |
| w13 N4096 K4096 | GROUPED_MASKED | 128 | 0.44 | num_sms | 132 | None | GENERIC (policy infra field) |
| w13 N4096 K4096 | GROUPED_MASKED | 256 | 0.89 | use_tma | False | True | GENERIC (policy infra field) |
| w13 N4096 K4096 | GROUPED_MASKED | 256 | 0.89 | use_mbarrier | False | True | GENERIC (policy infra field) |
| w13 N4096 K4096 | GROUPED_MASKED | 256 | 0.89 | use_warp_spec | False | True | GENERIC (policy infra field) |
| w13 N4096 K4096 | GROUPED_MASKED | 256 | 0.89 | num_sms | 132 | None | GENERIC (policy infra field) |
| w13 N4096 K4096 | GROUPED_MASKED | 512 | 1.78 | block_shape | (8, 512, 64) | (16, 512, 64) | UNATTRIBUTED |
| w13 N4096 K4096 | GROUPED_MASKED | 512 | 1.78 | warp_shape | (8, 64, 64) | (16, 64, 64) | UNATTRIBUTED |
| w13 N4096 K4096 | GROUPED_MASKED | 512 | 1.78 | use_tma | False | True | GENERIC (policy infra field) |
| w13 N4096 K4096 | GROUPED_MASKED | 512 | 1.78 | use_mbarrier | False | True | GENERIC (policy infra field) |
| w13 N4096 K4096 | GROUPED_MASKED | 512 | 1.78 | use_warp_spec | False | True | GENERIC (policy infra field) |
| w13 N4096 K4096 | GROUPED_MASKED | 512 | 1.78 | num_sms | 132 | None | GENERIC (policy infra field) |
| w13 N4096 K4096 | GROUPED_MASKED | 1024 | 3.56 | block_shape | (8, 512, 64) | (16, 512, 64) | UNATTRIBUTED |
| w13 N4096 K4096 | GROUPED_MASKED | 1024 | 3.56 | warp_shape | (8, 64, 64) | (16, 64, 64) | UNATTRIBUTED |
| w13 N4096 K4096 | GROUPED_MASKED | 1024 | 3.56 | use_tma | False | True | GENERIC (policy infra field) |
| w13 N4096 K4096 | GROUPED_MASKED | 1024 | 3.56 | use_mbarrier | False | True | GENERIC (policy infra field) |
| w13 N4096 K4096 | GROUPED_MASKED | 1024 | 3.56 | use_warp_spec | False | True | GENERIC (policy infra field) |
| w13 N4096 K4096 | GROUPED_MASKED | 1024 | 3.56 | num_sms | 132 | None | GENERIC (policy infra field) |
| w13 N4096 K4096 | GROUPED_MASKED | 2048 | 7.11 | block_shape | (16, 512, 64) | (24, 512, 64) | UNATTRIBUTED |
| w13 N4096 K4096 | GROUPED_MASKED | 2048 | 7.11 | warp_shape | (16, 64, 64) | (24, 64, 64) | UNATTRIBUTED |
| w13 N4096 K4096 | GROUPED_MASKED | 2048 | 7.11 | use_tma | False | True | GENERIC (policy infra field) |
| w13 N4096 K4096 | GROUPED_MASKED | 2048 | 7.11 | use_mbarrier | False | True | GENERIC (policy infra field) |
| w13 N4096 K4096 | GROUPED_MASKED | 2048 | 7.11 | use_warp_spec | False | True | GENERIC (policy infra field) |
| w13 N4096 K4096 | GROUPED_MASKED | 2048 | 7.11 | num_sms | 132 | None | GENERIC (policy infra field) |
| w13 N4096 K4096 | GROUPED_MASKED | 4096 | 14.22 | block_shape | (32, 128, 128) | (32, 128, 256) | UNATTRIBUTED |
| w13 N4096 K4096 | GROUPED_MASKED | 4096 | 14.22 | warp_shape | (32, 32, 64) | (32, 32, 128) | UNATTRIBUTED |
| w13 N4096 K4096 | GROUPED_MASKED | 4096 | 14.22 | use_tma | False | True | GENERIC (policy infra field) |
| w13 N4096 K4096 | GROUPED_MASKED | 4096 | 14.22 | use_mbarrier | False | True | GENERIC (policy infra field) |
| w13 N4096 K4096 | GROUPED_MASKED | 4096 | 14.22 | use_warp_spec | False | True | GENERIC (policy infra field) |
| w13 N4096 K4096 | GROUPED_MASKED | 4096 | 14.22 | num_sms | 132 | None | GENERIC (policy infra field) |
| w13 N4096 K4096 | GROUPED_MASKED | 8192 | 28.44 | block_shape | (32, 128, 128) | (48, 128, 128) | UNATTRIBUTED |
| w13 N4096 K4096 | GROUPED_MASKED | 8192 | 28.44 | warp_shape | (32, 32, 64) | (48, 32, 128) | UNATTRIBUTED |
| w13 N4096 K4096 | GROUPED_MASKED | 8192 | 28.44 | use_tma | False | True | GENERIC (policy infra field) |
| w13 N4096 K4096 | GROUPED_MASKED | 8192 | 28.44 | use_mbarrier | False | True | GENERIC (policy infra field) |
| w13 N4096 K4096 | GROUPED_MASKED | 8192 | 28.44 | use_warp_spec | False | True | GENERIC (policy infra field) |
| w13 N4096 K4096 | GROUPED_MASKED | 8192 | 28.44 | num_sms | 132 | None | GENERIC (policy infra field) |
| w13 N4096 K4096 | GROUPED_MASKED | 16384 | 56.89 | use_tma | False | True | GENERIC (policy infra field) |
| w13 N4096 K4096 | GROUPED_MASKED | 16384 | 56.89 | use_mbarrier | False | True | GENERIC (policy infra field) |
| w13 N4096 K4096 | GROUPED_MASKED | 16384 | 56.89 | use_warp_spec | False | True | GENERIC (policy infra field) |
| w13 N4096 K4096 | GROUPED_MASKED | 16384 | 56.89 | num_sms | 132 | None | GENERIC (policy infra field) |
| w2 N4096 K2048 | INDEXED | 64 | 0.22 | num_sms | 132 | None | GENERIC (policy infra field) |
| w2 N4096 K2048 | INDEXED | 128 | 0.44 | num_sms | 132 | None | GENERIC (policy infra field) |
| w2 N4096 K2048 | INDEXED | 256 | 0.89 | num_sms | 132 | None | GENERIC (policy infra field) |
| w2 N4096 K2048 | INDEXED | 512 | 1.78 | block_shape | (8, 512, 64) | (16, 512, 64) | UNATTRIBUTED |
| w2 N4096 K2048 | INDEXED | 512 | 1.78 | warp_shape | (8, 64, 64) | (16, 64, 64) | UNATTRIBUTED |
| w2 N4096 K2048 | INDEXED | 512 | 1.78 | num_sms | 132 | None | GENERIC (policy infra field) |
| w2 N4096 K2048 | INDEXED | 1024 | 3.56 | block_shape | (8, 512, 64) | (16, 512, 64) | UNATTRIBUTED |
| w2 N4096 K2048 | INDEXED | 1024 | 3.56 | warp_shape | (8, 64, 64) | (16, 64, 64) | UNATTRIBUTED |
| w2 N4096 K2048 | INDEXED | 1024 | 3.56 | num_sms | 132 | None | GENERIC (policy infra field) |
| w2 N4096 K2048 | INDEXED | 2048 | 7.11 | block_shape | (16, 512, 64) | (24, 512, 64) | UNATTRIBUTED |
| w2 N4096 K2048 | INDEXED | 2048 | 7.11 | warp_shape | (16, 64, 64) | (24, 64, 64) | UNATTRIBUTED |
| w2 N4096 K2048 | INDEXED | 2048 | 7.11 | num_sms | 132 | None | GENERIC (policy infra field) |
| w2 N4096 K2048 | INDEXED | 4096 | 14.22 | block_shape | (32, 128, 128) | (32, 128, 256) | UNATTRIBUTED |
| w2 N4096 K2048 | INDEXED | 4096 | 14.22 | warp_shape | (32, 32, 64) | (32, 32, 128) | UNATTRIBUTED |
| w2 N4096 K2048 | INDEXED | 4096 | 14.22 | num_sms | 132 | None | GENERIC (policy infra field) |
| w2 N4096 K2048 | INDEXED | 8192 | 28.44 | block_shape | (32, 128, 128) | (48, 128, 128) | UNATTRIBUTED |
| w2 N4096 K2048 | INDEXED | 8192 | 28.44 | warp_shape | (32, 32, 64) | (48, 32, 128) | UNATTRIBUTED |
| w2 N4096 K2048 | INDEXED | 8192 | 28.44 | num_sms | 132 | None | GENERIC (policy infra field) |
| w2 N4096 K2048 | INDEXED | 16384 | 56.89 | num_sms | 132 | None | GENERIC (policy infra field) |
| w2 N4096 K2048 | GROUPED_MASKED | 64 | 0.22 | use_tma | False | True | GENERIC (policy infra field) |
| w2 N4096 K2048 | GROUPED_MASKED | 64 | 0.22 | use_mbarrier | False | True | GENERIC (policy infra field) |
| w2 N4096 K2048 | GROUPED_MASKED | 64 | 0.22 | use_warp_spec | False | True | GENERIC (policy infra field) |
| w2 N4096 K2048 | GROUPED_MASKED | 64 | 0.22 | num_sms | 132 | None | GENERIC (policy infra field) |
| w2 N4096 K2048 | GROUPED_MASKED | 128 | 0.44 | use_tma | False | True | GENERIC (policy infra field) |
| w2 N4096 K2048 | GROUPED_MASKED | 128 | 0.44 | use_mbarrier | False | True | GENERIC (policy infra field) |
| w2 N4096 K2048 | GROUPED_MASKED | 128 | 0.44 | use_warp_spec | False | True | GENERIC (policy infra field) |
| w2 N4096 K2048 | GROUPED_MASKED | 128 | 0.44 | num_sms | 132 | None | GENERIC (policy infra field) |
| w2 N4096 K2048 | GROUPED_MASKED | 256 | 0.89 | use_tma | False | True | GENERIC (policy infra field) |
| w2 N4096 K2048 | GROUPED_MASKED | 256 | 0.89 | use_mbarrier | False | True | GENERIC (policy infra field) |
| w2 N4096 K2048 | GROUPED_MASKED | 256 | 0.89 | use_warp_spec | False | True | GENERIC (policy infra field) |
| w2 N4096 K2048 | GROUPED_MASKED | 256 | 0.89 | num_sms | 132 | None | GENERIC (policy infra field) |
| w2 N4096 K2048 | GROUPED_MASKED | 512 | 1.78 | block_shape | (8, 512, 64) | (16, 512, 64) | UNATTRIBUTED |
| w2 N4096 K2048 | GROUPED_MASKED | 512 | 1.78 | warp_shape | (8, 64, 64) | (16, 64, 64) | UNATTRIBUTED |
| w2 N4096 K2048 | GROUPED_MASKED | 512 | 1.78 | use_tma | False | True | GENERIC (policy infra field) |
| w2 N4096 K2048 | GROUPED_MASKED | 512 | 1.78 | use_mbarrier | False | True | GENERIC (policy infra field) |
| w2 N4096 K2048 | GROUPED_MASKED | 512 | 1.78 | use_warp_spec | False | True | GENERIC (policy infra field) |
| w2 N4096 K2048 | GROUPED_MASKED | 512 | 1.78 | num_sms | 132 | None | GENERIC (policy infra field) |
| w2 N4096 K2048 | GROUPED_MASKED | 1024 | 3.56 | block_shape | (8, 512, 64) | (16, 512, 64) | UNATTRIBUTED |
| w2 N4096 K2048 | GROUPED_MASKED | 1024 | 3.56 | warp_shape | (8, 64, 64) | (16, 64, 64) | UNATTRIBUTED |
| w2 N4096 K2048 | GROUPED_MASKED | 1024 | 3.56 | use_tma | False | True | GENERIC (policy infra field) |
| w2 N4096 K2048 | GROUPED_MASKED | 1024 | 3.56 | use_mbarrier | False | True | GENERIC (policy infra field) |
| w2 N4096 K2048 | GROUPED_MASKED | 1024 | 3.56 | use_warp_spec | False | True | GENERIC (policy infra field) |
| w2 N4096 K2048 | GROUPED_MASKED | 1024 | 3.56 | num_sms | 132 | None | GENERIC (policy infra field) |
| w2 N4096 K2048 | GROUPED_MASKED | 2048 | 7.11 | block_shape | (16, 512, 64) | (24, 512, 64) | UNATTRIBUTED |
| w2 N4096 K2048 | GROUPED_MASKED | 2048 | 7.11 | warp_shape | (16, 64, 64) | (24, 64, 64) | UNATTRIBUTED |
| w2 N4096 K2048 | GROUPED_MASKED | 2048 | 7.11 | use_tma | False | True | GENERIC (policy infra field) |
| w2 N4096 K2048 | GROUPED_MASKED | 2048 | 7.11 | use_mbarrier | False | True | GENERIC (policy infra field) |
| w2 N4096 K2048 | GROUPED_MASKED | 2048 | 7.11 | use_warp_spec | False | True | GENERIC (policy infra field) |
| w2 N4096 K2048 | GROUPED_MASKED | 2048 | 7.11 | num_sms | 132 | None | GENERIC (policy infra field) |
| w2 N4096 K2048 | GROUPED_MASKED | 4096 | 14.22 | block_shape | (32, 128, 128) | (32, 128, 256) | UNATTRIBUTED |
| w2 N4096 K2048 | GROUPED_MASKED | 4096 | 14.22 | warp_shape | (32, 32, 64) | (32, 32, 128) | UNATTRIBUTED |
| w2 N4096 K2048 | GROUPED_MASKED | 4096 | 14.22 | use_tma | False | True | GENERIC (policy infra field) |
| w2 N4096 K2048 | GROUPED_MASKED | 4096 | 14.22 | use_mbarrier | False | True | GENERIC (policy infra field) |
| w2 N4096 K2048 | GROUPED_MASKED | 4096 | 14.22 | use_warp_spec | False | True | GENERIC (policy infra field) |
| w2 N4096 K2048 | GROUPED_MASKED | 4096 | 14.22 | num_sms | 132 | None | GENERIC (policy infra field) |
| w2 N4096 K2048 | GROUPED_MASKED | 8192 | 28.44 | block_shape | (32, 128, 128) | (48, 128, 128) | UNATTRIBUTED |
| w2 N4096 K2048 | GROUPED_MASKED | 8192 | 28.44 | warp_shape | (32, 32, 64) | (48, 32, 128) | UNATTRIBUTED |
| w2 N4096 K2048 | GROUPED_MASKED | 8192 | 28.44 | use_tma | False | True | GENERIC (policy infra field) |
| w2 N4096 K2048 | GROUPED_MASKED | 8192 | 28.44 | use_mbarrier | False | True | GENERIC (policy infra field) |
| w2 N4096 K2048 | GROUPED_MASKED | 8192 | 28.44 | use_warp_spec | False | True | GENERIC (policy infra field) |
| w2 N4096 K2048 | GROUPED_MASKED | 8192 | 28.44 | num_sms | 132 | None | GENERIC (policy infra field) |
| w2 N4096 K2048 | GROUPED_MASKED | 16384 | 56.89 | use_tma | False | True | GENERIC (policy infra field) |
| w2 N4096 K2048 | GROUPED_MASKED | 16384 | 56.89 | use_mbarrier | False | True | GENERIC (policy infra field) |
| w2 N4096 K2048 | GROUPED_MASKED | 16384 | 56.89 | use_warp_spec | False | True | GENERIC (policy infra field) |
| w2 N4096 K2048 | GROUPED_MASKED | 16384 | 56.89 | num_sms | 132 | None | GENERIC (policy infra field) |

## Unattributed differences

| shape | gemm | M | field | old | new |
|---|---|---|---|---|---|
| w13 N4096 K4096 | INDEXED | 512 | block_shape | (8, 512, 64) | (16, 512, 64) |
| w13 N4096 K4096 | INDEXED | 512 | warp_shape | (8, 64, 64) | (16, 64, 64) |
| w13 N4096 K4096 | INDEXED | 1024 | block_shape | (8, 512, 64) | (16, 512, 64) |
| w13 N4096 K4096 | INDEXED | 1024 | warp_shape | (8, 64, 64) | (16, 64, 64) |
| w13 N4096 K4096 | INDEXED | 2048 | block_shape | (16, 512, 64) | (24, 512, 64) |
| w13 N4096 K4096 | INDEXED | 2048 | warp_shape | (16, 64, 64) | (24, 64, 64) |
| w13 N4096 K4096 | INDEXED | 4096 | block_shape | (32, 128, 128) | (32, 128, 256) |
| w13 N4096 K4096 | INDEXED | 4096 | warp_shape | (32, 32, 64) | (32, 32, 128) |
| w13 N4096 K4096 | INDEXED | 8192 | block_shape | (32, 128, 128) | (48, 128, 128) |
| w13 N4096 K4096 | INDEXED | 8192 | warp_shape | (32, 32, 64) | (48, 32, 128) |
| w13 N4096 K4096 | GROUPED_MASKED | 512 | block_shape | (8, 512, 64) | (16, 512, 64) |
| w13 N4096 K4096 | GROUPED_MASKED | 512 | warp_shape | (8, 64, 64) | (16, 64, 64) |
| w13 N4096 K4096 | GROUPED_MASKED | 1024 | block_shape | (8, 512, 64) | (16, 512, 64) |
| w13 N4096 K4096 | GROUPED_MASKED | 1024 | warp_shape | (8, 64, 64) | (16, 64, 64) |
| w13 N4096 K4096 | GROUPED_MASKED | 2048 | block_shape | (16, 512, 64) | (24, 512, 64) |
| w13 N4096 K4096 | GROUPED_MASKED | 2048 | warp_shape | (16, 64, 64) | (24, 64, 64) |
| w13 N4096 K4096 | GROUPED_MASKED | 4096 | block_shape | (32, 128, 128) | (32, 128, 256) |
| w13 N4096 K4096 | GROUPED_MASKED | 4096 | warp_shape | (32, 32, 64) | (32, 32, 128) |
| w13 N4096 K4096 | GROUPED_MASKED | 8192 | block_shape | (32, 128, 128) | (48, 128, 128) |
| w13 N4096 K4096 | GROUPED_MASKED | 8192 | warp_shape | (32, 32, 64) | (48, 32, 128) |
| w2 N4096 K2048 | INDEXED | 512 | block_shape | (8, 512, 64) | (16, 512, 64) |
| w2 N4096 K2048 | INDEXED | 512 | warp_shape | (8, 64, 64) | (16, 64, 64) |
| w2 N4096 K2048 | INDEXED | 1024 | block_shape | (8, 512, 64) | (16, 512, 64) |
| w2 N4096 K2048 | INDEXED | 1024 | warp_shape | (8, 64, 64) | (16, 64, 64) |
| w2 N4096 K2048 | INDEXED | 2048 | block_shape | (16, 512, 64) | (24, 512, 64) |
| w2 N4096 K2048 | INDEXED | 2048 | warp_shape | (16, 64, 64) | (24, 64, 64) |
| w2 N4096 K2048 | INDEXED | 4096 | block_shape | (32, 128, 128) | (32, 128, 256) |
| w2 N4096 K2048 | INDEXED | 4096 | warp_shape | (32, 32, 64) | (32, 32, 128) |
| w2 N4096 K2048 | INDEXED | 8192 | block_shape | (32, 128, 128) | (48, 128, 128) |
| w2 N4096 K2048 | INDEXED | 8192 | warp_shape | (32, 32, 64) | (48, 32, 128) |
| w2 N4096 K2048 | GROUPED_MASKED | 512 | block_shape | (8, 512, 64) | (16, 512, 64) |
| w2 N4096 K2048 | GROUPED_MASKED | 512 | warp_shape | (8, 64, 64) | (16, 64, 64) |
| w2 N4096 K2048 | GROUPED_MASKED | 1024 | block_shape | (8, 512, 64) | (16, 512, 64) |
| w2 N4096 K2048 | GROUPED_MASKED | 1024 | warp_shape | (8, 64, 64) | (16, 64, 64) |
| w2 N4096 K2048 | GROUPED_MASKED | 2048 | block_shape | (16, 512, 64) | (24, 512, 64) |
| w2 N4096 K2048 | GROUPED_MASKED | 2048 | warp_shape | (16, 64, 64) | (24, 64, 64) |
| w2 N4096 K2048 | GROUPED_MASKED | 4096 | block_shape | (32, 128, 128) | (32, 128, 256) |
| w2 N4096 K2048 | GROUPED_MASKED | 4096 | warp_shape | (32, 32, 64) | (32, 32, 128) |
| w2 N4096 K2048 | GROUPED_MASKED | 8192 | block_shape | (32, 128, 128) | (48, 128, 128) |
| w2 N4096 K2048 | GROUPED_MASKED | 8192 | warp_shape | (32, 32, 64) | (48, 32, 128) |

## Notes

- K1 = 2-CTA grouped-scale residency cap; K2 = large per_expert_m 256-N tile with stream-K off; K3 = small-M wide tile (block_n 512 / block_k 64 / warp_n 64).
- GENERIC = field produced by the new candidate-policy infrastructure (stage fitting, num_sms targets, TMA flags); not one of the three KEEP calibrations but expected to differ implementation-wise.
- moe_block_size threshold table check: old table ((8, 0.7), (16, 0.8), (32, 0.9), (48, 0.9), (64, 0.9)) vs new `_select_sm90_block_m` (measured-block-min over sampled expert loads).

## Analysis conclusions

Post-migration state (calibrations implemented and enabled at
num_sms >= 128):

1. Small/mid M (64-2048, per_expert_m < 8): both pick the K3 wide
   tile (block_n 512, block_k 64, warp_n 64, ctas 2, stages 3).
   Remaining diff is block_m only (old 8/8/8/8/16 vs new
   8/8/8/16/24): the old moe_block_size threshold table
   (8,0.7)(16,0.8)(32,0.9)(48,0.9)(64,0.9) picks smaller tiles than
   the new measured argmin (_select_sm90_block_m). Accepted as
   generic policy behavior — block_m within one tile step, same
   tile family and residency.
2. Large M (16384, per_expert_m ~57): exact match — (64, 256, 128),
   warp (64, 32, 128), ctas 1, stream-K off (K2).
3. Mid M (4096-8192, per_expert_m 14-28): old (32, 128, 128)
   warp_k 64 vs new (32, 128, 256) / (48, 128, 128). The block_k
   difference at M=4096 comes from the old num_warps==4 K-doubling
   branch (warp_k 64 -> block_k 128) which the candidate ladder
   does not replicate (prefers k256 at block_m<=32). NOT ported:
   the oracle KEEP list excluded it; new tiles are legal and
   resource-equivalent. The block_m 32 vs 48 difference at M=8192
   is the same threshold-table-vs-measured-argmin effect as (1).
4. per_expert_m>=96 block_m 128 branch: fires only at M >= ~28k
   (288 experts), outside the bench sweep; included in K2.
5. num_sms absent from new configs: the grouped-scale policy does
   not emit a num_sms field (unlike the legacy seed path which
   computed a launch-grid target). The kernel runtime derives the
   grid from the config; no H200 calibration depended on the
   emitted value.
6. Gate check: at num_sms=114 (H100 PCIe) all three calibrations
   are disabled and selection matches the pre-migration generic
   behavior exactly.
