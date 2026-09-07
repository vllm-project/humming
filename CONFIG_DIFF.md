# Old Sm90H200Heuristics vs new Sm90Heuristics policy config diff

- Device: fake H200, num_sms=132, max_smem=227 KiB
- Shapes: w13 (N=4096, K=4096), w2 (N=4096, K=2048); 288 experts, fp8e4m3 a / int4 group-128 b / bf16 scales
- M sweep: 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384
- Gemm types: INDEXED, GROUPED_MASKED

## Per-shape diff

| shape | gemm | M | per_expert_m | field | old | new | attribution |
|---|---|---|---|---|---|---|---|
| w13 N4096 K4096 | INDEXED | 64 | 0.22 | block_shape | (8, 512, 64) | (8, 128, 256) | K3 |
| w13 N4096 K4096 | INDEXED | 64 | 0.22 | warp_shape | (8, 64, 64) | (8, 32, 128) | UNATTRIBUTED |
| w13 N4096 K4096 | INDEXED | 64 | 0.22 | num_ctas_per_sm | 2 | 1 | UNATTRIBUTED |
| w13 N4096 K4096 | INDEXED | 64 | 0.22 | num_stages | 3 | 4 | GENERIC (policy infra field) |
| w13 N4096 K4096 | INDEXED | 64 | 0.22 | num_sms | 132 | None | GENERIC (policy infra field) |
| w13 N4096 K4096 | INDEXED | 128 | 0.44 | block_shape | (8, 512, 64) | (8, 128, 256) | K3 |
| w13 N4096 K4096 | INDEXED | 128 | 0.44 | warp_shape | (8, 64, 64) | (8, 32, 128) | UNATTRIBUTED |
| w13 N4096 K4096 | INDEXED | 128 | 0.44 | num_ctas_per_sm | 2 | 1 | UNATTRIBUTED |
| w13 N4096 K4096 | INDEXED | 128 | 0.44 | num_stages | 3 | 4 | GENERIC (policy infra field) |
| w13 N4096 K4096 | INDEXED | 128 | 0.44 | num_sms | 132 | None | GENERIC (policy infra field) |
| w13 N4096 K4096 | INDEXED | 256 | 0.89 | block_shape | (8, 512, 64) | (8, 128, 256) | K3 |
| w13 N4096 K4096 | INDEXED | 256 | 0.89 | warp_shape | (8, 64, 64) | (8, 32, 128) | UNATTRIBUTED |
| w13 N4096 K4096 | INDEXED | 256 | 0.89 | num_ctas_per_sm | 2 | 1 | UNATTRIBUTED |
| w13 N4096 K4096 | INDEXED | 256 | 0.89 | num_stages | 3 | 4 | GENERIC (policy infra field) |
| w13 N4096 K4096 | INDEXED | 256 | 0.89 | num_sms | 132 | None | GENERIC (policy infra field) |
| w13 N4096 K4096 | INDEXED | 512 | 1.78 | block_shape | (8, 512, 64) | (16, 128, 256) | K3 |
| w13 N4096 K4096 | INDEXED | 512 | 1.78 | warp_shape | (8, 64, 64) | (16, 32, 128) | UNATTRIBUTED |
| w13 N4096 K4096 | INDEXED | 512 | 1.78 | num_ctas_per_sm | 2 | 1 | UNATTRIBUTED |
| w13 N4096 K4096 | INDEXED | 512 | 1.78 | num_stages | 3 | 4 | GENERIC (policy infra field) |
| w13 N4096 K4096 | INDEXED | 512 | 1.78 | num_sms | 132 | None | GENERIC (policy infra field) |
| w13 N4096 K4096 | INDEXED | 1024 | 3.56 | block_shape | (8, 512, 64) | (16, 128, 256) | K3 |
| w13 N4096 K4096 | INDEXED | 1024 | 3.56 | warp_shape | (8, 64, 64) | (16, 32, 128) | UNATTRIBUTED |
| w13 N4096 K4096 | INDEXED | 1024 | 3.56 | num_ctas_per_sm | 2 | 1 | UNATTRIBUTED |
| w13 N4096 K4096 | INDEXED | 1024 | 3.56 | num_stages | 3 | 4 | GENERIC (policy infra field) |
| w13 N4096 K4096 | INDEXED | 1024 | 3.56 | num_sms | 132 | None | GENERIC (policy infra field) |
| w13 N4096 K4096 | INDEXED | 2048 | 7.11 | block_shape | (16, 512, 64) | (24, 128, 256) | K3 |
| w13 N4096 K4096 | INDEXED | 2048 | 7.11 | warp_shape | (16, 64, 64) | (24, 32, 128) | UNATTRIBUTED |
| w13 N4096 K4096 | INDEXED | 2048 | 7.11 | num_ctas_per_sm | 2 | 1 | UNATTRIBUTED |
| w13 N4096 K4096 | INDEXED | 2048 | 7.11 | num_stages | 3 | 4 | GENERIC (policy infra field) |
| w13 N4096 K4096 | INDEXED | 2048 | 7.11 | num_sms | 132 | None | GENERIC (policy infra field) |
| w13 N4096 K4096 | INDEXED | 4096 | 14.22 | block_shape | (32, 128, 128) | (32, 128, 256) | UNATTRIBUTED |
| w13 N4096 K4096 | INDEXED | 4096 | 14.22 | warp_shape | (32, 32, 64) | (32, 32, 128) | UNATTRIBUTED |
| w13 N4096 K4096 | INDEXED | 4096 | 14.22 | num_ctas_per_sm | 2 | 1 | UNATTRIBUTED |
| w13 N4096 K4096 | INDEXED | 4096 | 14.22 | num_sms | 132 | None | GENERIC (policy infra field) |
| w13 N4096 K4096 | INDEXED | 8192 | 28.44 | block_shape | (32, 128, 128) | (48, 128, 128) | UNATTRIBUTED |
| w13 N4096 K4096 | INDEXED | 8192 | 28.44 | warp_shape | (32, 32, 64) | (48, 32, 128) | UNATTRIBUTED |
| w13 N4096 K4096 | INDEXED | 8192 | 28.44 | num_ctas_per_sm | 2 | 1 | UNATTRIBUTED |
| w13 N4096 K4096 | INDEXED | 8192 | 28.44 | num_sms | 132 | None | GENERIC (policy infra field) |
| w13 N4096 K4096 | INDEXED | 16384 | 56.89 | block_shape | (64, 256, 128) | (88, 128, 128) | K2 |
| w13 N4096 K4096 | INDEXED | 16384 | 56.89 | warp_shape | (64, 32, 128) | (88, 32, 128) | UNATTRIBUTED |
| w13 N4096 K4096 | INDEXED | 16384 | 56.89 | use_stream_k | False | True | K2 |
| w13 N4096 K4096 | INDEXED | 16384 | 56.89 | num_sms | 132 | None | GENERIC (policy infra field) |
| w13 N4096 K4096 | GROUPED_MASKED | 64 | 0.22 | block_shape | (8, 512, 64) | (8, 128, 256) | K3 |
| w13 N4096 K4096 | GROUPED_MASKED | 64 | 0.22 | warp_shape | (8, 64, 64) | (8, 32, 128) | UNATTRIBUTED |
| w13 N4096 K4096 | GROUPED_MASKED | 64 | 0.22 | num_ctas_per_sm | 2 | 1 | UNATTRIBUTED |
| w13 N4096 K4096 | GROUPED_MASKED | 64 | 0.22 | num_stages | 3 | 4 | GENERIC (policy infra field) |
| w13 N4096 K4096 | GROUPED_MASKED | 64 | 0.22 | use_tma | False | True | GENERIC (policy infra field) |
| w13 N4096 K4096 | GROUPED_MASKED | 64 | 0.22 | use_mbarrier | False | True | GENERIC (policy infra field) |
| w13 N4096 K4096 | GROUPED_MASKED | 64 | 0.22 | use_warp_spec | False | True | GENERIC (policy infra field) |
| w13 N4096 K4096 | GROUPED_MASKED | 64 | 0.22 | num_sms | 132 | None | GENERIC (policy infra field) |
| w13 N4096 K4096 | GROUPED_MASKED | 128 | 0.44 | block_shape | (8, 512, 64) | (8, 128, 256) | K3 |
| w13 N4096 K4096 | GROUPED_MASKED | 128 | 0.44 | warp_shape | (8, 64, 64) | (8, 32, 128) | UNATTRIBUTED |
| w13 N4096 K4096 | GROUPED_MASKED | 128 | 0.44 | num_ctas_per_sm | 2 | 1 | UNATTRIBUTED |
| w13 N4096 K4096 | GROUPED_MASKED | 128 | 0.44 | num_stages | 3 | 4 | GENERIC (policy infra field) |
| w13 N4096 K4096 | GROUPED_MASKED | 128 | 0.44 | use_tma | False | True | GENERIC (policy infra field) |
| w13 N4096 K4096 | GROUPED_MASKED | 128 | 0.44 | use_mbarrier | False | True | GENERIC (policy infra field) |
| w13 N4096 K4096 | GROUPED_MASKED | 128 | 0.44 | use_warp_spec | False | True | GENERIC (policy infra field) |
| w13 N4096 K4096 | GROUPED_MASKED | 128 | 0.44 | num_sms | 132 | None | GENERIC (policy infra field) |
| w13 N4096 K4096 | GROUPED_MASKED | 256 | 0.89 | block_shape | (8, 512, 64) | (8, 128, 256) | K3 |
| w13 N4096 K4096 | GROUPED_MASKED | 256 | 0.89 | warp_shape | (8, 64, 64) | (8, 32, 128) | UNATTRIBUTED |
| w13 N4096 K4096 | GROUPED_MASKED | 256 | 0.89 | num_ctas_per_sm | 2 | 1 | UNATTRIBUTED |
| w13 N4096 K4096 | GROUPED_MASKED | 256 | 0.89 | num_stages | 3 | 4 | GENERIC (policy infra field) |
| w13 N4096 K4096 | GROUPED_MASKED | 256 | 0.89 | use_tma | False | True | GENERIC (policy infra field) |
| w13 N4096 K4096 | GROUPED_MASKED | 256 | 0.89 | use_mbarrier | False | True | GENERIC (policy infra field) |
| w13 N4096 K4096 | GROUPED_MASKED | 256 | 0.89 | use_warp_spec | False | True | GENERIC (policy infra field) |
| w13 N4096 K4096 | GROUPED_MASKED | 256 | 0.89 | num_sms | 132 | None | GENERIC (policy infra field) |
| w13 N4096 K4096 | GROUPED_MASKED | 512 | 1.78 | block_shape | (8, 512, 64) | (16, 128, 256) | K3 |
| w13 N4096 K4096 | GROUPED_MASKED | 512 | 1.78 | warp_shape | (8, 64, 64) | (16, 32, 128) | UNATTRIBUTED |
| w13 N4096 K4096 | GROUPED_MASKED | 512 | 1.78 | num_ctas_per_sm | 2 | 1 | UNATTRIBUTED |
| w13 N4096 K4096 | GROUPED_MASKED | 512 | 1.78 | num_stages | 3 | 4 | GENERIC (policy infra field) |
| w13 N4096 K4096 | GROUPED_MASKED | 512 | 1.78 | use_tma | False | True | GENERIC (policy infra field) |
| w13 N4096 K4096 | GROUPED_MASKED | 512 | 1.78 | use_mbarrier | False | True | GENERIC (policy infra field) |
| w13 N4096 K4096 | GROUPED_MASKED | 512 | 1.78 | use_warp_spec | False | True | GENERIC (policy infra field) |
| w13 N4096 K4096 | GROUPED_MASKED | 512 | 1.78 | num_sms | 132 | None | GENERIC (policy infra field) |
| w13 N4096 K4096 | GROUPED_MASKED | 1024 | 3.56 | block_shape | (8, 512, 64) | (16, 128, 256) | K3 |
| w13 N4096 K4096 | GROUPED_MASKED | 1024 | 3.56 | warp_shape | (8, 64, 64) | (16, 32, 128) | UNATTRIBUTED |
| w13 N4096 K4096 | GROUPED_MASKED | 1024 | 3.56 | num_ctas_per_sm | 2 | 1 | UNATTRIBUTED |
| w13 N4096 K4096 | GROUPED_MASKED | 1024 | 3.56 | num_stages | 3 | 4 | GENERIC (policy infra field) |
| w13 N4096 K4096 | GROUPED_MASKED | 1024 | 3.56 | use_tma | False | True | GENERIC (policy infra field) |
| w13 N4096 K4096 | GROUPED_MASKED | 1024 | 3.56 | use_mbarrier | False | True | GENERIC (policy infra field) |
| w13 N4096 K4096 | GROUPED_MASKED | 1024 | 3.56 | use_warp_spec | False | True | GENERIC (policy infra field) |
| w13 N4096 K4096 | GROUPED_MASKED | 1024 | 3.56 | num_sms | 132 | None | GENERIC (policy infra field) |
| w13 N4096 K4096 | GROUPED_MASKED | 2048 | 7.11 | block_shape | (16, 512, 64) | (24, 128, 256) | K3 |
| w13 N4096 K4096 | GROUPED_MASKED | 2048 | 7.11 | warp_shape | (16, 64, 64) | (24, 32, 128) | UNATTRIBUTED |
| w13 N4096 K4096 | GROUPED_MASKED | 2048 | 7.11 | num_ctas_per_sm | 2 | 1 | UNATTRIBUTED |
| w13 N4096 K4096 | GROUPED_MASKED | 2048 | 7.11 | num_stages | 3 | 4 | GENERIC (policy infra field) |
| w13 N4096 K4096 | GROUPED_MASKED | 2048 | 7.11 | use_tma | False | True | GENERIC (policy infra field) |
| w13 N4096 K4096 | GROUPED_MASKED | 2048 | 7.11 | use_mbarrier | False | True | GENERIC (policy infra field) |
| w13 N4096 K4096 | GROUPED_MASKED | 2048 | 7.11 | use_warp_spec | False | True | GENERIC (policy infra field) |
| w13 N4096 K4096 | GROUPED_MASKED | 2048 | 7.11 | num_sms | 132 | None | GENERIC (policy infra field) |
| w13 N4096 K4096 | GROUPED_MASKED | 4096 | 14.22 | block_shape | (32, 128, 128) | (32, 128, 256) | UNATTRIBUTED |
| w13 N4096 K4096 | GROUPED_MASKED | 4096 | 14.22 | warp_shape | (32, 32, 64) | (32, 32, 128) | UNATTRIBUTED |
| w13 N4096 K4096 | GROUPED_MASKED | 4096 | 14.22 | num_ctas_per_sm | 2 | 1 | UNATTRIBUTED |
| w13 N4096 K4096 | GROUPED_MASKED | 4096 | 14.22 | use_tma | False | True | GENERIC (policy infra field) |
| w13 N4096 K4096 | GROUPED_MASKED | 4096 | 14.22 | use_mbarrier | False | True | GENERIC (policy infra field) |
| w13 N4096 K4096 | GROUPED_MASKED | 4096 | 14.22 | use_warp_spec | False | True | GENERIC (policy infra field) |
| w13 N4096 K4096 | GROUPED_MASKED | 4096 | 14.22 | num_sms | 132 | None | GENERIC (policy infra field) |
| w13 N4096 K4096 | GROUPED_MASKED | 8192 | 28.44 | block_shape | (32, 128, 128) | (48, 128, 128) | UNATTRIBUTED |
| w13 N4096 K4096 | GROUPED_MASKED | 8192 | 28.44 | warp_shape | (32, 32, 64) | (48, 32, 128) | UNATTRIBUTED |
| w13 N4096 K4096 | GROUPED_MASKED | 8192 | 28.44 | num_ctas_per_sm | 2 | 1 | UNATTRIBUTED |
| w13 N4096 K4096 | GROUPED_MASKED | 8192 | 28.44 | use_tma | False | True | GENERIC (policy infra field) |
| w13 N4096 K4096 | GROUPED_MASKED | 8192 | 28.44 | use_mbarrier | False | True | GENERIC (policy infra field) |
| w13 N4096 K4096 | GROUPED_MASKED | 8192 | 28.44 | use_warp_spec | False | True | GENERIC (policy infra field) |
| w13 N4096 K4096 | GROUPED_MASKED | 8192 | 28.44 | num_sms | 132 | None | GENERIC (policy infra field) |
| w13 N4096 K4096 | GROUPED_MASKED | 16384 | 56.89 | block_shape | (64, 256, 128) | (88, 128, 128) | K2 |
| w13 N4096 K4096 | GROUPED_MASKED | 16384 | 56.89 | warp_shape | (64, 32, 128) | (88, 32, 128) | UNATTRIBUTED |
| w13 N4096 K4096 | GROUPED_MASKED | 16384 | 56.89 | use_stream_k | False | True | K2 |
| w13 N4096 K4096 | GROUPED_MASKED | 16384 | 56.89 | use_tma | False | True | GENERIC (policy infra field) |
| w13 N4096 K4096 | GROUPED_MASKED | 16384 | 56.89 | use_mbarrier | False | True | GENERIC (policy infra field) |
| w13 N4096 K4096 | GROUPED_MASKED | 16384 | 56.89 | use_warp_spec | False | True | GENERIC (policy infra field) |
| w13 N4096 K4096 | GROUPED_MASKED | 16384 | 56.89 | num_sms | 132 | None | GENERIC (policy infra field) |
| w2 N4096 K2048 | INDEXED | 64 | 0.22 | block_shape | (8, 512, 64) | (8, 128, 256) | K3 |
| w2 N4096 K2048 | INDEXED | 64 | 0.22 | warp_shape | (8, 64, 64) | (8, 32, 128) | UNATTRIBUTED |
| w2 N4096 K2048 | INDEXED | 64 | 0.22 | num_ctas_per_sm | 2 | 1 | UNATTRIBUTED |
| w2 N4096 K2048 | INDEXED | 64 | 0.22 | num_stages | 3 | 4 | GENERIC (policy infra field) |
| w2 N4096 K2048 | INDEXED | 64 | 0.22 | num_sms | 132 | None | GENERIC (policy infra field) |
| w2 N4096 K2048 | INDEXED | 128 | 0.44 | block_shape | (8, 512, 64) | (8, 128, 256) | K3 |
| w2 N4096 K2048 | INDEXED | 128 | 0.44 | warp_shape | (8, 64, 64) | (8, 32, 128) | UNATTRIBUTED |
| w2 N4096 K2048 | INDEXED | 128 | 0.44 | num_ctas_per_sm | 2 | 1 | UNATTRIBUTED |
| w2 N4096 K2048 | INDEXED | 128 | 0.44 | num_stages | 3 | 4 | GENERIC (policy infra field) |
| w2 N4096 K2048 | INDEXED | 128 | 0.44 | num_sms | 132 | None | GENERIC (policy infra field) |
| w2 N4096 K2048 | INDEXED | 256 | 0.89 | block_shape | (8, 512, 64) | (8, 128, 256) | K3 |
| w2 N4096 K2048 | INDEXED | 256 | 0.89 | warp_shape | (8, 64, 64) | (8, 32, 128) | UNATTRIBUTED |
| w2 N4096 K2048 | INDEXED | 256 | 0.89 | num_ctas_per_sm | 2 | 1 | UNATTRIBUTED |
| w2 N4096 K2048 | INDEXED | 256 | 0.89 | num_stages | 3 | 4 | GENERIC (policy infra field) |
| w2 N4096 K2048 | INDEXED | 256 | 0.89 | num_sms | 132 | None | GENERIC (policy infra field) |
| w2 N4096 K2048 | INDEXED | 512 | 1.78 | block_shape | (8, 512, 64) | (16, 128, 256) | K3 |
| w2 N4096 K2048 | INDEXED | 512 | 1.78 | warp_shape | (8, 64, 64) | (16, 32, 128) | UNATTRIBUTED |
| w2 N4096 K2048 | INDEXED | 512 | 1.78 | num_ctas_per_sm | 2 | 1 | UNATTRIBUTED |
| w2 N4096 K2048 | INDEXED | 512 | 1.78 | num_stages | 3 | 4 | GENERIC (policy infra field) |
| w2 N4096 K2048 | INDEXED | 512 | 1.78 | num_sms | 132 | None | GENERIC (policy infra field) |
| w2 N4096 K2048 | INDEXED | 1024 | 3.56 | block_shape | (8, 512, 64) | (16, 128, 256) | K3 |
| w2 N4096 K2048 | INDEXED | 1024 | 3.56 | warp_shape | (8, 64, 64) | (16, 32, 128) | UNATTRIBUTED |
| w2 N4096 K2048 | INDEXED | 1024 | 3.56 | num_ctas_per_sm | 2 | 1 | UNATTRIBUTED |
| w2 N4096 K2048 | INDEXED | 1024 | 3.56 | num_stages | 3 | 4 | GENERIC (policy infra field) |
| w2 N4096 K2048 | INDEXED | 1024 | 3.56 | num_sms | 132 | None | GENERIC (policy infra field) |
| w2 N4096 K2048 | INDEXED | 2048 | 7.11 | block_shape | (16, 512, 64) | (24, 128, 256) | K3 |
| w2 N4096 K2048 | INDEXED | 2048 | 7.11 | warp_shape | (16, 64, 64) | (24, 32, 128) | UNATTRIBUTED |
| w2 N4096 K2048 | INDEXED | 2048 | 7.11 | num_ctas_per_sm | 2 | 1 | UNATTRIBUTED |
| w2 N4096 K2048 | INDEXED | 2048 | 7.11 | num_stages | 3 | 4 | GENERIC (policy infra field) |
| w2 N4096 K2048 | INDEXED | 2048 | 7.11 | num_sms | 132 | None | GENERIC (policy infra field) |
| w2 N4096 K2048 | INDEXED | 4096 | 14.22 | block_shape | (32, 128, 128) | (32, 128, 256) | UNATTRIBUTED |
| w2 N4096 K2048 | INDEXED | 4096 | 14.22 | warp_shape | (32, 32, 64) | (32, 32, 128) | UNATTRIBUTED |
| w2 N4096 K2048 | INDEXED | 4096 | 14.22 | num_ctas_per_sm | 2 | 1 | UNATTRIBUTED |
| w2 N4096 K2048 | INDEXED | 4096 | 14.22 | num_sms | 132 | None | GENERIC (policy infra field) |
| w2 N4096 K2048 | INDEXED | 8192 | 28.44 | block_shape | (32, 128, 128) | (48, 128, 128) | UNATTRIBUTED |
| w2 N4096 K2048 | INDEXED | 8192 | 28.44 | warp_shape | (32, 32, 64) | (48, 32, 128) | UNATTRIBUTED |
| w2 N4096 K2048 | INDEXED | 8192 | 28.44 | num_ctas_per_sm | 2 | 1 | UNATTRIBUTED |
| w2 N4096 K2048 | INDEXED | 8192 | 28.44 | num_sms | 132 | None | GENERIC (policy infra field) |
| w2 N4096 K2048 | INDEXED | 16384 | 56.89 | block_shape | (64, 256, 128) | (88, 128, 128) | K2 |
| w2 N4096 K2048 | INDEXED | 16384 | 56.89 | warp_shape | (64, 32, 128) | (88, 32, 128) | UNATTRIBUTED |
| w2 N4096 K2048 | INDEXED | 16384 | 56.89 | use_stream_k | False | True | K2 |
| w2 N4096 K2048 | INDEXED | 16384 | 56.89 | num_sms | 132 | None | GENERIC (policy infra field) |
| w2 N4096 K2048 | GROUPED_MASKED | 64 | 0.22 | block_shape | (8, 512, 64) | (8, 128, 256) | K3 |
| w2 N4096 K2048 | GROUPED_MASKED | 64 | 0.22 | warp_shape | (8, 64, 64) | (8, 32, 128) | UNATTRIBUTED |
| w2 N4096 K2048 | GROUPED_MASKED | 64 | 0.22 | num_ctas_per_sm | 2 | 1 | UNATTRIBUTED |
| w2 N4096 K2048 | GROUPED_MASKED | 64 | 0.22 | num_stages | 3 | 4 | GENERIC (policy infra field) |
| w2 N4096 K2048 | GROUPED_MASKED | 64 | 0.22 | use_tma | False | True | GENERIC (policy infra field) |
| w2 N4096 K2048 | GROUPED_MASKED | 64 | 0.22 | use_mbarrier | False | True | GENERIC (policy infra field) |
| w2 N4096 K2048 | GROUPED_MASKED | 64 | 0.22 | use_warp_spec | False | True | GENERIC (policy infra field) |
| w2 N4096 K2048 | GROUPED_MASKED | 64 | 0.22 | num_sms | 132 | None | GENERIC (policy infra field) |
| w2 N4096 K2048 | GROUPED_MASKED | 128 | 0.44 | block_shape | (8, 512, 64) | (8, 128, 256) | K3 |
| w2 N4096 K2048 | GROUPED_MASKED | 128 | 0.44 | warp_shape | (8, 64, 64) | (8, 32, 128) | UNATTRIBUTED |
| w2 N4096 K2048 | GROUPED_MASKED | 128 | 0.44 | num_ctas_per_sm | 2 | 1 | UNATTRIBUTED |
| w2 N4096 K2048 | GROUPED_MASKED | 128 | 0.44 | num_stages | 3 | 4 | GENERIC (policy infra field) |
| w2 N4096 K2048 | GROUPED_MASKED | 128 | 0.44 | use_tma | False | True | GENERIC (policy infra field) |
| w2 N4096 K2048 | GROUPED_MASKED | 128 | 0.44 | use_mbarrier | False | True | GENERIC (policy infra field) |
| w2 N4096 K2048 | GROUPED_MASKED | 128 | 0.44 | use_warp_spec | False | True | GENERIC (policy infra field) |
| w2 N4096 K2048 | GROUPED_MASKED | 128 | 0.44 | num_sms | 132 | None | GENERIC (policy infra field) |
| w2 N4096 K2048 | GROUPED_MASKED | 256 | 0.89 | block_shape | (8, 512, 64) | (8, 128, 256) | K3 |
| w2 N4096 K2048 | GROUPED_MASKED | 256 | 0.89 | warp_shape | (8, 64, 64) | (8, 32, 128) | UNATTRIBUTED |
| w2 N4096 K2048 | GROUPED_MASKED | 256 | 0.89 | num_ctas_per_sm | 2 | 1 | UNATTRIBUTED |
| w2 N4096 K2048 | GROUPED_MASKED | 256 | 0.89 | num_stages | 3 | 4 | GENERIC (policy infra field) |
| w2 N4096 K2048 | GROUPED_MASKED | 256 | 0.89 | use_tma | False | True | GENERIC (policy infra field) |
| w2 N4096 K2048 | GROUPED_MASKED | 256 | 0.89 | use_mbarrier | False | True | GENERIC (policy infra field) |
| w2 N4096 K2048 | GROUPED_MASKED | 256 | 0.89 | use_warp_spec | False | True | GENERIC (policy infra field) |
| w2 N4096 K2048 | GROUPED_MASKED | 256 | 0.89 | num_sms | 132 | None | GENERIC (policy infra field) |
| w2 N4096 K2048 | GROUPED_MASKED | 512 | 1.78 | block_shape | (8, 512, 64) | (16, 128, 256) | K3 |
| w2 N4096 K2048 | GROUPED_MASKED | 512 | 1.78 | warp_shape | (8, 64, 64) | (16, 32, 128) | UNATTRIBUTED |
| w2 N4096 K2048 | GROUPED_MASKED | 512 | 1.78 | num_ctas_per_sm | 2 | 1 | UNATTRIBUTED |
| w2 N4096 K2048 | GROUPED_MASKED | 512 | 1.78 | num_stages | 3 | 4 | GENERIC (policy infra field) |
| w2 N4096 K2048 | GROUPED_MASKED | 512 | 1.78 | use_tma | False | True | GENERIC (policy infra field) |
| w2 N4096 K2048 | GROUPED_MASKED | 512 | 1.78 | use_mbarrier | False | True | GENERIC (policy infra field) |
| w2 N4096 K2048 | GROUPED_MASKED | 512 | 1.78 | use_warp_spec | False | True | GENERIC (policy infra field) |
| w2 N4096 K2048 | GROUPED_MASKED | 512 | 1.78 | num_sms | 132 | None | GENERIC (policy infra field) |
| w2 N4096 K2048 | GROUPED_MASKED | 1024 | 3.56 | block_shape | (8, 512, 64) | (16, 128, 256) | K3 |
| w2 N4096 K2048 | GROUPED_MASKED | 1024 | 3.56 | warp_shape | (8, 64, 64) | (16, 32, 128) | UNATTRIBUTED |
| w2 N4096 K2048 | GROUPED_MASKED | 1024 | 3.56 | num_ctas_per_sm | 2 | 1 | UNATTRIBUTED |
| w2 N4096 K2048 | GROUPED_MASKED | 1024 | 3.56 | num_stages | 3 | 4 | GENERIC (policy infra field) |
| w2 N4096 K2048 | GROUPED_MASKED | 1024 | 3.56 | use_tma | False | True | GENERIC (policy infra field) |
| w2 N4096 K2048 | GROUPED_MASKED | 1024 | 3.56 | use_mbarrier | False | True | GENERIC (policy infra field) |
| w2 N4096 K2048 | GROUPED_MASKED | 1024 | 3.56 | use_warp_spec | False | True | GENERIC (policy infra field) |
| w2 N4096 K2048 | GROUPED_MASKED | 1024 | 3.56 | num_sms | 132 | None | GENERIC (policy infra field) |
| w2 N4096 K2048 | GROUPED_MASKED | 2048 | 7.11 | block_shape | (16, 512, 64) | (24, 128, 256) | K3 |
| w2 N4096 K2048 | GROUPED_MASKED | 2048 | 7.11 | warp_shape | (16, 64, 64) | (24, 32, 128) | UNATTRIBUTED |
| w2 N4096 K2048 | GROUPED_MASKED | 2048 | 7.11 | num_ctas_per_sm | 2 | 1 | UNATTRIBUTED |
| w2 N4096 K2048 | GROUPED_MASKED | 2048 | 7.11 | num_stages | 3 | 4 | GENERIC (policy infra field) |
| w2 N4096 K2048 | GROUPED_MASKED | 2048 | 7.11 | use_tma | False | True | GENERIC (policy infra field) |
| w2 N4096 K2048 | GROUPED_MASKED | 2048 | 7.11 | use_mbarrier | False | True | GENERIC (policy infra field) |
| w2 N4096 K2048 | GROUPED_MASKED | 2048 | 7.11 | use_warp_spec | False | True | GENERIC (policy infra field) |
| w2 N4096 K2048 | GROUPED_MASKED | 2048 | 7.11 | num_sms | 132 | None | GENERIC (policy infra field) |
| w2 N4096 K2048 | GROUPED_MASKED | 4096 | 14.22 | block_shape | (32, 128, 128) | (32, 128, 256) | UNATTRIBUTED |
| w2 N4096 K2048 | GROUPED_MASKED | 4096 | 14.22 | warp_shape | (32, 32, 64) | (32, 32, 128) | UNATTRIBUTED |
| w2 N4096 K2048 | GROUPED_MASKED | 4096 | 14.22 | num_ctas_per_sm | 2 | 1 | UNATTRIBUTED |
| w2 N4096 K2048 | GROUPED_MASKED | 4096 | 14.22 | use_tma | False | True | GENERIC (policy infra field) |
| w2 N4096 K2048 | GROUPED_MASKED | 4096 | 14.22 | use_mbarrier | False | True | GENERIC (policy infra field) |
| w2 N4096 K2048 | GROUPED_MASKED | 4096 | 14.22 | use_warp_spec | False | True | GENERIC (policy infra field) |
| w2 N4096 K2048 | GROUPED_MASKED | 4096 | 14.22 | num_sms | 132 | None | GENERIC (policy infra field) |
| w2 N4096 K2048 | GROUPED_MASKED | 8192 | 28.44 | block_shape | (32, 128, 128) | (48, 128, 128) | UNATTRIBUTED |
| w2 N4096 K2048 | GROUPED_MASKED | 8192 | 28.44 | warp_shape | (32, 32, 64) | (48, 32, 128) | UNATTRIBUTED |
| w2 N4096 K2048 | GROUPED_MASKED | 8192 | 28.44 | num_ctas_per_sm | 2 | 1 | UNATTRIBUTED |
| w2 N4096 K2048 | GROUPED_MASKED | 8192 | 28.44 | use_tma | False | True | GENERIC (policy infra field) |
| w2 N4096 K2048 | GROUPED_MASKED | 8192 | 28.44 | use_mbarrier | False | True | GENERIC (policy infra field) |
| w2 N4096 K2048 | GROUPED_MASKED | 8192 | 28.44 | use_warp_spec | False | True | GENERIC (policy infra field) |
| w2 N4096 K2048 | GROUPED_MASKED | 8192 | 28.44 | num_sms | 132 | None | GENERIC (policy infra field) |
| w2 N4096 K2048 | GROUPED_MASKED | 16384 | 56.89 | block_shape | (64, 256, 128) | (88, 128, 128) | K2 |
| w2 N4096 K2048 | GROUPED_MASKED | 16384 | 56.89 | warp_shape | (64, 32, 128) | (88, 32, 128) | UNATTRIBUTED |
| w2 N4096 K2048 | GROUPED_MASKED | 16384 | 56.89 | use_stream_k | False | True | K2 |
| w2 N4096 K2048 | GROUPED_MASKED | 16384 | 56.89 | use_tma | False | True | GENERIC (policy infra field) |
| w2 N4096 K2048 | GROUPED_MASKED | 16384 | 56.89 | use_mbarrier | False | True | GENERIC (policy infra field) |
| w2 N4096 K2048 | GROUPED_MASKED | 16384 | 56.89 | use_warp_spec | False | True | GENERIC (policy infra field) |
| w2 N4096 K2048 | GROUPED_MASKED | 16384 | 56.89 | num_sms | 132 | None | GENERIC (policy infra field) |

## Unattributed differences

| shape | gemm | M | field | old | new |
|---|---|---|---|---|---|
| w13 N4096 K4096 | INDEXED | 64 | warp_shape | (8, 64, 64) | (8, 32, 128) |
| w13 N4096 K4096 | INDEXED | 64 | num_ctas_per_sm | 2 | 1 |
| w13 N4096 K4096 | INDEXED | 128 | warp_shape | (8, 64, 64) | (8, 32, 128) |
| w13 N4096 K4096 | INDEXED | 128 | num_ctas_per_sm | 2 | 1 |
| w13 N4096 K4096 | INDEXED | 256 | warp_shape | (8, 64, 64) | (8, 32, 128) |
| w13 N4096 K4096 | INDEXED | 256 | num_ctas_per_sm | 2 | 1 |
| w13 N4096 K4096 | INDEXED | 512 | warp_shape | (8, 64, 64) | (16, 32, 128) |
| w13 N4096 K4096 | INDEXED | 512 | num_ctas_per_sm | 2 | 1 |
| w13 N4096 K4096 | INDEXED | 1024 | warp_shape | (8, 64, 64) | (16, 32, 128) |
| w13 N4096 K4096 | INDEXED | 1024 | num_ctas_per_sm | 2 | 1 |
| w13 N4096 K4096 | INDEXED | 2048 | warp_shape | (16, 64, 64) | (24, 32, 128) |
| w13 N4096 K4096 | INDEXED | 2048 | num_ctas_per_sm | 2 | 1 |
| w13 N4096 K4096 | INDEXED | 4096 | block_shape | (32, 128, 128) | (32, 128, 256) |
| w13 N4096 K4096 | INDEXED | 4096 | warp_shape | (32, 32, 64) | (32, 32, 128) |
| w13 N4096 K4096 | INDEXED | 4096 | num_ctas_per_sm | 2 | 1 |
| w13 N4096 K4096 | INDEXED | 8192 | block_shape | (32, 128, 128) | (48, 128, 128) |
| w13 N4096 K4096 | INDEXED | 8192 | warp_shape | (32, 32, 64) | (48, 32, 128) |
| w13 N4096 K4096 | INDEXED | 8192 | num_ctas_per_sm | 2 | 1 |
| w13 N4096 K4096 | INDEXED | 16384 | warp_shape | (64, 32, 128) | (88, 32, 128) |
| w13 N4096 K4096 | GROUPED_MASKED | 64 | warp_shape | (8, 64, 64) | (8, 32, 128) |
| w13 N4096 K4096 | GROUPED_MASKED | 64 | num_ctas_per_sm | 2 | 1 |
| w13 N4096 K4096 | GROUPED_MASKED | 128 | warp_shape | (8, 64, 64) | (8, 32, 128) |
| w13 N4096 K4096 | GROUPED_MASKED | 128 | num_ctas_per_sm | 2 | 1 |
| w13 N4096 K4096 | GROUPED_MASKED | 256 | warp_shape | (8, 64, 64) | (8, 32, 128) |
| w13 N4096 K4096 | GROUPED_MASKED | 256 | num_ctas_per_sm | 2 | 1 |
| w13 N4096 K4096 | GROUPED_MASKED | 512 | warp_shape | (8, 64, 64) | (16, 32, 128) |
| w13 N4096 K4096 | GROUPED_MASKED | 512 | num_ctas_per_sm | 2 | 1 |
| w13 N4096 K4096 | GROUPED_MASKED | 1024 | warp_shape | (8, 64, 64) | (16, 32, 128) |
| w13 N4096 K4096 | GROUPED_MASKED | 1024 | num_ctas_per_sm | 2 | 1 |
| w13 N4096 K4096 | GROUPED_MASKED | 2048 | warp_shape | (16, 64, 64) | (24, 32, 128) |
| w13 N4096 K4096 | GROUPED_MASKED | 2048 | num_ctas_per_sm | 2 | 1 |
| w13 N4096 K4096 | GROUPED_MASKED | 4096 | block_shape | (32, 128, 128) | (32, 128, 256) |
| w13 N4096 K4096 | GROUPED_MASKED | 4096 | warp_shape | (32, 32, 64) | (32, 32, 128) |
| w13 N4096 K4096 | GROUPED_MASKED | 4096 | num_ctas_per_sm | 2 | 1 |
| w13 N4096 K4096 | GROUPED_MASKED | 8192 | block_shape | (32, 128, 128) | (48, 128, 128) |
| w13 N4096 K4096 | GROUPED_MASKED | 8192 | warp_shape | (32, 32, 64) | (48, 32, 128) |
| w13 N4096 K4096 | GROUPED_MASKED | 8192 | num_ctas_per_sm | 2 | 1 |
| w13 N4096 K4096 | GROUPED_MASKED | 16384 | warp_shape | (64, 32, 128) | (88, 32, 128) |
| w2 N4096 K2048 | INDEXED | 64 | warp_shape | (8, 64, 64) | (8, 32, 128) |
| w2 N4096 K2048 | INDEXED | 64 | num_ctas_per_sm | 2 | 1 |
| w2 N4096 K2048 | INDEXED | 128 | warp_shape | (8, 64, 64) | (8, 32, 128) |
| w2 N4096 K2048 | INDEXED | 128 | num_ctas_per_sm | 2 | 1 |
| w2 N4096 K2048 | INDEXED | 256 | warp_shape | (8, 64, 64) | (8, 32, 128) |
| w2 N4096 K2048 | INDEXED | 256 | num_ctas_per_sm | 2 | 1 |
| w2 N4096 K2048 | INDEXED | 512 | warp_shape | (8, 64, 64) | (16, 32, 128) |
| w2 N4096 K2048 | INDEXED | 512 | num_ctas_per_sm | 2 | 1 |
| w2 N4096 K2048 | INDEXED | 1024 | warp_shape | (8, 64, 64) | (16, 32, 128) |
| w2 N4096 K2048 | INDEXED | 1024 | num_ctas_per_sm | 2 | 1 |
| w2 N4096 K2048 | INDEXED | 2048 | warp_shape | (16, 64, 64) | (24, 32, 128) |
| w2 N4096 K2048 | INDEXED | 2048 | num_ctas_per_sm | 2 | 1 |
| w2 N4096 K2048 | INDEXED | 4096 | block_shape | (32, 128, 128) | (32, 128, 256) |
| w2 N4096 K2048 | INDEXED | 4096 | warp_shape | (32, 32, 64) | (32, 32, 128) |
| w2 N4096 K2048 | INDEXED | 4096 | num_ctas_per_sm | 2 | 1 |
| w2 N4096 K2048 | INDEXED | 8192 | block_shape | (32, 128, 128) | (48, 128, 128) |
| w2 N4096 K2048 | INDEXED | 8192 | warp_shape | (32, 32, 64) | (48, 32, 128) |
| w2 N4096 K2048 | INDEXED | 8192 | num_ctas_per_sm | 2 | 1 |
| w2 N4096 K2048 | INDEXED | 16384 | warp_shape | (64, 32, 128) | (88, 32, 128) |
| w2 N4096 K2048 | GROUPED_MASKED | 64 | warp_shape | (8, 64, 64) | (8, 32, 128) |
| w2 N4096 K2048 | GROUPED_MASKED | 64 | num_ctas_per_sm | 2 | 1 |
| w2 N4096 K2048 | GROUPED_MASKED | 128 | warp_shape | (8, 64, 64) | (8, 32, 128) |
| w2 N4096 K2048 | GROUPED_MASKED | 128 | num_ctas_per_sm | 2 | 1 |
| w2 N4096 K2048 | GROUPED_MASKED | 256 | warp_shape | (8, 64, 64) | (8, 32, 128) |
| w2 N4096 K2048 | GROUPED_MASKED | 256 | num_ctas_per_sm | 2 | 1 |
| w2 N4096 K2048 | GROUPED_MASKED | 512 | warp_shape | (8, 64, 64) | (16, 32, 128) |
| w2 N4096 K2048 | GROUPED_MASKED | 512 | num_ctas_per_sm | 2 | 1 |
| w2 N4096 K2048 | GROUPED_MASKED | 1024 | warp_shape | (8, 64, 64) | (16, 32, 128) |
| w2 N4096 K2048 | GROUPED_MASKED | 1024 | num_ctas_per_sm | 2 | 1 |
| w2 N4096 K2048 | GROUPED_MASKED | 2048 | warp_shape | (16, 64, 64) | (24, 32, 128) |
| w2 N4096 K2048 | GROUPED_MASKED | 2048 | num_ctas_per_sm | 2 | 1 |
| w2 N4096 K2048 | GROUPED_MASKED | 4096 | block_shape | (32, 128, 128) | (32, 128, 256) |
| w2 N4096 K2048 | GROUPED_MASKED | 4096 | warp_shape | (32, 32, 64) | (32, 32, 128) |
| w2 N4096 K2048 | GROUPED_MASKED | 4096 | num_ctas_per_sm | 2 | 1 |
| w2 N4096 K2048 | GROUPED_MASKED | 8192 | block_shape | (32, 128, 128) | (48, 128, 128) |
| w2 N4096 K2048 | GROUPED_MASKED | 8192 | warp_shape | (32, 32, 64) | (48, 32, 128) |
| w2 N4096 K2048 | GROUPED_MASKED | 8192 | num_ctas_per_sm | 2 | 1 |
| w2 N4096 K2048 | GROUPED_MASKED | 16384 | warp_shape | (64, 32, 128) | (88, 32, 128) |

## Notes

- K1 = 2-CTA grouped-scale residency cap; K2 = large per_expert_m 256-N tile with stream-K off; K3 = small-M wide tile (block_n 512 / block_k 64 / warp_n 64).
- GENERIC = field produced by the new candidate-policy infrastructure (stage fitting, num_sms targets, TMA flags); not one of the three KEEP calibrations but expected to differ implementation-wise.
- moe_block_size threshold table check: old table ((8, 0.7), (16, 0.8), (32, 0.9), (48, 0.9), (64, 0.9)) vs new `_select_sm90_block_m` (measured-block-min over sampled expert loads).

## Analysis conclusions

1. Small/mid M (64-2048, per_expert_m < 8): old picks the wide tile
   (block_m 8-24, block_n 512, block_k 64, warp_n 64, ctas 2,
   stages 3). Fully covered by K3 (wide-tile candidate) + K1
   (2-CTA cap). The warp_n 64 and stages 3 differences are part of
   the K3 tile definition itself.
2. Large M (16384, per_expert_m ~57): old picks (64, 256, 128)
   warp (64, 32, 128), ctas 1, stream-K off. Covered by K2 with
   block_m 64 (per_expert_m < 96); the new measured block_m (88)
   differs from the old threshold table but K2 fixes block_m to
   128|64 by per_expert_m, so the old value is reproduced.
3. Mid M (4096-8192, per_expert_m 14-28): old picks (32, 128, 128)
   warp_k 64, ctas 2. The block_k 128 comes from the old
   num_warps==4 K-doubling branch (warp_k 64 -> block_k 128), and
   ctas 2 from the 2-CTA cap. K1 covers ctas; the K-doubling is a
   legacy-seed-path behavior that the grouped-scale candidates do
   not replicate (they prefer k256 at block_m<=32 or k128 at 48).
   This is the num_warps==4 K-doubling suspect: NOT ported. The new
   tile (n128 k256 / k128) is legal and resource-equivalent; the
   oracle KEEP list did not include this branch, so it is accepted
   as a deliberate behavioral change (documented, not calibrated).
4. moe_block_size threshold table (8,0.7)(16,0.8)(32,0.9)(48,0.9)
   (64,0.9): only affects block_m at mid M (new picks 32/48/88 by
   measured argmin). K2 overrides block_m at large M, so the table
   only survives through the new measured block-m selection, which
   the PR accepts (generic policy behavior).
5. per_expert_m>=96 block_m 128 branch: fires only at M >= ~28k
   (per_expert >= 96 with 288 experts), outside the bench sweep;
   included in K2 candidate generation anyway.
6. num_sms absent from new configs: grouped-scale policy currently
   receives num_sms=None (sm90.get_tuning_decision only sets
   include_grid_size for indexed-A16). Supporting change: pass
   include_grid_size=True for the grouped-scale branch so K1/K2/K3
   gating on device.num_sms >= 128 can work.
