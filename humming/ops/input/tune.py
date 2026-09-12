"""Hardware-derived launch planning for process-input kernels."""

import dataclasses
import functools
import math
from types import SimpleNamespace

from humming import dtypes
from humming.config import ActivationType, InputQuantizationMode, ProcessInputConfig, ProcessInputTuningConfig
from humming.device import DeviceInfo, get_device_index
from humming.utils.math import ceil_div, positive_divisors, powers_of_two_up_to


def _finalize_rows(rows: int) -> int:
    return min(8, 1 << (max(rows, 1).bit_length() - 1))


def _valid_values(problem, values: int, block_size: int) -> bool:
    if problem.should_quantize and values < 2:
        return False
    if problem.hidden_size % values or problem.tile_size % values or block_size % values:
        return False
    return not problem.should_quantize or values * problem.target_bits % 8 == 0


def _fits_shared_memory(problem, device, block_size: int, threads: int, values: int, tokens: int) -> bool:
    has_hadamard = problem.hadamard_block_size > 1
    transform_lanes = block_size // values
    uses_shared_transform = has_hadamard and transform_lanes > 32
    num_transform_values = 0
    if uses_shared_transform:
        num_transform_values = threads * values * tokens
    num_warps = threads * tokens // 32
    num_reduction_values = 0
    dynamic_scale_mode = problem.quant_mode.dynamic_scale_mode
    if dynamic_scale_mode is not None:
        scale_size = problem.quant_group_size
        if dynamic_scale_mode == "token":
            scale_size = problem.hidden_size
        num_group_reduction_values = num_warps if scale_size // values > 32 else 0
        num_token_reduction_values = num_warps if dynamic_scale_mode == "group_token" and threads > 32 else 0
        num_reduction_values = max(num_group_reduction_values, num_token_reduction_values)
    required_smem_size = 4 * max(1, num_transform_values + num_reduction_values)
    # Pre-SM100 ptxas limits statically declared shared storage to the default
    # per-block capacity; its larger opt-in capacity applies to dynamic shared.
    limit = device.max_smem_size if device.sm_major >= 10 else device.default_smem_size
    return required_smem_size <= limit


def _token_tuning_candidates(problem, device, block_size: int):
    dynamic_scale_mode = problem.quant_mode.dynamic_scale_mode
    for values in powers_of_two_up_to(min(problem.hidden_size, problem.tile_size, block_size)):
        if not _valid_values(problem, values, block_size):
            continue
        lanes = problem.hidden_size // values
        unit = max(32, block_size // values)
        threads = ceil_div(lanes, unit) * unit
        if not 32 <= threads <= 1024:
            continue
        if dynamic_scale_mode == "group_token" and (values > 32 or threads & (threads - 1)):
            continue
        tokens = 1
        while tokens <= min(problem.num_work_rows, 1024 // threads, 16):
            transform_lanes = block_size // values
            cta_threads = threads * tokens
            fitted = _fits_shared_memory(problem, device, block_size, threads, values, tokens)
            if (transform_lanes <= 32 or cta_threads // transform_lanes <= 16) and fitted:
                two_stage = dynamic_scale_mode == "token" and values > 32
                yield ProcessInputTuningConfig(
                    threads_per_task=threads,
                    values_per_thread=values,
                    tokens_per_block=tokens,
                    two_stage=two_stage,
                    finalize_tokens_per_block=_finalize_rows(problem.num_work_rows),
                )
            tokens *= 2


def _tile_tuning_candidates(problem, device, block_size: int):
    num_tiles = problem.num_tiles
    first_tile_count = max(1, block_size // problem.tile_size)
    divisors = [count for count in positive_divisors(num_tiles) if count >= first_tile_count]
    pure_hadamard = not problem.should_quantize and problem.activation_type == ActivationType.None_
    pure_hadamard &= problem.hadamard_block_size > 1
    if pure_hadamard:
        powers = [count for count in powers_of_two_up_to(num_tiles) if count >= first_tile_count]
        tile_counts = [count for count in sorted(set(powers + divisors)) if count <= 16]
    else:
        powers = [count for count in powers_of_two_up_to(num_tiles) if count >= first_tile_count]
        tile_counts = sorted(set(powers + divisors))

    for tiles_per_block in tile_counts:
        columns = tiles_per_block * problem.tile_size
        for values in powers_of_two_up_to(min(problem.tile_size, block_size)):
            if not _valid_values(problem, values, block_size):
                continue
            tile_lanes = problem.tile_size // values
            transform_lanes = block_size // values
            unit = max(32, tile_lanes, transform_lanes)
            threads = ceil_div(columns // values, unit) * unit
            if threads * values % block_size or threads * values % problem.tile_size:
                continue
            if threads % 32 or not 32 <= threads <= 1024:
                continue
            if transform_lanes > 32 and threads // transform_lanes > 16:
                continue
            if problem.quant_mode.dynamic_scale_mode is not None:
                if tile_lanes > 32 and threads // tile_lanes > 16:
                    continue
            if not _fits_shared_memory(problem, device, block_size, threads, values, 1):
                continue
            yield ProcessInputTuningConfig(
                threads_per_task=threads,
                values_per_thread=values,
                use_tile_partition=True,
                finalize_tokens_per_block=_finalize_rows(problem.num_work_rows),
            )


def _token_score(problem, device, tuning_config: ProcessInputTuningConfig):
    blocks = ceil_div(problem.num_work_rows, tuning_config.tokens_per_block)
    resident = min(16, device.max_threads_per_sm // tuning_config.threads)
    wave_blocks = device.sm_count * resident
    full_waves, remaining = divmod(blocks, wave_blocks)
    remaining_slots = ceil_div(remaining, device.sm_count) if remaining else 0
    slots = full_waves * resident + remaining_slots

    transform_stages = problem.hadamard_block_size.bit_length() - 1
    identity = transform_stages == 0 and device.sm_major >= 10
    setup_units = 13 if identity else 6
    thread_work = tuning_config.threads * tuning_config.values_per_thread * (6 + transform_stages)
    work = slots * (thread_work + problem.hidden_size * setup_units)
    if tuning_config.values_per_thread > 16:
        vector_penalty = tuning_config.values_per_thread if identity else tuning_config.values_per_thread + 48
        vector_divisor = 16 if identity else 64
        work = work * vector_penalty // vector_divisor

    token_reduction = problem.quant_mode.dynamic_scale_mode == "token"
    token_reduction &= problem.activation_type == ActivationType.None_
    multi_warp = problem.hidden_size // tuning_config.values_per_thread > 32
    prefer_single_token = identity and token_reduction and multi_warp
    work *= 2 if prefer_single_token and tuning_config.tokens_per_block > 1 else 1
    if tuning_config.two_stage:
        work *= 2
    if problem.hadamard_block_size > 1:
        return work, blocks, abs(tuning_config.values_per_thread - 8), abs(tuning_config.threads - 512)
    if prefer_single_token:
        return work, abs(tuning_config.values_per_thread.bit_length() - 5), abs(tuning_config.threads - 512)
    return (
        work,
        -tuning_config.tokens_per_block,
        abs(tuning_config.threads - 512),
        tuning_config.values_per_thread,
    )


def _direct_group_score(problem, device, tuning_config: ProcessInputTuningConfig):
    vector_floor = 8 if problem.target_bits == 4 else 4
    large_grid = problem.num_work_rows >= 2 * device.sm_count
    full_row_friendly = device.sm_major >= 10 and large_grid
    full_row_friendly &= problem.activation_type == ActivationType.None_
    thread_limit = device.max_threads_per_block if full_row_friendly else 256
    if tuning_config.threads > thread_limit or tuning_config.values_per_thread < vector_floor:
        return (2,)

    blocks = problem.num_work_rows * ceil_div(problem.hidden_size, tuning_config.columns_per_task)
    useful = min(problem.hidden_size, tuning_config.columns_per_task)
    if full_row_friendly:
        idle = tuning_config.threads * tuning_config.values_per_thread - useful
        value_error = abs(tuning_config.values_per_thread.bit_length() - 5)
        return 0, blocks, idle, value_error, tuning_config.threads

    warps = blocks * useful // (32 * tuning_config.values_per_thread)
    for costly in (problem.target_bits == 4, problem.activation_type != ActivationType.None_):
        if costly and tuning_config.values_per_thread > 8:
            warps = warps * 8 // tuning_config.values_per_thread

    underfilled = warps < device.sm_count * 32
    preferred_values = 8 if problem.activation_type != ActivationType.None_ else 16
    if underfilled:
        thread_work = problem.tile_size // tuning_config.values_per_thread * 8
    else:
        thread_work = 2048 // tuning_config.values_per_thread
    target_threads = (
        128 if problem.activation_type != ActivationType.None_ else min(256, max(128, thread_work))
    )
    value_error = abs(tuning_config.values_per_thread.bit_length() - preferred_values.bit_length())
    thread_error = abs(tuning_config.threads - target_threads)
    if underfilled:
        return 1, -warps, thread_error, blocks, value_error
    return 0, value_error, thread_error, blocks, tuning_config.threads


def _partitioned_tile_score(problem, device, tuning_config: ProcessInputTuningConfig):
    """Shared-transform and staged-scale score from grid capacity."""
    rows = problem.num_work_rows
    blocks = rows * ceil_div(problem.hidden_size, tuning_config.columns_per_task)
    target_threads = 128 if device.sm_major < 10 else (64 if problem.hadamard_block_size > 1 else 32)
    if device.sm_major < 10 and problem.working_set_bytes > device.l2_cache_size:
        target_threads *= 2
    capacity = device.sm_count * max(1, device.max_threads_per_sm // target_threads // 4)
    row_blocks = max(1, rows * 4)
    occupancy_blocks = min(device.sm_count // 2, rows * min(8, rows * 4))
    target_blocks = min(capacity, max(row_blocks, occupancy_blocks))

    largest = max(problem.tile_size, problem.hadamard_block_size)
    preferred_values = max(16, largest // 32) if largest <= 1024 else 16
    if problem.working_set_bytes > device.l2_cache_size:
        preferred_values = min(preferred_values, 8)

    binary_activation = problem.activation_type in (
        ActivationType.BinarySplit,
        ActivationType.BinaryInterleaved,
    )
    natural_values = 8 if binary_activation or not problem.should_quantize else 16
    natural_threads = ceil_div(problem.hidden_size // natural_values, 32) * 32
    raw_unary = problem.activation_type == ActivationType.Unary and not problem.should_quantize
    binary_lowbit = binary_activation and problem.target_bits == 4
    resident_blocks = 4 if raw_unary or binary_lowbit else 1
    thread_limit = device.max_threads_per_sm // resident_blocks
    thread_limit = min(device.max_threads_per_block, thread_limit)
    large_grid = problem.num_work_rows >= 2 * device.sm_count
    full_row = device.sm_major >= 10 and large_grid
    full_row &= problem.quant_mode.dynamic_scale_mode != "group_token" and natural_threads <= thread_limit
    if full_row:
        target_blocks = rows
        target_threads = natural_threads
        preferred_values = natural_values

    columns = min(problem.hidden_size, tuning_config.columns_per_task)
    idle = tuning_config.threads * tuning_config.values_per_thread - columns
    block_error = abs(blocks - target_blocks) / max(blocks, target_blocks)
    return (
        block_error,
        idle,
        abs(tuning_config.values_per_thread - preferred_values),
        abs(tuning_config.threads - target_threads),
    )


def _raw_tile_score(problem, device, tuning_config: ProcessInputTuningConfig):
    """Rank elementwise tuning candidates from occupancy, tail work, and launch work."""
    source_bits = problem.source_bits
    preferred_values = 8 if problem.activation_type != ActivationType.None_ else max(1, 128 // source_bits)
    target_threads = 256 if device.sm_major >= 10 and problem.activation_type == ActivationType.Unary else 128
    columns = min(problem.hidden_size, tuning_config.columns_per_task)
    idle = tuning_config.threads * tuning_config.values_per_thread - columns
    underfilled = problem.num_work_rows < 2 * device.sm_count
    if problem.quant_mode == InputQuantizationMode.StaticTensor:
        target_threads = 128 if underfilled else 256
        target_columns = min(problem.hidden_size, 8 * target_threads)
        return (
            abs(columns - target_columns),
            idle,
            abs(tuning_config.values_per_thread - 8),
            abs(tuning_config.threads - target_threads),
        )
    if device.sm_major >= 10 and problem.activation_type == ActivationType.Unary and underfilled:
        blocks = problem.num_work_rows * ceil_div(problem.hidden_size, tuning_config.columns_per_task)
        target_blocks = 2 * device.sm_count
        return (
            tuning_config.values_per_thread < 4,
            tuning_config.values_per_thread > 8,
            max(0, target_blocks - blocks),
            idle,
            abs(tuning_config.threads - 128),
            abs(tuning_config.values_per_thread - 8),
            blocks,
        )
    if device.sm_major >= 10 and problem.activation_type == ActivationType.Unary and not underfilled:
        blocks = problem.num_work_rows * ceil_div(problem.hidden_size, tuning_config.columns_per_task)
        useful = problem.num_work_rows * problem.hidden_size
        capacity = blocks * tuning_config.threads * tuning_config.values_per_thread
        tail_ratio = (capacity - useful) / useful
        warps = blocks * tuning_config.threads // 32
        warps_per_sm = min(32, device.max_threads_per_sm // 32)
        warp_target = warps_per_sm * device.sm_count
        warp_shortfall = max(0, warp_target - warps) / warp_target
        resident_blocks = 4
        latency_waves = 4
        vector_block_target = latency_waves * resident_blocks * device.sm_count
        vector_block_shortfall = max(0, vector_block_target - blocks) / vector_block_target
        target_threads = min(256, device.max_threads_per_block)
        vector_thread_shortfall = max(0, target_threads - tuning_config.threads) / target_threads
        vector_shortfall = max(vector_block_shortfall, vector_thread_shortfall)
        vector_shortfall = vector_shortfall if tuning_config.values_per_thread > 8 else 0
        thread_limit = device.max_threads_per_sm // resident_blocks
        setup_values = resident_blocks * problem.tile_size
        estimated_work = capacity + capacity / tuning_config.values_per_thread + blocks * setup_values
        work_cost = estimated_work / useful
        execution_threads = target_threads if tuning_config.values_per_thread > 8 else target_threads // 2
        thread_error = abs(tuning_config.threads - execution_threads) / execution_threads
        occupancy_weight = (resident_blocks - 1) / resident_blocks
        work_cost += occupancy_weight * thread_error
        return (
            tuning_config.values_per_thread < 4,
            tuning_config.values_per_thread > 16,
            vector_shortfall,
            warp_shortfall,
            tuning_config.values_per_thread > 8 and tuning_config.threads > target_threads,
            tuning_config.threads > thread_limit,
            work_cost,
            tail_ratio,
            blocks,
        )
    if device.sm_major >= 10 and not underfilled and problem.activation_type != ActivationType.Unary:
        target_threads = device.max_threads_per_block
        if problem.should_quantize:
            preferred_values = 16
    return (
        idle,
        abs(tuning_config.values_per_thread - preferred_values),
        abs(tuning_config.threads - target_threads),
    )


def _pure_hadamard_score(problem, device, tuning_config: ProcessInputTuningConfig):
    """Rank transform tilings; short irregular rows need extra lane parallelism."""
    block_size = problem.hadamard_block_size
    transforms = problem.hidden_size // block_size
    source_bits = problem.source_bits
    fp32 = source_bits == 32
    natural_values = min(4, block_size) if fp32 else max(128 // source_bits, block_size // 32)
    values = natural_values
    lanes = block_size // values
    if fp32:
        tile_capacity = 1 if lanes > 32 else max(1, 128 // lanes)
        tiles = min(transforms, 1 << (tile_capacity.bit_length() - 1))
        while transforms % tiles:
            tiles //= 2
        values = min(values, max(1, block_size * tiles // 32))
        threads = block_size // values * tiles
        return (
            tuning_config.values_per_thread != values,
            (tuning_config.columns_per_task // problem.tile_size) != tiles,
            abs(tuning_config.threads - threads),
        )

    underfilled = problem.num_work_rows < 2 * device.sm_count
    irregular_tiles = transforms < 16 and transforms & (transforms - 1)
    if underfilled and irregular_tiles:
        values = max(4, natural_values // 2)
    blocks = problem.num_work_rows * ceil_div(
        transforms, (tuning_config.columns_per_task // problem.tile_size)
    )
    columns = (tuning_config.columns_per_task // problem.tile_size) * block_size
    idle = tuning_config.threads * tuning_config.values_per_thread - columns
    if not underfilled:
        target_tiles = min(transforms, values)
        row_capacity = (
            ceil_div(transforms, (tuning_config.columns_per_task // problem.tile_size))
            * tuning_config.threads
            * tuning_config.values_per_thread
        )
        capacity_overhead = (row_capacity - problem.hidden_size) / problem.hidden_size
        tile_error = abs((tuning_config.columns_per_task // problem.tile_size) - target_tiles) / target_tiles
        return (
            tuning_config.values_per_thread != values,
            capacity_overhead + tile_error,
            idle,
            abs(tuning_config.threads - 128),
        )
    target_blocks = ceil_div(3 * device.sm_count, 2)
    grid_shortfall = max(0, target_blocks - blocks)
    return (
        tuning_config.values_per_thread < 4,
        grid_shortfall,
        idle,
        abs(tuning_config.threads - 128),
        abs(tuning_config.values_per_thread - natural_values),
        blocks,
    )


def _select_tuning_config(problem, device) -> ProcessInputTuningConfig:
    block_size = problem.tile_size
    if problem.hadamard_block_size > 1:
        block_size = problem.hadamard_block_size
    token_tuning_configs = tuple(_token_tuning_candidates(problem, device, block_size))
    tile_tuning_configs = tuple(_tile_tuning_candidates(problem, device, block_size))

    dynamic_scale_mode = problem.quant_mode.dynamic_scale_mode
    token_partition = dynamic_scale_mode == "token"
    if dynamic_scale_mode == "group_token":
        token_partition = bool(token_tuning_configs) and problem.num_work_rows <= device.sm_count
    if token_partition:
        assert token_tuning_configs
        return min(
            token_tuning_configs, key=lambda tuning_config: _token_score(problem, device, tuning_config)
        )

    assert tile_tuning_configs
    pure_hadamard = not problem.should_quantize and problem.activation_type == ActivationType.None_
    pure_hadamard &= problem.hadamard_block_size > 1
    direct_group = problem.should_quantize and problem.hadamard_block_size <= 1
    direct_group &= dynamic_scale_mode == "group"
    if pure_hadamard:
        tuning_config = min(tile_tuning_configs, key=lambda item: _pure_hadamard_score(problem, device, item))
    elif direct_group:
        tuning_config = min(tile_tuning_configs, key=lambda item: _direct_group_score(problem, device, item))
    elif problem.hadamard_block_size > 1 or dynamic_scale_mode == "group_token":
        tuning_config = min(
            tile_tuning_configs, key=lambda item: _partitioned_tile_score(problem, device, item)
        )
    else:
        tuning_config = min(tile_tuning_configs, key=lambda item: _raw_tile_score(problem, device, item))
    if dynamic_scale_mode == "group_token":
        tuning_config = dataclasses.replace(tuning_config, finalize_tokens_per_block=4)
    return tuning_config


def select_process_input_tuning_config(config: ProcessInputConfig, shape_m: int) -> ProcessInputTuningConfig:
    """Choose a tuning configuration for shape_m on the current CUDA device."""
    return _tuning_config_for_device(config, shape_m, get_device_index())


@functools.lru_cache(maxsize=1024)
def _tuning_config_for_device(config, shape_m, device_index):
    device = DeviceInfo(device_index)
    mode = config.quant_mode
    minimum_sm = {
        dtypes.int4: 75,
        dtypes.int8: 75,
        dtypes.float8e4m3: 89,
        dtypes.float8e5m2: 89,
        dtypes.float4e0m3: 100,
        dtypes.float4e2m1: 100,
        dtypes.float8e3m4: 100,
    }.get(config.quant_dtype)
    if config.quant_dtype is not None and minimum_sm is None:
        raise ValueError(f"unsupported quant_dtype: {config.quant_dtype}")
    if minimum_sm is not None and device.sm_version < minimum_sm:
        raise RuntimeError(f"{config.quant_dtype} output requires SM{minimum_sm} or newer")
    rows = max(shape_m, 1)
    tile_size = min(config.hidden_size & -config.hidden_size, 256)
    if mode.has_group_scale:
        tile_size = config.quant_group_size
    elif not mode.should_quantize and config.hadamard_block_size > 1:
        tile_size = config.hadamard_block_size
    # These restrictions are needed to enumerate legal tuning configurations before JIT.
    for size in (tile_size, config.hadamard_block_size):
        assert size > 0 and size <= 512 and size & (size - 1) == 0
        assert config.hidden_size % size == 0
    assert not mode.has_group_scale or config.quant_group_size >= 2
    separate = config.scatter_width > 1 and config.activation_type == ActivationType.None_
    separate &= config.hadamard_block_size == 1
    row_limit = device.sm_count if mode.has_token_scale else (device.sm_count + 1) // 2
    separate &= rows * config.scatter_width <= row_limit
    source_bits = config.input_dtype.num_bits
    target_bits = (config.quant_dtype or config.input_dtype).num_bits
    output_rows = rows * config.scatter_width
    working_set_bytes = _working_set_bytes(config, rows)
    problem = SimpleNamespace(
        **config.to_dict(),
        should_quantize=mode.should_quantize,
        tile_size=tile_size,
        num_tiles=config.hidden_size // tile_size,
        source_bits=source_bits,
        target_bits=target_bits,
        num_work_rows=output_rows if separate else rows,
        outputs_per_task=1 if separate else config.scatter_width,
        working_set_bytes=working_set_bytes,
    )
    tuning_config = _select_tuning_config(problem, device)
    return dataclasses.replace(tuning_config, separate_outputs=separate)


@functools.lru_cache(maxsize=128)
def _tuning_intervals_for_device(config, device_index, use_pdl):
    device = DeviceInfo(device_index)
    tuning_configs_by_shape_m = {}

    def get_tuning_config(shape_m):
        if shape_m not in tuning_configs_by_shape_m:
            tuning_configs_by_shape_m[shape_m] = dataclasses.replace(
                _tuning_config_for_device(config, shape_m, device_index),
                use_pdl=use_pdl and device.sm_major >= 9,
            )
        return tuning_configs_by_shape_m[shape_m]

    def partition(min_shape_m: int, max_shape_m: int):
        # Match launcher ranges: min_shape_m < shape_m <= max_shape_m.
        if min_shape_m >= max_shape_m:
            return []
        distance = max_shape_m - min_shape_m - 1
        probes = {
            min_shape_m + 1,
            max_shape_m,
            min_shape_m + 1 + distance // 4,
            min_shape_m + 1 + distance // 2,
            min_shape_m + 1 + 3 * distance // 4,
        }
        probe_tuning_configs = {get_tuning_config(row) for row in probes}
        if len(probe_tuning_configs) == 1:
            return [(min_shape_m, max_shape_m, probe_tuning_configs.pop())]
        if distance <= 16:
            return [(row - 1, row, get_tuning_config(row)) for row in range(min_shape_m + 1, max_shape_m + 1)]
        middle = (min_shape_m + 1 + max_shape_m) // 2
        return partition(min_shape_m, middle) + partition(middle, max_shape_m)

    small_limit = max(1, 4 * device.sm_count)
    tuning_intervals = [(row - 1, row, get_tuning_config(row)) for row in range(1, small_limit + 1)]
    bytes_per_row = _working_set_bytes(config, 1)
    l2_rows = max(1, device.l2_cache_size // bytes_per_row)
    maximum = 1 << 30
    cuts = {small_limit, maximum}
    cuts.update(row for row in range(l2_rows - 2, l2_rows + 3) if small_limit < row < maximum)
    min_shape_m = small_limit
    for max_shape_m in sorted(cuts):
        if max_shape_m <= min_shape_m:
            continue
        tuning_intervals.extend(partition(min_shape_m, max_shape_m))
        min_shape_m = max_shape_m

    merged = []
    for min_shape_m, max_shape_m, tuning_config in tuning_intervals:
        if merged and merged[-1][1] == min_shape_m and merged[-1][2] == tuning_config:
            merged[-1] = (merged[-1][0], max_shape_m, tuning_config)
        else:
            merged.append((min_shape_m, max_shape_m, tuning_config))
    return merged


def get_process_input_tuning_intervals(config: ProcessInputConfig, use_pdl: bool = False):
    """Return (min_shape_m, max_shape_m, tuning_config) with min < shape_m <= max."""
    return _tuning_intervals_for_device(config, get_device_index(), use_pdl)


def _working_set_bytes(config, rows):
    mode = config.quant_mode
    source_bits = config.input_dtype.num_bits
    target_bits = (config.quant_dtype or config.input_dtype).num_bits
    output_rows = rows * config.scatter_width
    working_set_bytes = (
        rows * config.input_row_size * source_bits + output_rows * config.hidden_size * target_bits
    ) // 8
    if mode.has_group_scale:
        working_set_bytes += (
            math.prod(config.get_group_scale_shape(output_rows)) * config.group_scale_dtype.num_bits // 8
        )
    if mode.has_token_scale:
        working_set_bytes += output_rows * 4
    return working_set_bytes
