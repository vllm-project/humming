#pragma once

#include <humming/kernel/process_input/context.cuh>

// Logical columns before activation addressing or output packing. skip_all
// never permits skipping the warp/block collectives required by the algorithm.
template <class Context>
struct ProcessInputThreadTask {
  uint64_t input_row;
  uint64_t output_row;
  uint32_t input_col;
  uint32_t output_col;
  bool skip_read;
  bool skip_all;
  bool skip_output;
  bool zero;

  // Reading is shared across routes: an invalid route must not suppress the
  // computation needed by another valid destination.
  CUDA_INLINE explicit ProcessInputThreadTask(const Context &ctx, uint32_t route = 0)
      : input_row(ctx.input_row), output_row(ctx.output_rows[route]),
        input_col(ctx.column), output_col(ctx.column) {
    bool active_column = ctx.column < Context::kHiddenSize;
    skip_read = !active_column || !ctx.load;
    skip_all = !active_column || (!ctx.load && !ctx.zero);
    skip_output = !active_column || output_row == ~uint64_t{0};
    zero = ctx.zero_outputs[route];
  }

  CUDA_INLINE bool zero_output() const { return zero && !skip_output; }

  template <class SourceType>
  CUDA_INLINE const SourceType *input(const SourceType *base) const {
    return base + input_row * Context::kInputRowSize + input_col;
  }

  template <uint32_t kBits>
  CUDA_INLINE uint8_t *output(void *base) const {
    return reinterpret_cast<uint8_t *>(base) + (output_row * Context::kHiddenSize + output_col) * kBits / 8;
  }
};
