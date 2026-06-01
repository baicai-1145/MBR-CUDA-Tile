#pragma once

#include "tensor.h"

namespace cudasep::chunk_tile {

void accumulate_chunk(Tensor& dest,
                      Tensor& weight_sum,
                      const Tensor& src,
                      const Tensor& window,
                      int64_t offset);

void normalize_by_weights(Tensor& data, const Tensor& weight_sum);

}  // namespace cudasep::chunk_tile
