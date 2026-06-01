#pragma once

#include "tensor.h"

namespace cudasep::tensor_tile {

constexpr int kMaxDims = 8;

struct StridedCopyMeta {
    int64_t src_strides[kMaxDims];
    int64_t dst_shape[kMaxDims];
    int ndim;
};

struct BroadcastMeta {
    int64_t out_shape[kMaxDims];
    int64_t a_strides[kMaxDims];
    int64_t b_strides[kMaxDims];
    int ndim;
};

struct PadMeta {
    int64_t src_shape[kMaxDims];
    int64_t dst_shape[kMaxDims];
    int64_t pad_before[kMaxDims];
    int ndim;
};

struct CatSrcInfo {
    const void* ptr;
    int64_t dim_size;
    int64_t dim_offset;
};

enum class BinaryOp { Add, Sub, Mul, Div };

void fill(Tensor& dst, float value);
void arange(Tensor& dst, int64_t start);
void convert_dtype(const Tensor& src, Tensor& dst);
void add_scalar(Tensor& data, float value);
void mul_scalar(Tensor& data, float value);
void clamp(Tensor& data, float min_value, float max_value);
void negate(const Tensor& src, Tensor& dst);
void greater_than(const Tensor& src, Tensor& dst, float value);

void binary_same_shape(const Tensor& a, const Tensor& b, Tensor& out, BinaryOp op);
void binary_inplace_same_shape(Tensor& a, const Tensor& b, BinaryOp op);

void binary_glue(const Tensor& a, const Tensor& b, Tensor& out,
                 const BroadcastMeta& meta, BinaryOp op);
void binary_inplace_glue(Tensor& a, const Tensor& b,
                         const BroadcastMeta& meta, BinaryOp op);

void index_select_glue(const Tensor& src, Tensor& dst, const Tensor& indices,
                       const StridedCopyMeta& meta, int select_dim);
void strided_copy_glue(const Tensor& src, Tensor& dst, const StridedCopyMeta& meta);
void pad_const_glue(const Tensor& src, Tensor& dst, const PadMeta& meta, float value);
void pad_reflect_glue(const Tensor& src, Tensor& dst, const PadMeta& meta);
void cat_glue(char* out, const CatSrcInfo* infos, int n_infos,
              int64_t outer_size, int64_t total_cat_dim,
              int64_t inner_bytes, int64_t max_src_bytes);

void sum_reduce_last_dim(const Tensor& src, Tensor& dst,
                         int64_t inner_size, int64_t outer_size);
void max_reduce_last_dim(const Tensor& src, Tensor& dst,
                         int64_t inner_size, int64_t outer_size);
void sum_reduce_general_glue(const Tensor& src, Tensor& dst,
                             int64_t outer, int64_t reduce, int64_t inner);
void max_reduce_general_glue(const Tensor& src, Tensor& dst,
                             int64_t outer, int64_t reduce, int64_t inner);

}  // namespace cudasep::tensor_tile
