#include "tensor.h"
#include "memory_pool.h"
#include "tensor_cuda_tile.h"
#include <algorithm>
#include <sstream>
#include <cmath>

namespace cudasep {

// ============================================================================
// Tensor Implementation
// ============================================================================

Tensor::Tensor()
    : storage_(nullptr), data_(nullptr), numel_(0), dtype_(DType::Float32) {}

std::shared_ptr<void> Tensor::alloc_gpu(size_t bytes) {
    if (bytes == 0) return nullptr;
    void* ptr = CudaMemoryPool::instance().allocate(bytes);
    return std::shared_ptr<void>(ptr, [](void* p) {
        CudaMemoryPool::instance().deallocate(p);
    });
}

void Tensor::compute_strides() {
    strides_.resize(shape_.size());
    if (shape_.empty()) return;
    strides_.back() = 1;
    for (int i = (int)shape_.size() - 2; i >= 0; --i) {
        strides_[i] = strides_[i + 1] * shape_[i + 1];
    }
}

// --- size ---
int64_t Tensor::size(int dim) const {
    if (dim < 0) dim += ndim();
    if (dim < 0 || dim >= ndim())
        throw std::out_of_range("Tensor::size: dim out of range");
    return shape_[dim];
}

// --- is_contiguous ---
bool Tensor::is_contiguous() const {
    if (numel_ <= 1) return true;
    int64_t expected = 1;
    for (int i = (int)shape_.size() - 1; i >= 0; --i) {
        if (shape_[i] != 1 && strides_[i] != expected) return false;
        expected *= shape_[i];
    }
    return true;
}

// --- Factory: empty ---
Tensor Tensor::empty(std::vector<int64_t> shape, DType dtype) {
    Tensor t;
    t.shape_ = std::move(shape);
    t.dtype_ = dtype;
    t.numel_ = 1;
    for (auto s : t.shape_) t.numel_ *= s;
    t.compute_strides();
    size_t bytes = t.numel_ * dtype_size(dtype);
    t.storage_ = alloc_gpu(bytes);
    t.data_ = t.storage_.get();
    return t;
}

// --- Factory: zeros ---
Tensor Tensor::zeros(std::vector<int64_t> shape, DType dtype) {
    Tensor t = empty(std::move(shape), dtype);
    if (t.numel_ > 0) {
        CUDA_CHECK(cudaMemset(t.data_, 0, t.numel_ * dtype_size(dtype)));
    }
    return t;
}

// --- Factory: ones ---
Tensor Tensor::ones(std::vector<int64_t> shape, DType dtype) {
    return full(std::move(shape), 1.0f, dtype);
}

// --- Factory: full ---
Tensor Tensor::full(std::vector<int64_t> shape, float value, DType dtype) {
    Tensor t = empty(std::move(shape), dtype);
    if (t.numel_ == 0) return t;
    tensor_tile::fill(t, value);
    return t;
}

// --- Factory: from_cpu_f32 ---
Tensor Tensor::from_cpu_f32(const float* data, std::vector<int64_t> shape) {
    Tensor t = empty(shape, DType::Float32);
    if (t.numel_ > 0) {
        CUDA_CHECK(cudaMemcpy(t.data_, data, t.numel_ * sizeof(float), cudaMemcpyHostToDevice));
    }
    return t;
}

// --- Factory: from_cpu_f16 ---
Tensor Tensor::from_cpu_f16(const void* data, std::vector<int64_t> shape) {
    Tensor t = empty(shape, DType::Float16);
    if (t.numel_ > 0) {
        CUDA_CHECK(cudaMemcpy(t.data_, data, t.numel_ * sizeof(__half), cudaMemcpyHostToDevice));
    }
    return t;
}

// --- Factory: from_cpu_i64 ---
Tensor Tensor::from_cpu_i64(const int64_t* data, std::vector<int64_t> shape) {
    Tensor t = empty(shape, DType::Int64);
    if (t.numel_ > 0) {
        CUDA_CHECK(cudaMemcpy(t.data_, data, t.numel_ * sizeof(int64_t), cudaMemcpyHostToDevice));
    }
    return t;
}

// --- Factory: arange ---
Tensor Tensor::arange(int64_t start, int64_t end, DType dtype) {
    int64_t N = end - start;
    if (N <= 0) return Tensor();
    Tensor t = empty({N}, dtype);
    tensor_tile::arange(t, start);
    return t;
}

// --- to_cpu_f32 ---
std::vector<float> Tensor::to_cpu_f32() const {
    if (dtype_ == DType::Float32) {
        Tensor c = is_contiguous() ? *this : contiguous();
        std::vector<float> result(numel_);
        CUDA_CHECK(cudaMemcpy(result.data(), c.data_, numel_ * sizeof(float), cudaMemcpyDeviceToHost));
        return result;
    } else if (dtype_ == DType::Float16) {
        Tensor f32 = to_f32();
        return f32.to_cpu_f32();
    } else {
        // Int64 -> float
        Tensor f32 = to_dtype(DType::Float32);
        return f32.to_cpu_f32();
    }
}

// --- copy_from_cpu ---
void Tensor::copy_from_cpu(const void* src, size_t bytes) {
    CUDA_CHECK(cudaMemcpy(data_, src, bytes, cudaMemcpyHostToDevice));
}

// --- copy_to_cpu ---
void Tensor::copy_to_cpu(void* dst, size_t bytes) const {
    CUDA_CHECK(cudaMemcpy(dst, data_, bytes, cudaMemcpyDeviceToHost));
}

// --- reshape ---
Tensor Tensor::reshape(std::vector<int64_t> new_shape) const {
    // Resolve -1
    int neg_idx = -1;
    int64_t prod = 1;
    for (int i = 0; i < (int)new_shape.size(); ++i) {
        if (new_shape[i] == -1) {
            if (neg_idx >= 0) throw std::runtime_error("reshape: only one -1 allowed");
            neg_idx = i;
        } else {
            prod *= new_shape[i];
        }
    }
    if (neg_idx >= 0) {
        new_shape[neg_idx] = numel_ / prod;
    }
    // Verify
    int64_t new_numel = 1;
    for (auto s : new_shape) new_numel *= s;
    if (new_numel != numel_)
        throw std::runtime_error("reshape: incompatible sizes " +
                                 std::to_string(numel_) + " vs " + std::to_string(new_numel));

    if (!is_contiguous()) {
        // Must copy first
        Tensor c = contiguous();
        Tensor t;
        t.storage_ = c.storage_;
        t.data_ = c.data_;
        t.shape_ = std::move(new_shape);
        t.numel_ = numel_;
        t.dtype_ = dtype_;
        t.compute_strides();
        return t;
    }

    Tensor t;
    t.storage_ = storage_;
    t.data_ = data_;
    t.shape_ = std::move(new_shape);
    t.numel_ = numel_;
    t.dtype_ = dtype_;
    t.compute_strides();
    return t;
}

// --- permute (lazy - zero-copy stride reordering) ---
Tensor Tensor::permute(std::vector<int> dims) const {
    int nd = ndim();
    if ((int)dims.size() != nd)
        throw std::runtime_error("permute: dims size mismatch");

    Tensor out;
    out.storage_ = storage_;
    out.data_ = data_;
    out.dtype_ = dtype_;
    out.numel_ = numel_;
    out.shape_.resize(nd);
    out.strides_.resize(nd);
    for (int i = 0; i < nd; ++i) {
        out.shape_[i] = shape_[dims[i]];
        out.strides_[i] = strides_[dims[i]];
    }
    return out;
}

// --- transpose ---
Tensor Tensor::transpose(int dim0, int dim1) const {
    int nd = ndim();
    if (dim0 < 0) dim0 += nd;
    if (dim1 < 0) dim1 += nd;
    std::vector<int> dims(nd);
    std::iota(dims.begin(), dims.end(), 0);
    std::swap(dims[dim0], dims[dim1]);
    return permute(dims);
}

// --- contiguous ---
Tensor Tensor::contiguous() const {
    if (is_contiguous()) {
        return *this;  // Already contiguous - return shared view (no copy)
    }
    // Need to copy with strides
    int nd = ndim();
    Tensor out = Tensor::empty(shape_, dtype_);
    if (numel_ == 0) return out;

    tensor_tile::StridedCopyMeta meta;
    meta.ndim = nd;
    for (int i = 0; i < nd; ++i) {
        meta.src_strides[i] = strides_[i];
        meta.dst_shape[i] = shape_[i];
    }
    tensor_tile::strided_copy_glue(*this, out, meta);

    return out;
}

// --- unsqueeze ---
Tensor Tensor::unsqueeze(int dim) const {
    if (dim < 0) dim += ndim() + 1;
    std::vector<int64_t> new_shape = shape_;
    new_shape.insert(new_shape.begin() + dim, 1);
    return reshape(new_shape);
}

// --- squeeze ---
Tensor Tensor::squeeze(int dim) const {
    if (dim < 0) dim += ndim();
    if (shape_[dim] != 1)
        throw std::runtime_error("squeeze: dimension is not 1");
    std::vector<int64_t> new_shape = shape_;
    new_shape.erase(new_shape.begin() + dim);
    return reshape(new_shape);
}

// --- expand (lazy view - zero-copy with broadcast strides) ---
Tensor Tensor::expand(std::vector<int64_t> new_shape) const {
    int nd = (int)new_shape.size();
    if (nd < ndim()) throw std::runtime_error("expand: cannot reduce dims");

    // Pad shape/strides on the left with 1s if needed
    std::vector<int64_t> src_shape = shape_;
    std::vector<int64_t> src_strides = strides_;
    while ((int)src_shape.size() < nd) {
        src_shape.insert(src_shape.begin(), 1);
        src_strides.insert(src_strides.begin(), 0);
    }

    // Validate and compute broadcast strides
    std::vector<int64_t> bcast_strides(nd);
    for (int d = 0; d < nd; ++d) {
        if (src_shape[d] == new_shape[d]) {
            bcast_strides[d] = src_strides[d];
        } else if (src_shape[d] == 1) {
            bcast_strides[d] = 0; // broadcast
        } else {
            throw std::runtime_error("expand: incompatible shape at dim " + std::to_string(d));
        }
    }

    Tensor out;
    out.storage_ = storage_;  // share ownership
    out.data_ = data_;
    out.dtype_ = dtype_;
    out.shape_ = std::move(new_shape);
    out.strides_ = std::move(bcast_strides);
    out.numel_ = 1;
    for (auto s : out.shape_) out.numel_ *= s;
    return out;
}

// --- slice (lazy view - zero-copy) ---
Tensor Tensor::slice(int dim, int64_t start, int64_t end) const {
    if (dim < 0) dim += ndim();
    if (start < 0) start += shape_[dim];
    if (end < 0) end += shape_[dim];
    if (start < 0) start = 0;
    if (end > shape_[dim]) end = shape_[dim];
    if (start >= end) return Tensor();

    Tensor out;
    out.storage_ = storage_;  // share ownership
    out.dtype_ = dtype_;
    out.shape_ = shape_;
    out.shape_[dim] = end - start;
    out.strides_ = strides_;
    out.numel_ = 1;
    for (auto s : out.shape_) out.numel_ *= s;
    // Offset data pointer by start * stride along the sliced dimension
    out.data_ = (char*)data_ + start * strides_[dim] * (int64_t)dtype_size(dtype_);
    return out;
}

// --- index_select ---
Tensor Tensor::index_select(int dim, const Tensor& indices) const {
    if (dim < 0) dim += ndim();
    if (indices.dtype() != DType::Int64)
        throw std::runtime_error("index_select: indices must be Int64");

    int64_t n_idx = indices.numel();
    std::vector<int64_t> out_shape = shape_;
    out_shape[dim] = n_idx;

    int64_t out_numel = 1;
    for (auto s : out_shape) out_numel *= s;
    Tensor out = Tensor::empty(out_shape, dtype_);
    if (out_numel == 0) return out;

    int nd = ndim();
    tensor_tile::StridedCopyMeta meta;
    meta.ndim = nd;
    for (int i = 0; i < nd; ++i) {
        meta.src_strides[i] = strides_[i];
        meta.dst_shape[i] = out_shape[i];
    }
    tensor_tile::index_select_glue(*this, out, indices, meta, dim);

    return out;
}

// --- cat ---
Tensor Tensor::cat(const std::vector<Tensor>& tensors, int dim) {
    if (tensors.empty())
        throw std::runtime_error("cat: empty tensor list");

    int nd = tensors[0].ndim();
    if (dim < 0) dim += nd;

    DType dt = tensors[0].dtype();
    std::vector<int64_t> out_shape = tensors[0].shape();

    // Sum the concat dim
    int64_t total_dim = 0;
    for (auto& t : tensors) {
        if (t.ndim() != nd)
            throw std::runtime_error("cat: ndim mismatch");
        if (t.dtype() != dt)
            throw std::runtime_error("cat: dtype mismatch");
        total_dim += t.size(dim);
    }
    out_shape[dim] = total_dim;

    Tensor out = Tensor::empty(out_shape, dt);
    int elem_size = (int)dtype_size(dt);

    // Compute: outer_size = product of dims before `dim`
    //          inner_size = product of dims after `dim`
    int64_t outer_size = 1;
    for (int d = 0; d < dim; ++d) outer_size *= out_shape[d];
    int64_t inner_size = 1;
    for (int d = dim + 1; d < nd; ++d) inner_size *= out_shape[d];

    int64_t inner_bytes = inner_size * elem_size;
    int N = (int)tensors.size();

    // Ensure all tensors are contiguous and build metadata
    std::vector<Tensor> contig(N);
    std::vector<tensor_tile::CatSrcInfo> h_infos(N);
    int64_t dim_offset = 0;
    int64_t max_src_bytes = 0;
    for (int i = 0; i < N; i++) {
        contig[i] = tensors[i].contiguous();
        h_infos[i].ptr = contig[i].data_;
        h_infos[i].dim_size = contig[i].size(dim);
        h_infos[i].dim_offset = dim_offset;
        int64_t sb = outer_size * h_infos[i].dim_size * inner_bytes;
        if (sb > max_src_bytes) max_src_bytes = sb;
        dim_offset += h_infos[i].dim_size;
    }

    tensor_tile::cat_glue((char*)out.data_, h_infos.data(), N,
                          outer_size, total_dim, inner_bytes, max_src_bytes);

    return out;
}

// --- stack ---
Tensor Tensor::stack(const std::vector<Tensor>& tensors, int dim) {
    if (tensors.empty())
        throw std::runtime_error("stack: empty tensor list");

    // Unsqueeze all tensors at dim, then cat
    std::vector<Tensor> unsqueezed;
    unsqueezed.reserve(tensors.size());
    for (auto& t : tensors) {
        unsqueezed.push_back(t.unsqueeze(dim));
    }
    return cat(unsqueezed, dim);
}

// --- split ---
std::vector<Tensor> Tensor::split(int dim, const std::vector<int64_t>& sizes) const {
    if (dim < 0) dim += ndim();
    int64_t total = 0;
    for (auto s : sizes) total += s;
    if (total != shape_[dim])
        throw std::runtime_error("split: sizes don't sum to dim size");

    std::vector<Tensor> result;
    result.reserve(sizes.size());
    int64_t offset = 0;
    for (auto sz : sizes) {
        result.push_back(slice(dim, offset, offset + sz));
        offset += sz;
    }
    return result;
}

// --- unbind ---
std::vector<Tensor> Tensor::unbind(int dim) const {
    if (dim < 0) dim += ndim();
    int64_t n = shape_[dim];
    std::vector<Tensor> result;
    result.reserve(n);
    for (int64_t i = 0; i < n; ++i) {
        Tensor s = slice(dim, i, i + 1);
        // Squeeze the dim
        result.push_back(s.squeeze(dim));
    }
    return result;
}

// --- pad ---
Tensor Tensor::pad(const std::vector<int64_t>& padding, float value) const {
    // padding is {left, right} for last dim, or {left, right, top, bottom} etc.
    // PyTorch convention: pairs from last dim backwards
    int n_pairs = (int)padding.size() / 2;
    int nd = ndim();

    // Build per-dim pad_before and pad_after
    std::vector<int64_t> pad_before(nd, 0);
    std::vector<int64_t> pad_after(nd, 0);
    for (int i = 0; i < n_pairs && i < nd; ++i) {
        int dim_idx = nd - 1 - i;
        pad_before[dim_idx] = padding[2 * i];
        pad_after[dim_idx] = padding[2 * i + 1];
    }

    std::vector<int64_t> out_shape(nd);
    for (int d = 0; d < nd; ++d) {
        out_shape[d] = shape_[d] + pad_before[d] + pad_after[d];
    }

    int64_t out_numel = 1;
    for (auto s : out_shape) out_numel *= s;
    Tensor out = Tensor::empty(out_shape, dtype_);
    if (out_numel == 0) return out;

    tensor_tile::PadMeta pmeta;
    pmeta.ndim = nd;
    for (int i = 0; i < nd; ++i) {
        pmeta.src_shape[i] = shape_[i];
        pmeta.dst_shape[i] = out_shape[i];
        pmeta.pad_before[i] = pad_before[i];
    }

    tensor_tile::pad_const_glue(*this, out, pmeta, value);

    return out;
}

// --- pad_reflect ---
Tensor Tensor::pad_reflect(const std::vector<int64_t>& padding) const {
    int n_pairs = (int)padding.size() / 2;
    int nd = ndim();

    std::vector<int64_t> pad_before(nd, 0);
    std::vector<int64_t> pad_after(nd, 0);
    for (int i = 0; i < n_pairs && i < nd; ++i) {
        int dim_idx = nd - 1 - i;
        pad_before[dim_idx] = padding[2 * i];
        pad_after[dim_idx] = padding[2 * i + 1];
    }

    std::vector<int64_t> out_shape(nd);
    for (int d = 0; d < nd; ++d) {
        out_shape[d] = shape_[d] + pad_before[d] + pad_after[d];
    }

    int64_t out_numel = 1;
    for (auto s : out_shape) out_numel *= s;
    Tensor out = Tensor::empty(out_shape, dtype_);
    if (out_numel == 0) return out;

    tensor_tile::PadMeta pmeta;
    pmeta.ndim = nd;
    for (int i = 0; i < nd; ++i) {
        pmeta.src_shape[i] = shape_[i];
        pmeta.dst_shape[i] = out_shape[i];
        pmeta.pad_before[i] = pad_before[i];
    }

    tensor_tile::pad_reflect_glue(*this, out, pmeta);

    return out;
}

// --- to_dtype ---
Tensor Tensor::to_dtype(DType new_dtype) const {
    if (new_dtype == dtype_) return *this;  // No-op when already correct dtype

    Tensor out = Tensor::empty(shape_, new_dtype);
    if (numel_ == 0) return out;
    tensor_tile::convert_dtype(*this, out);
    return out;
}

// ============================================================================
// Broadcasting helpers
// ============================================================================

static tensor_tile::BroadcastMeta compute_broadcast_meta(
    const std::vector<int64_t>& a_shape,
    const std::vector<int64_t>& a_strides,
    const std::vector<int64_t>& b_shape,
    const std::vector<int64_t>& b_strides,
    std::vector<int64_t>& out_shape) {
    int nd_a = (int)a_shape.size();
    int nd_b = (int)b_shape.size();
    int nd = std::max(nd_a, nd_b);
    if (nd > tensor_tile::kMaxDims) throw std::runtime_error("broadcast: too many dims (max 8)");

    out_shape.resize(nd);
    tensor_tile::BroadcastMeta meta;
    meta.ndim = nd;

    for (int d = 0; d < nd; ++d) {
        int a_idx = nd_a - nd + d;
        int b_idx = nd_b - nd + d;
        int64_t sa = (a_idx >= 0) ? a_shape[a_idx] : 1;
        int64_t sb = (b_idx >= 0) ? b_shape[b_idx] : 1;
        int64_t stride_a = (a_idx >= 0 && sa != 1) ? a_strides[a_idx] : 0;
        int64_t stride_b = (b_idx >= 0 && sb != 1) ? b_strides[b_idx] : 0;

        if (sa != sb && sa != 1 && sb != 1) {
            throw std::runtime_error("broadcast: incompatible shapes");
        }
        out_shape[d] = std::max(sa, sb);
        meta.out_shape[d] = out_shape[d];
        meta.a_strides[d] = (sa == 1) ? 0 : stride_a;
        meta.b_strides[d] = (sb == 1) ? 0 : stride_b;
    }
    return meta;
}

// For in-place: `a` is the output, so a_shape must match out_shape.
static tensor_tile::BroadcastMeta compute_broadcast_meta_inplace(
    const std::vector<int64_t>& a_shape,
    const std::vector<int64_t>& b_shape,
    const std::vector<int64_t>& b_strides) {
    int nd_a = (int)a_shape.size();
    int nd_b = (int)b_shape.size();
    int nd = nd_a; // a is output, so nd == nd_a
    if (nd > tensor_tile::kMaxDims) throw std::runtime_error("broadcast inplace: too many dims");

    tensor_tile::BroadcastMeta meta;
    meta.ndim = nd;

    for (int d = 0; d < nd; ++d) {
        int b_idx = nd_b - nd + d;
        int64_t sb = (b_idx >= 0) ? b_shape[b_idx] : 1;
        int64_t stride_b = (b_idx >= 0 && sb != 1) ? b_strides[b_idx] : 0;

        if (a_shape[d] != sb && sb != 1) {
            throw std::runtime_error("broadcast inplace: incompatible shapes");
        }
        meta.out_shape[d] = a_shape[d];
        meta.a_strides[d] = 0; // not used for inplace (a is contiguous)
        meta.b_strides[d] = (sb == 1) ? 0 : stride_b;
    }
    return meta;
}

// ============================================================================
// Element-wise operators
// ============================================================================

template <tensor_tile::BinaryOp op>
static Tensor binary_op(const Tensor& a, const Tensor& b) {
    DType dt = a.dtype();
    if (dt != b.dtype()) throw std::runtime_error("binary op: dtype mismatch");

    if (a.shape() == b.shape() && a.is_contiguous() && b.is_contiguous()) {
        Tensor out = Tensor::empty(a.shape(), dt);
        tensor_tile::binary_same_shape(a, b, out, op);
        return out;
    }

    std::vector<int64_t> out_shape;
    tensor_tile::BroadcastMeta meta = compute_broadcast_meta(
        a.shape(), a.strides(), b.shape(), b.strides(), out_shape);

    int64_t out_numel = 1;
    for (auto s : out_shape) out_numel *= s;

    Tensor out = Tensor::empty(out_shape, dt);
    if (out_numel == 0) return out;
    tensor_tile::binary_glue(a, b, out, meta, op);
    return out;
}

Tensor Tensor::operator+(const Tensor& other) const { return binary_op<tensor_tile::BinaryOp::Add>(*this, other); }
Tensor Tensor::operator-(const Tensor& other) const { return binary_op<tensor_tile::BinaryOp::Sub>(*this, other); }
Tensor Tensor::operator*(const Tensor& other) const { return binary_op<tensor_tile::BinaryOp::Mul>(*this, other); }
Tensor Tensor::operator/(const Tensor& other) const { return binary_op<tensor_tile::BinaryOp::Div>(*this, other); }

Tensor Tensor::operator-() const {
    Tensor out = Tensor::empty(shape_, dtype_);
    if (numel_ == 0) return out;
    tensor_tile::negate(*this, out);
    return out;
}

// --- In-place operations ---

template <tensor_tile::BinaryOp op>
static void binary_op_inplace(Tensor& a, const Tensor& b) {
    DType dt = a.dtype();
    if (dt != b.dtype()) throw std::runtime_error("inplace binary op: dtype mismatch");

    if (a.shape() == b.shape() && a.is_contiguous() && b.is_contiguous()) {
        tensor_tile::binary_inplace_same_shape(a, b, op);
        return;
    }

    tensor_tile::BroadcastMeta meta = compute_broadcast_meta_inplace(a.shape(), b.shape(), b.strides());
    int64_t N = a.numel();
    if (N == 0) return;
    tensor_tile::binary_inplace_glue(a, b, meta, op);
}

Tensor& Tensor::add_(const Tensor& other) {
    binary_op_inplace<tensor_tile::BinaryOp::Add>(*this, other);
    return *this;
}

Tensor& Tensor::mul_(const Tensor& other) {
    binary_op_inplace<tensor_tile::BinaryOp::Mul>(*this, other);
    return *this;
}

Tensor& Tensor::add_scalar_(float val) {
    tensor_tile::add_scalar(*this, val);
    return *this;
}

Tensor& Tensor::mul_scalar_(float val) {
    tensor_tile::mul_scalar(*this, val);
    return *this;
}

Tensor& Tensor::clamp_(float min_val, float max_val) {
    tensor_tile::clamp(*this, min_val, max_val);
    return *this;
}

Tensor& Tensor::fill_(float val) {
    tensor_tile::fill(*this, val);
    return *this;
}

// --- Comparison ---
Tensor Tensor::operator>(float val) const {
    Tensor out = Tensor::empty(shape_, dtype_);
    if (numel_ == 0) return out;
    tensor_tile::greater_than(*this, out, val);
    return out;
}

// ============================================================================
// Reduction
// ============================================================================

static void compute_reduce_dims(const std::vector<int64_t>& shape, int dim,
                                int64_t& outer, int64_t& reduce, int64_t& inner) {
    outer = 1;
    for (int d = 0; d < dim; ++d) outer *= shape[d];
    reduce = shape[dim];
    inner = 1;
    for (int d = dim + 1; d < (int)shape.size(); ++d) inner *= shape[d];
}

Tensor Tensor::sum(int dim, bool keepdim) const {
    if (dim < 0) dim += ndim();
    if (dtype_ != DType::Float32) {
        // Convert to f32 first
        return to_f32().sum(dim, keepdim);
    }

    int64_t outer, reduce, inner;
    compute_reduce_dims(shape_, dim, outer, reduce, inner);

    std::vector<int64_t> out_shape = shape_;
    if (keepdim) {
        out_shape[dim] = 1;
    } else {
        out_shape.erase(out_shape.begin() + dim);
    }
    if (out_shape.empty()) out_shape.push_back(1);

    Tensor out = Tensor::empty(out_shape, DType::Float32);

    if (dim == ndim() - 1 && inner == 1) {
        tensor_tile::sum_reduce_last_dim(*this, out, reduce, outer);
    } else {
        tensor_tile::sum_reduce_general_glue(*this, out, outer, reduce, inner);
    }
    return out;
}

Tensor Tensor::mean(int dim, bool keepdim) const {
    if (dim < 0) dim += ndim();
    Tensor s = sum(dim, keepdim);
    float n = (float)shape_[dim < 0 ? dim + ndim() : dim];
    s.mul_scalar_(1.0f / n);
    return s;
}

Tensor Tensor::max(int dim, bool keepdim) const {
    if (dim < 0) dim += ndim();
    if (dtype_ != DType::Float32) {
        return to_f32().max(dim, keepdim);
    }

    int64_t outer, reduce, inner;
    compute_reduce_dims(shape_, dim, outer, reduce, inner);

    std::vector<int64_t> out_shape = shape_;
    if (keepdim) {
        out_shape[dim] = 1;
    } else {
        out_shape.erase(out_shape.begin() + dim);
    }
    if (out_shape.empty()) out_shape.push_back(1);

    Tensor out = Tensor::empty(out_shape, DType::Float32);

    if (dim == ndim() - 1 && inner == 1) {
        tensor_tile::max_reduce_last_dim(*this, out, reduce, outer);
    } else {
        tensor_tile::max_reduce_general_glue(*this, out, outer, reduce, inner);
    }
    return out;
}

// ============================================================================
// Copy
// ============================================================================

Tensor Tensor::clone() const {
    Tensor out = Tensor::empty(shape_, dtype_);
    if (numel_ > 0) {
        if (is_contiguous()) {
            CUDA_CHECK(cudaMemcpy(out.data_, data_, numel_ * dtype_size(dtype_),
                                   cudaMemcpyDeviceToDevice));
        } else {
            // Strided copy
            int nd = ndim();
            tensor_tile::StridedCopyMeta meta;
            meta.ndim = nd;
            for (int i = 0; i < nd; ++i) {
                meta.src_strides[i] = strides_[i];
                meta.dst_shape[i] = shape_[i];
            }
            tensor_tile::strided_copy_glue(*this, out, meta);
        }
    }
    return out;
}

void Tensor::copy_from(const Tensor& src) {
    if (numel_ != src.numel_)
        throw std::runtime_error("copy_from: size mismatch");
    if (dtype_ != src.dtype_)
        throw std::runtime_error("copy_from: dtype mismatch");
    if (numel_ > 0) {
        CUDA_CHECK(cudaMemcpy(data_, src.data_, numel_ * dtype_size(dtype_),
                               cudaMemcpyDeviceToDevice));
    }
}

// ============================================================================
// Debug
// ============================================================================

std::string Tensor::shape_str() const {
    std::ostringstream ss;
    ss << "(";
    for (int i = 0; i < ndim(); ++i) {
        if (i > 0) ss << ", ";
        ss << shape_[i];
    }
    ss << ")";
    return ss.str();
}

void Tensor::print(const std::string& name, int max_elements) const {
    std::string dtype_str;
    switch (dtype_) {
        case DType::Float32: dtype_str = "float32"; break;
        case DType::Float16: dtype_str = "float16"; break;
        case DType::Int64: dtype_str = "int64"; break;
    }
    std::cout << "Tensor";
    if (!name.empty()) std::cout << " \"" << name << "\"";
    std::cout << " shape=" << shape_str() << " dtype=" << dtype_str
              << " numel=" << numel_ << std::endl;

    if (numel_ == 0) {
        std::cout << "  (empty)" << std::endl;
        return;
    }

    // Print first few elements
    std::vector<float> vals = to_cpu_f32();
    int n = std::min((int)vals.size(), max_elements);
    std::cout << "  [";
    for (int i = 0; i < n; ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << vals[i];
    }
    if ((int)vals.size() > max_elements) std::cout << ", ...";
    std::cout << "]" << std::endl;
}

} // namespace cudasep
