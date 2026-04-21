#include "tensor.h"
#include "logger.h"
#include <cstring>
#include <cstdlib>
#include <sstream>
#include <stdexcept>
#include <cmath>

#ifdef _WIN32
#include <malloc.h> // For _aligned_malloc
#endif

namespace ash {

// TensorShape implementation
int64_t TensorShape::numel() const {
    if (dims.empty()) return 0;
    int64_t n = 1;
    for (auto d : dims) {
        n *= d;
    }
    return n;
}

std::string TensorShape::to_string() const {
    std::stringstream ss;
    ss << "[";
    for (size_t i = 0; i < dims.size(); ++i) {
        if (i > 0) ss << ", ";
        ss << dims[i];
    }
    ss << "]";
    return ss.str();
}

// Tensor implementation
Tensor::Tensor(TensorShape shape, DType dtype)
    : shape_(shape), dtype_(dtype), data_(nullptr), owns_data_(true) {
    allocate();
}

Tensor::~Tensor() {
    free();
}

Tensor::Tensor(Tensor&& other) noexcept
    : shape_(std::move(other.shape_))
    , dtype_(other.dtype_)
    , data_(other.data_)
    , owns_data_(other.owns_data_) {
    other.data_ = nullptr;
    other.owns_data_ = false;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        free();
        shape_ = std::move(other.shape_);
        dtype_ = other.dtype_;
        data_ = other.data_;
        owns_data_ = other.owns_data_;
        other.data_ = nullptr;
        other.owns_data_ = false;
    }
    return *this;
}

Tensor Tensor::from_data(const void* data, TensorShape shape, DType dtype) {
    Tensor t(shape, dtype);
    std::memcpy(t.data_, data, t.size_bytes());
    return t;
}

Tensor Tensor::empty(TensorShape shape, DType dtype) {
    return Tensor(shape, dtype);
}

Tensor Tensor::zeros(TensorShape shape, DType dtype) {
    Tensor t(shape, dtype);
    std::memset(t.data_, 0, t.size_bytes());
    return t;
}

size_t Tensor::size_bytes() const {
    int64_t n = shape_.numel();
    
    // For quantized types, size depends on block structure
    switch (dtype_) {
        case DType::F32: return n * 4;
        case DType::F16: return n * 2;
        case DType::I32: return n * 4;
        case DType::I16: return n * 2;
        case DType::I8:  return n;
        
        // Quantized types (approximate - actual depends on block structure)
        case DType::Q8_0: return (n / 32) * 34; // 32 elements per block + 2 bytes metadata
        case DType::Q4_K: return (n / 256) * 144; // K-quant block structure
        case DType::Q5_K: return (n / 256) * 176;
        case DType::Q6_K: return (n / 256) * 210;
        
        default:
            throw std::runtime_error("Unknown dtype");
    }
}

void Tensor::allocate() {
    if (data_) return;
    
    size_t bytes = size_bytes();
    if (bytes == 0) return;
    
    // Align to 64 bytes for SIMD
    #ifdef _WIN32
        data_ = _aligned_malloc(bytes, 64);
    #else
        data_ = std::aligned_alloc(64, bytes);
    #endif
    
    if (!data_) {
        throw std::runtime_error("Failed to allocate tensor memory");
    }
    
    owns_data_ = true;
}

void Tensor::free() {
    if (data_ && owns_data_) {
        #ifdef _WIN32
            _aligned_free(data_);
        #else
            std::free(data_);
        #endif
        data_ = nullptr;
    }
}

Tensor Tensor::dequantize() const {
    if (dtype_ == DType::F32) {
        // Already F32, just copy
        return Tensor::from_data(data_, shape_, DType::F32);
    }
    
    // Create F32 output tensor
    Tensor result = Tensor::empty(shape_, DType::F32);
    int64_t n = shape_.numel();
    
    // Dequantize based on type
    switch (dtype_) {
        case DType::Q8_0:
            dequantize_q8_0(data_, result.data_f32(), n);
            break;
        case DType::Q4_K:
            dequantize_q4_k(data_, result.data_f32(), n);
            break;
        case DType::Q5_K:
            dequantize_q5_k(data_, result.data_f32(), n);
            break;
        case DType::Q6_K:
            dequantize_q6_k(data_, result.data_f32(), n);
            break;
        case DType::F16:
            // TODO: F16 → F32 conversion
            throw std::runtime_error("F16 dequantization not yet implemented");
        default:
            throw std::runtime_error("Cannot dequantize this dtype");
    }
    
    return result;
}

std::string Tensor::info() const {
    std::stringstream ss;
    ss << "Tensor(shape=" << shape_.to_string();
    ss << ", dtype=" << dtype_name(dtype_);
    ss << ", bytes=" << size_bytes();
    ss << ", allocated=" << (data_ != nullptr ? "yes" : "no");
    ss << ")";
    return ss.str();
}

// Dtype utilities
size_t dtype_size(DType dtype) {
    switch (dtype) {
        case DType::F32: return 4;
        case DType::F16: return 2;
        case DType::I32: return 4;
        case DType::I16: return 2;
        case DType::I8:  return 1;
        // Quantized types don't have a fixed per-element size
        default: return 0;
    }
}

const char* dtype_name(DType dtype) {
    switch (dtype) {
        case DType::F32: return "float32";
        case DType::F16: return "float16";
        case DType::Q8_0: return "Q8_0";
        case DType::Q4_K: return "Q4_K";
        case DType::Q5_K: return "Q5_K";
        case DType::Q6_K: return "Q6_K";
        case DType::I32: return "int32";
        case DType::I16: return "int16";
        case DType::I8:  return "int8";
        default: return "unknown";
    }
}

// Dequantization implementations (simplified - production needs optimized SIMD versions)

void dequantize_q8_0(const void* src, float* dst, int64_t n) {
    // Q8_0: 32 elements per block
    // Block structure: float16 scale + 32x int8 values
    // Each block: 2 bytes (scale) + 32 bytes (values) = 34 bytes
    
    const int BLOCK_SIZE = 32;
    int64_t num_blocks = n / BLOCK_SIZE;
    
    const uint8_t* src_bytes = reinterpret_cast<const uint8_t*>(src);
    
    for (int64_t i = 0; i < num_blocks; ++i) {
        // Read scale (simplified - actual F16 decode needed)
        uint16_t scale_bits = *reinterpret_cast<const uint16_t*>(src_bytes);
        float scale = static_cast<float>(scale_bits) / 32768.0f; // Simplified F16→F32
        src_bytes += 2;
        
        // Read and dequantize 32 int8 values
        const int8_t* values = reinterpret_cast<const int8_t*>(src_bytes);
        for (int j = 0; j < BLOCK_SIZE; ++j) {
            dst[i * BLOCK_SIZE + j] = scale * static_cast<float>(values[j]);
        }
        src_bytes += BLOCK_SIZE;
    }
    
    Logger::instance().debug("Dequantized Q8_0: " + std::to_string(n) + " elements");
}

void dequantize_q4_k(const void* src, float* dst, int64_t n) {
    // Q4_K: Complex k-quant structure
    // TODO: Implement proper k-quant dequantization
    // For now, just zero-fill (placeholder)
    std::memset(dst, 0, n * sizeof(float));
    Logger::instance().debug("Q4_K dequantization: placeholder (zeroed " + std::to_string(n) + " elements)");
}

void dequantize_q5_k(const void* src, float* dst, int64_t n) {
    // Q5_K: Complex k-quant structure
    // TODO: Implement proper k-quant dequantization
    std::memset(dst, 0, n * sizeof(float));
    Logger::instance().debug("Q5_K dequantization: placeholder (zeroed " + std::to_string(n) + " elements)");
}

void dequantize_q6_k(const void* src, float* dst, int64_t n) {
    // Q6_K: Complex k-quant structure
    // TODO: Implement proper k-quant dequantization
    std::memset(dst, 0, n * sizeof(float));
    Logger::instance().debug("Q6_K dequantization: placeholder (zeroed " + std::to_string(n) + " elements)");
}

} // namespace ash
