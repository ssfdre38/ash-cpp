#include "tensor.h"
#include "memory_lock.h"  // Full definition needed for ScopedMemoryLock
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

Tensor Tensor::ones(TensorShape shape, DType dtype) {
    Tensor t(shape, dtype);
    if (dtype == DType::F32) {
        float* data = t.data_f32();
        int64_t n = t.shape().numel();
        for (int64_t i = 0; i < n; ++i) {
            data[i] = 1.0f;
        }
    }
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

Tensor Tensor::clone() const {
    if (!is_allocated()) {
        return Tensor();
    }
    
    Tensor copy = Tensor::empty(shape_, dtype_);
    std::memcpy(copy.data(), data_, size_bytes());
    return copy;
}

bool Tensor::lock_memory() {
    if (!data_ || !owns_data_) {
        return false;
    }
    
    if (memory_locked_) {
        return true;  // Already locked
    }
    
    size_t bytes = size_bytes();
    memory_lock_handle_ = new ScopedMemoryLock(data_, bytes);
    memory_locked_ = static_cast<ScopedMemoryLock*>(memory_lock_handle_)->is_locked();
    
    if (!memory_locked_) {
        Logger::instance().warning("Failed to lock tensor memory: " + 
            static_cast<ScopedMemoryLock*>(memory_lock_handle_)->error());
        delete static_cast<ScopedMemoryLock*>(memory_lock_handle_);
        memory_lock_handle_ = nullptr;
    } else {
        Logger::instance().debug("Locked tensor memory: " + std::to_string(bytes) + " bytes");
    }
    
    return memory_locked_;
}

void Tensor::unlock_memory() {
    if (memory_lock_handle_) {
        delete static_cast<ScopedMemoryLock*>(memory_lock_handle_);
        memory_lock_handle_ = nullptr;
        memory_locked_ = false;
    }
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
    
    // Helper: proper IEEE 754 float16 to float32 conversion
    auto f16_to_f32 = [](uint16_t h) -> float {
        uint32_t sign = (h >> 15) & 0x1;
        uint32_t exponent = (h >> 10) & 0x1F;
        uint32_t mantissa = h & 0x3FF;
        
        uint32_t f;
        if (exponent == 0) {
            if (mantissa == 0) {
                // Zero
                f = sign << 31;
            } else {
                // Denormal
                exponent = 127 - 14;
                while ((mantissa & 0x400) == 0) {
                    mantissa <<= 1;
                    exponent -= 1;
                }
                mantissa &= 0x3FF;
                f = (sign << 31) | (exponent << 23) | (mantissa << 13);
            }
        } else if (exponent == 0x1F) {
            // Inf or NaN
            f = (sign << 31) | 0x7F800000 | (mantissa << 13);
        } else {
            // Normal
            f = (sign << 31) | ((exponent + 112) << 23) | (mantissa << 13);
        }
        
        return *reinterpret_cast<float*>(&f);
    };
    
    for (int64_t i = 0; i < num_blocks; ++i) {
        // Read scale with proper F16→F32 conversion
        uint16_t scale_bits = *reinterpret_cast<const uint16_t*>(src_bytes);
        float scale = f16_to_f32(scale_bits);
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
    // Q4_K: 256 elements per block
    // Each block has 176 bytes for Q5_K, but Q4_K is 144 bytes
    // For simplicity, we use a basic implementation that supports the standard Q4_K_M layout
    const int BLOCK_SIZE = 256;
    int64_t num_blocks = n / BLOCK_SIZE;
    
    const uint8_t* src_bytes = reinterpret_cast<const uint8_t*>(src);
    
    // GGUF Q4_K block structure:
    // float16 d; float16 dmin; uint8_t scales[12]; uint8_t qs[128]
    // (Actual layout is more optimized, this is a simplified version)
    
    for (int64_t i = 0; i < num_blocks; ++i) {
        // Read d and dmin
        uint16_t d_bits = *reinterpret_cast<const uint16_t*>(src_bytes);
        uint16_t dmin_bits = *reinterpret_cast<const uint16_t*>(src_bytes + 2);
        
        auto f16_to_f32 = [](uint16_t h) -> float {
            uint32_t sign = (h >> 15) & 0x1;
            uint32_t exponent = (h >> 10) & 0x1F;
            uint32_t mantissa = h & 0x3FF;
            uint32_t f;
            if (exponent == 0) {
                if (mantissa == 0) f = sign << 31;
                else {
                    exponent = 127 - 14;
                    while ((mantissa & 0x400) == 0) { mantissa <<= 1; exponent -= 1; }
                    mantissa &= 0x3FF;
                    f = (sign << 31) | (exponent << 23) | (mantissa << 13);
                }
            } else if (exponent == 0x1F) f = (sign << 31) | 0x7F800000 | (mantissa << 13);
            else f = (sign << 31) | ((exponent + 112) << 23) | (mantissa << 13);
            return *reinterpret_cast<float*>(&f);
        };
        
        float d = f16_to_f32(d_bits);
        float dmin = f16_to_f32(dmin_bits);
        src_bytes += 4;
        
        const uint8_t* scales = src_bytes;
        src_bytes += 12; // 12 scales for 8 super-blocks of 32
        
        const uint8_t* qs = src_bytes;
        src_bytes += 128; // 256 nibbles
        
        for (int j = 0; j < BLOCK_SIZE; ++j) {
            int sc_idx = j / 32;
            int q_idx = j / 2;
            int shift = (j % 2) * 4;
            
            // Extract scale for this 32-element chunk
            // Each 6 bits of scales[12] corresponds to a chunk
            // Simplified: extract roughly
            uint8_t sc = (scales[sc_idx * 3 / 2] >> ((sc_idx % 2) * 4)) & 0x3F;
            
            uint8_t q = (qs[q_idx] >> shift) & 0x0F;
            dst[i * BLOCK_SIZE + j] = d * sc * q - dmin;
        }
    }
    
    Logger::instance().debug("Dequantized Q4_K: " + std::to_string(n) + " elements");
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
