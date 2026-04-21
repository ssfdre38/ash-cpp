#include "tensor.h"
#include "logger.h"
#include <iostream>
#include <iomanip>
#include <cstring>

using namespace ash;

int main() {
    std::cout << "🔢 Testing Ash's Tensor System...\n\n";
    
    Logger::instance().set_min_level(LogLevel::DEBUG);
    
    // Test 1: Create F32 tensor
    std::cout << "Test 1: Creating F32 tensors\n";
    auto t1 = Tensor::empty({3, 4}, DType::F32);
    std::cout << "  " << t1.info() << "\n";
    std::cout << "  Shape: " << t1.shape().to_string() << "\n";
    std::cout << "  Elements: " << t1.shape().numel() << "\n";
    std::cout << "  Size: " << t1.size_bytes() << " bytes\n\n";
    
    // Test 2: Create and fill tensor
    std::cout << "Test 2: Creating and filling tensor\n";
    auto t2 = Tensor::zeros({2, 3}, DType::F32);
    float* data = t2.data_f32();
    for (int i = 0; i < 6; ++i) {
        data[i] = static_cast<float>(i) * 0.5f;
    }
    
    std::cout << "  Data: ";
    for (int i = 0; i < 6; ++i) {
        std::cout << std::fixed << std::setprecision(1) << data[i] << " ";
    }
    std::cout << "\n\n";
    
    // Test 3: Create from data
    std::cout << "Test 3: Create tensor from existing data\n";
    std::vector<float> vec = {1.0f, 2.0f, 3.0f, 4.0f};
    auto t3 = Tensor::from_data(vec.data(), {2, 2}, DType::F32);
    std::cout << "  " << t3.info() << "\n";
    std::cout << "  Data: ";
    float* t3_data = t3.data_f32();
    for (int i = 0; i < 4; ++i) {
        std::cout << t3_data[i] << " ";
    }
    std::cout << "\n\n";
    
    // Test 4: Tensor move semantics
    std::cout << "Test 4: Testing move semantics\n";
    auto t4 = Tensor::empty({5, 5}, DType::F32);
    std::cout << "  Before move: " << t4.info() << "\n";
    
    auto t5 = std::move(t4);
    std::cout << "  After move to t5: " << t5.info() << "\n";
    std::cout << "  Original t4: " << t4.info() << "\n\n";
    
    // Test 5: Different dtypes
    std::cout << "Test 5: Testing different data types\n";
    auto t_f32 = Tensor::empty({10, 10}, DType::F32);
    auto t_f16 = Tensor::empty({10, 10}, DType::F16);
    auto t_q8  = Tensor::empty({256}, DType::Q8_0);
    auto t_q4k = Tensor::empty({256}, DType::Q4_K);
    
    std::cout << "  F32:  " << t_f32.size_bytes() << " bytes for " << t_f32.shape().numel() << " elements\n";
    std::cout << "  F16:  " << t_f16.size_bytes() << " bytes for " << t_f16.shape().numel() << " elements\n";
    std::cout << "  Q8_0: " << t_q8.size_bytes() << " bytes for " << t_q8.shape().numel() << " elements\n";
    std::cout << "  Q4_K: " << t_q4k.size_bytes() << " bytes for " << t_q4k.shape().numel() << " elements\n";
    std::cout << "  Compression: F32→Q4_K = " << std::fixed << std::setprecision(1) 
              << (static_cast<float>(t_f32.size_bytes()) / t_q4k.size_bytes()) << "x smaller\n\n";
    
    // Test 6: Q8_0 dequantization (simplified test)
    std::cout << "Test 6: Q8_0 dequantization test\n";
    
    // Create a simple Q8_0 tensor (1 block = 32 elements)
    // Block structure: uint16 scale + 32x int8
    const int BLOCK_SIZE = 32;
    std::vector<uint8_t> q8_data(34); // 2 bytes scale + 32 bytes data
    
    // Set scale (simplified)
    uint16_t scale = 16384; // ~0.5 in simplified F16
    std::memcpy(q8_data.data(), &scale, 2);
    
    // Set quantized values
    int8_t* values = reinterpret_cast<int8_t*>(q8_data.data() + 2);
    for (int i = 0; i < BLOCK_SIZE; ++i) {
        values[i] = static_cast<int8_t>(i - 16); // -16 to +15
    }
    
    auto q8_tensor = Tensor::from_data(q8_data.data(), {BLOCK_SIZE}, DType::Q8_0);
    std::cout << "  Q8_0 tensor: " << q8_tensor.info() << "\n";
    
    auto dequant = q8_tensor.dequantize();
    std::cout << "  Dequantized: " << dequant.info() << "\n";
    std::cout << "  First 8 values: ";
    float* dq_data = dequant.data_f32();
    for (int i = 0; i < 8; ++i) {
        std::cout << std::fixed << std::setprecision(2) << dq_data[i] << " ";
    }
    std::cout << "\n\n";
    
    // Test 7: Multi-dimensional tensors
    std::cout << "Test 7: Multi-dimensional tensors\n";
    auto t3d = Tensor::zeros({2, 3, 4}, DType::F32);
    std::cout << "  3D tensor: " << t3d.info() << "\n";
    std::cout << "  Shape: " << t3d.shape().to_string() << "\n";
    std::cout << "  Dimensions: " << t3d.shape().ndim() << "\n";
    std::cout << "  Dim 0 size: " << t3d.shape().size(0) << "\n";
    std::cout << "  Dim 1 size: " << t3d.shape().size(1) << "\n";
    std::cout << "  Dim 2 size: " << t3d.shape().size(2) << "\n\n";
    
    std::cout << "✓ Tensor system test complete!\n";
    std::cout << "🔥 Ready to load model tensors from GGUF files.\n";
    std::cout << "Next: Matrix operations (matmul, softmax, RMSNorm)\n";
    
    return 0;
}
