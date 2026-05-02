/**
 * Test loading Qwen2.5-3B from safetensors and verify norm weights are correct
 */
#include "safetensors_parser.h"
#include "logger.h"
#include <iostream>
#include <cmath>
#include <filesystem>

using namespace ash;
namespace fs = std::filesystem;

int main(int argc, char** argv) {
    // Setup logging
    Logger::instance().set_min_level(LogLevel::INFO);
    
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <safetensors_dir>\n";
        std::cerr << "Example: " << argv[0] << " qwen2.5-3b-safetensors\n";
        return 1;
    }
    
    std::string model_dir = argv[1];
    
    // Find all safetensors files in directory
    std::vector<std::string> shard_paths;
    for (const auto& entry : fs::directory_iterator(model_dir)) {
        if (entry.path().extension() == ".safetensors") {
            shard_paths.push_back(entry.path().string());
        }
    }
    
    if (shard_paths.empty()) {
        std::cerr << "No .safetensors files found in: " << model_dir << "\n";
        return 1;
    }
    
    std::sort(shard_paths.begin(), shard_paths.end());
    std::cout << "Found " << shard_paths.size() << " shard(s)\n";
    
    // Load all shards
    std::unordered_map<std::string, Tensor> all_tensors;
    for (const auto& shard_path : shard_paths) {
        std::cout << "\nLoading shard: " << shard_path << "\n";
        
        SafeTensorsParser parser;
        if (!parser.parse(shard_path)) {
            std::cerr << "Failed to parse: " << shard_path << "\n";
            return 1;
        }
        
        // Load all tensors from this shard
        auto tensors = parser.load_all_tensors();
        for (auto& [name, tensor] : tensors) {
            all_tensors[name] = std::move(tensor);
        }
    }
    
    std::cout << "\n✅ Total tensors loaded: " << all_tensors.size() << "\n\n";
    
    // Check layer 0 and 1 norm weights (these were corrupted in GGUF)
    std::vector<std::string> check_tensors = {
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.post_attention_layernorm.weight",
        "model.layers.1.input_layernorm.weight",
        "model.layers.1.post_attention_layernorm.weight"
    };
    
    for (const auto& name : check_tensors) {
        auto it = all_tensors.find(name);
        if (it == all_tensors.end()) {
            std::cerr << "⚠️  Tensor not found: " << name << "\n";
            continue;
        }
        
        const Tensor& tensor = it->second;
        
        // Calculate L2 norm
        const float* data = tensor.data_f32();
        size_t numel = tensor.shape().numel();
        
        float l2_norm = 0.0f;
        for (size_t i = 0; i < numel; i++) {
            l2_norm += data[i] * data[i];
        }
        l2_norm = std::sqrt(l2_norm);
        
        std::cout << name << ":\n";
        std::cout << "  Shape: " << tensor.shape().to_string() << "\n";
        std::cout << "  DType: " << static_cast<int>(tensor.dtype()) << "\n";
        std::cout << "  L2 norm: " << l2_norm << "\n";
        std::cout << "  First 5 values: ";
        for (size_t i = 0; i < std::min(size_t(5), numel); i++) {
            std::cout << data[i] << " ";
        }
        std::cout << "\n\n";
    }
    
    // Expected values (from Python):
    // Layer 0 input_layernorm: L2=15.85
    // Layer 0 post_attention_layernorm: L2=18.65
    // Layer 1 input_layernorm: L2=12.51
    // Layer 1 post_attention_layernorm: L2=58.83
    
    std::cout << "Expected L2 norms (from Python):\n";
    std::cout << "  Layer 0 input_layernorm: 15.85\n";
    std::cout << "  Layer 0 post_attention_layernorm: 18.65\n";
    std::cout << "  Layer 1 input_layernorm: 12.51\n";
    std::cout << "  Layer 1 post_attention_layernorm: 58.83\n";
    
    return 0;
}
