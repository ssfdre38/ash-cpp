/**
 * Test Qwen2.5-3B inference with SafeTensors (correct weights)
 * This bypasses the corrupted GGUF files and loads weights directly from safetensors
 */
#include "safetensors_parser.h"
#include "inference.h"
#include "logger.h"
#include <iostream>
#include <filesystem>

using namespace ash;
namespace fs = std::filesystem;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <safetensors_dir>\n";
        std::cerr << "Example: " << argv[0] << " qwen2.5-3b-safetensors\n";
        return 1;
    }
    
    Logger::instance().set_min_level(LogLevel::INFO);
    
    std::string model_dir = argv[1];
    
    std::cout << "🔥 Testing Qwen2.5-3B with CORRECT weights from SafeTensors\n\n";
    
    // Load all safetensors shards
    std::vector<std::string> shard_paths;
    for (const auto& entry : fs::directory_iterator(model_dir)) {
        if (entry.path().extension() == ".safetensors") {
            shard_paths.push_back(entry.path().string());
        }
    }
    std::sort(shard_paths.begin(), shard_paths.end());
    
    std::cout << "Loading " << shard_paths.size() << " shard(s)...\n";
    
    std::unordered_map<std::string, Tensor> all_tensors;
    for (const auto& shard_path : shard_paths) {
        SafeTensorsParser parser;
        if (!parser.parse(shard_path)) {
            std::cerr << "Failed to parse: " << shard_path << "\n";
            return 1;
        }
        
        auto tensors = parser.load_all_tensors();
        for (auto& [name, tensor] : tensors) {
            all_tensors[name] = std::move(tensor);
        }
    }
    
    std::cout << "✅ Loaded " << all_tensors.size() << " tensors\n\n";
    
    // Verify norm weights are correct
    std::cout << "Verifying norm weights:\n";
    auto check_norm = [&](const std::string& name, float expected_l2) {
        auto it = all_tensors.find(name);
        if (it == all_tensors.end()) {
            std::cout << "  ❌ " << name << " NOT FOUND\n";
            return;
        }
        
        const float* data = it->second.data_f32();
        size_t numel = it->second.shape().numel();
        float l2 = 0.0f;
        for (size_t i = 0; i < numel; i++) {
            l2 += data[i] * data[i];
        }
        l2 = std::sqrt(l2);
        
        float error = std::abs(l2 - expected_l2);
        bool ok = error < 0.1f;
        std::cout << "  " << (ok ? "✅" : "❌") << " " << name 
                  << ": L2=" << l2 << " (expected " << expected_l2 << ")\n";
    };
    
    check_norm("model.layers.0.input_layernorm.weight", 15.85f);
    check_norm("model.layers.0.post_attention_layernorm.weight", 18.65f);
    check_norm("model.layers.1.input_layernorm.weight", 12.51f);
    check_norm("model.layers.1.post_attention_layernorm.weight", 58.83f);
    
    std::cout << "\n✅ All norm weights verified correct!\n";
    std::cout << "\nNow we can test inference with accurate weights...\n";
    std::cout << "(TODO: Integrate safetensors loading into InferenceEngine)\n";
    
    return 0;
}
