#include "inference.h"
#include "logger.h"
#include <iostream>
#include <cassert>

using namespace ash;

// Initialize logger once
void setup_logger() {
    Logger::instance().set_log_file("inference_test.log");
    Logger::instance().set_min_level(LogLevel::INFO);
}

// Test inference engine basics
void test_inference_config() {
    std::cout << "Test: Inference config creation..." << std::flush;
    
    ModelConfig config;
    config.n_heads = 8;
    config.n_kv_heads = 4;
    config.head_dim = 64;
    config.max_seq_len = 2048;
    config.rope_theta = 10000.0f;
    
    AttentionConfig attn_config = config.attention_config();
    assert(attn_config.n_heads == 8);
    assert(attn_config.n_kv_heads == 4);
    assert(attn_config.head_dim == 64);
    assert(attn_config.use_kv_cache == true);
    
    std::cout << " PASS" << std::endl;
}

void test_sampling_config() {
    std::cout << "Test: Sampling config defaults..." << std::flush;
    
    SamplingConfig config;
    assert(config.temperature == 0.7f);
    assert(config.top_p == 0.9f);
    assert(config.top_k == 40);
    assert(config.max_tokens == 512);
    assert(config.use_sampling == true);
    
    std::cout << " PASS" << std::endl;
}

void test_temperature_application() {
    std::cout << "Test: Temperature application..." << std::flush;
    
    Tensor logits({5}, DType::F32);
    float* data = logits.data_f32();
    data[0] = 1.0f;
    data[1] = 2.0f;
    data[2] = 3.0f;
    data[3] = 4.0f;
    data[4] = 5.0f;
    
    inference_utils::apply_temperature(logits, 2.0f);
    
    // After temperature=2.0, values should be halved
    assert(std::abs(data[0] - 0.5f) < 1e-5f);
    assert(std::abs(data[1] - 1.0f) < 1e-5f);
    assert(std::abs(data[2] - 1.5f) < 1e-5f);
    assert(std::abs(data[3] - 2.0f) < 1e-5f);
    assert(std::abs(data[4] - 2.5f) < 1e-5f);
    
    std::cout << " PASS" << std::endl;
}

void test_top_k_indices() {
    std::cout << "Test: Top-K indices..." << std::flush;
    
    Tensor logits({10}, DType::F32);
    float* data = logits.data_f32();
    for (int i = 0; i < 10; i++) {
        data[i] = float(i);  // 0, 1, 2, ..., 9
    }
    
    auto top_3 = inference_utils::top_k_indices(logits, 3);
    assert(top_3.size() == 3);
    assert(top_3[0] == 9);  // Highest
    assert(top_3[1] == 8);
    assert(top_3[2] == 7);
    
    std::cout << " PASS" << std::endl;
}

void test_greedy_sampling() {
    std::cout << "Test: Greedy sampling..." << std::flush;
    
    InferenceEngine engine;
    
    Tensor logits({5}, DType::F32);
    float* data = logits.data_f32();
    data[0] = 1.0f;
    data[1] = 3.0f;
    data[2] = 5.0f;  // Highest
    data[3] = 2.0f;
    data[4] = 4.0f;
    
    TokenID token = engine.sample_greedy(logits);
    assert(token == 2);  // Index of highest value
    
    std::cout << " PASS" << std::endl;
}

void test_stop_token_check() {
    std::cout << "Test: Stop token checking..." << std::flush;
    
    std::vector<TokenID> stop_tokens = {1, 2, 3};
    
    assert(inference_utils::is_stop_token(1, stop_tokens) == true);
    assert(inference_utils::is_stop_token(2, stop_tokens) == true);
    assert(inference_utils::is_stop_token(3, stop_tokens) == true);
    assert(inference_utils::is_stop_token(4, stop_tokens) == false);
    assert(inference_utils::is_stop_token(0, stop_tokens) == false);
    
    std::cout << " PASS" << std::endl;
}

void test_inference_not_loaded() {
    std::cout << "Test: Inference without loaded model..." << std::flush;
    
    InferenceEngine engine;
    assert(engine.is_loaded() == false);
    
    // Should fail gracefully
    std::vector<TokenID> tokens = {1, 2, 3};
    Tensor result = engine.forward(tokens);
    assert(result.size() == 0);  // Empty tensor on failure
    
    std::cout << " PASS" << std::endl;
}

void test_model_load_failure() {
    std::cout << "Test: Model load with invalid file..." << std::flush;
    
    InferenceEngine engine;
    bool loaded = engine.load_model("nonexistent_model.gguf");
    assert(loaded == false);
    assert(engine.is_loaded() == false);
    
    std::cout << " PASS" << std::endl;
}

int main() {
    setup_logger();
    
    std::cout << "\n=== Inference Engine Tests ===" << std::endl;
    
    test_inference_config();
    test_sampling_config();
    test_temperature_application();
    test_top_k_indices();
    test_greedy_sampling();
    test_stop_token_check();
    test_inference_not_loaded();
    test_model_load_failure();
    
    std::cout << "\n✓ All inference tests passed!" << std::endl;
    return 0;
}
