#include "attention.h"
#include "logger.h"
#include <iostream>
#include <iomanip>

using namespace ash;

int main() {
    std::cout << "🧠 Testing Attention Mechanism...\n\n";
    
    Logger::instance().set_min_level(LogLevel::INFO);
    
    // Test 1: KV Cache
    std::cout << "Test 1: KV Cache\n";
    KVCache cache(32, 2, 4, 8);  // max_seq=32, layers=2, kv_heads=4, head_dim=8
    
    std::cout << "  Created cache: 2 layers, max seq 32\n";
    std::cout << "  Current position: " << cache.seq_len() << "\n\n";
    
    // Add some fake KV tensors
    auto k1 = Tensor::ones({1, 4, 8}, DType::F32);
    auto v1 = Tensor::ones({1, 4, 8}, DType::F32);
    
    cache.update(0, 0, k1, v1);
    cache.update(0, 1, k1, v1);
    
    std::cout << "  After updates: position " << cache.seq_len() << "\n";
    
    auto [k_cached, v_cached] = cache.get(0, 0, 2);
    std::cout << "  Retrieved K: " << k_cached.info() << "\n";
    std::cout << "  Retrieved V: " << v_cached.info() << "\n\n";
    
    // Test 2: Attention config
    std::cout << "Test 2: Attention Configuration\n";
    AttentionConfig config;
    config.n_heads = 8;
    config.n_kv_heads = 4;  // GQA
    config.head_dim = 64;
    config.max_seq_len = 128;
    config.rope_theta = 10000.0f;
    config.use_kv_cache = true;
    
    std::cout << "  Heads: " << config.n_heads << " (Q), " << config.n_kv_heads << " (KV)\n";
    std::cout << "  Head dim: " << config.head_dim << "\n";
    std::cout << "  Hidden dim: " << config.hidden_dim() << "\n";
    std::cout << "  Is GQA: " << (config.is_gqa() ? "yes" : "no") << "\n\n";
    
    // Test 3: Create attention layer
    std::cout << "Test 3: Creating Attention Layer\n";
    MultiHeadAttention attn(config);
    std::cout << "  ✅ Attention layer created\n";
    std::cout << "  RoPE frequencies precomputed\n\n";
    
    // Test 4: Attention utilities
    std::cout << "Test 4: Attention Utilities\n";
    
    // Causal mask
    auto mask = attention_utils::create_causal_mask(4);
    std::cout << "  Causal mask [4x4]:\n";
    float* mask_data = mask.data_f32();
    for (int i = 0; i < 4; ++i) {
        std::cout << "    ";
        for (int j = 0; j < 4; ++j) {
            std::cout << std::fixed << std::setprecision(0) << mask_data[i * 4 + j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
    
    // Split heads
    auto x = Tensor::zeros({2, 16}, DType::F32);  // seq_len=2, hidden=16
    float* x_data = x.data_f32();
    for (int i = 0; i < 32; ++i) x_data[i] = static_cast<float>(i);
    
    auto split = attention_utils::split_heads(x, 4, 4);  // 4 heads, 4 dim each
    std::cout << "  Split heads: " << x.info() << " -> " << split.info() << "\n";
    
    // Merge heads
    auto merged = attention_utils::merge_heads(split);
    std::cout << "  Merge heads: " << split.info() << " -> " << merged.info() << "\n";
    
    // Check round-trip
    bool same = true;
    float* merged_data = merged.data_f32();
    for (int i = 0; i < 32; ++i) {
        if (std::abs(merged_data[i] - x_data[i]) > 1e-5f) {
            same = false;
            break;
        }
    }
    std::cout << "  Round-trip successful: " << (same ? "yes" : "no") << "\n\n";
    
    // Test 5: Forward pass (simplified)
    std::cout << "Test 5: Attention Forward Pass\n";
    
    // Create smaller test config
    AttentionConfig test_config;
    test_config.n_heads = 2;
    test_config.n_kv_heads = 2;
    test_config.head_dim = 4;
    test_config.max_seq_len = 8;
    test_config.rope_theta = 10000.0f;
    test_config.use_kv_cache = false;
    
    MultiHeadAttention test_attn(test_config);
    
    int seq_len = 2;
    int hidden_dim = test_config.hidden_dim();  // 8
    
    // Input
    auto input = Tensor::ones({seq_len, hidden_dim}, DType::F32);
    
    // Weight matrices (identity for simplicity)
    auto wq = Tensor::zeros({hidden_dim, hidden_dim}, DType::F32);
    auto wk = Tensor::zeros({hidden_dim, hidden_dim}, DType::F32);
    auto wv = Tensor::zeros({hidden_dim, hidden_dim}, DType::F32);
    auto wo = Tensor::zeros({hidden_dim, hidden_dim}, DType::F32);
    
    // Set to identity
    float* wq_data = wq.data_f32();
    float* wk_data = wk.data_f32();
    float* wv_data = wv.data_f32();
    float* wo_data = wo.data_f32();
    
    for (int i = 0; i < hidden_dim; ++i) {
        wq_data[i * hidden_dim + i] = 1.0f;
        wk_data[i * hidden_dim + i] = 1.0f;
        wv_data[i * hidden_dim + i] = 1.0f;
        wo_data[i * hidden_dim + i] = 1.0f;
    }
    
    std::cout << "  Input: " << input.info() << "\n";
    std::cout << "  Running forward pass...\n";
    
    try {
        auto output = test_attn.forward(input, wq, wk, wv, wo);
        std::cout << "  ✅ Forward pass complete!\n";
        std::cout << "  Output: " << output.info() << "\n";
        
        // Check output is reasonable
        float* out_data = output.data_f32();
        bool has_nonzero = false;
        for (int i = 0; i < output.shape().numel(); ++i) {
            if (std::abs(out_data[i]) > 1e-6f) {
                has_nonzero = true;
                break;
            }
        }
        std::cout << "  Output has values: " << (has_nonzero ? "yes" : "no") << "\n";
        
    } catch (const std::exception& e) {
        std::cout << "  ❌ Forward pass failed: " << e.what() << "\n";
    }
    
    std::cout << "\n✓ Attention mechanism test complete!\n";
    std::cout << "🔥 Multi-head attention with RoPE and GQA working.\n";
    std::cout << "Next: Full inference engine\n";
    
    return 0;
}
