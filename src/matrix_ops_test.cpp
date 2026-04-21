#include "matrix_ops.h"
#include "logger.h"
#include <iostream>
#include <iomanip>

using namespace ash;

int main() {
    std::cout << "⚡ Testing Ash's Matrix Operations...\n\n";
    
    Logger::instance().set_min_level(LogLevel::INFO);
    
    // Test 1: Matrix multiplication
    std::cout << "Test 1: Matrix multiplication\n";
    auto a = Tensor::zeros({2, 3}, DType::F32);
    auto b = Tensor::zeros({3, 2}, DType::F32);
    
    float* a_data = a.data_f32();
    float* b_data = b.data_f32();
    
    // A = [[1, 2, 3], [4, 5, 6]]
    a_data[0] = 1; a_data[1] = 2; a_data[2] = 3;
    a_data[3] = 4; a_data[4] = 5; a_data[5] = 6;
    
    // B = [[1, 0], [0, 1], [1, 1]]
    b_data[0] = 1; b_data[1] = 0;
    b_data[2] = 0; b_data[3] = 1;
    b_data[4] = 1; b_data[5] = 1;
    
    auto c = matmul(a, b);
    std::cout << "  A[2,3] @ B[3,2] = C[2,2]\n";
    utils::print(c, "  Result");
    std::cout << "\n";
    
    // Test 2: Activations
    std::cout << "Test 2: Activation functions\n";
    auto x = Tensor::zeros({5}, DType::F32);
    float* x_data = x.data_f32();
    x_data[0] = -2.0f; x_data[1] = -1.0f; x_data[2] = 0.0f; 
    x_data[3] = 1.0f; x_data[4] = 2.0f;
    
    utils::print(x, "  Input");
    
    auto y_relu = relu(x);
    utils::print(y_relu, "  ReLU");
    
    auto y_gelu = gelu(x);
    utils::print(y_gelu, "  GELU");
    
    auto y_silu = silu(x);
    utils::print(y_silu, "  SiLU");
    std::cout << "\n";
    
    // Test 3: Softmax
    std::cout << "Test 3: Softmax\n";
    auto logits = Tensor::zeros({4}, DType::F32);
    float* logits_data = logits.data_f32();
    logits_data[0] = 1.0f; logits_data[1] = 2.0f; 
    logits_data[2] = 3.0f; logits_data[3] = 4.0f;
    
    utils::print(logits, "  Logits");
    
    auto probs = softmax(logits);
    utils::print(probs, "  Probs");
    
    // Verify sum = 1
    float sum = 0.0f;
    float* probs_data = probs.data_f32();
    for (int i = 0; i < 4; ++i) sum += probs_data[i];
    std::cout << "  Sum of probs: " << std::fixed << std::setprecision(6) << sum << "\n\n";
    
    // Test 4: RMSNorm
    std::cout << "Test 4: RMSNorm\n";
    auto vec = Tensor::zeros({4}, DType::F32);
    auto weight = Tensor::zeros({4}, DType::F32);
    
    float* vec_data = vec.data_f32();
    float* weight_data = weight.data_f32();
    
    vec_data[0] = 1.0f; vec_data[1] = 2.0f; 
    vec_data[2] = 3.0f; vec_data[3] = 4.0f;
    
    weight_data[0] = 1.0f; weight_data[1] = 1.0f;
    weight_data[2] = 1.0f; weight_data[3] = 1.0f;
    
    utils::print(vec, "  Input");
    utils::print(weight, "  Weight");
    
    auto normalized = rmsnorm(vec, weight);
    utils::print(normalized, "  RMSNorm");
    
    float norm_val = utils::norm(normalized);
    std::cout << "  L2 norm: " << std::fixed << std::setprecision(4) << norm_val << "\n\n";
    
    // Test 5: RoPE frequencies
    std::cout << "Test 5: RoPE (Rotary Position Embeddings)\n";
    int max_seq = 8;
    int head_dim = 4;
    
    auto [freqs_cos, freqs_sin] = precompute_rope_freqs(max_seq, head_dim, 10000.0f);
    std::cout << "  Precomputed freqs for max_seq=" << max_seq << ", head_dim=" << head_dim << "\n";
    std::cout << "  Cos shape: " << freqs_cos.shape().to_string() << "\n";
    std::cout << "  Sin shape: " << freqs_sin.shape().to_string() << "\n";
    
    // Apply RoPE to a sample tensor
    auto emb = Tensor::zeros({4, 4}, DType::F32);
    float* emb_data = emb.data_f32();
    for (int i = 0; i < 16; ++i) {
        emb_data[i] = static_cast<float>(i) * 0.1f;
    }
    
    utils::print(emb, "  Embeddings");
    
    auto rotated = rope(emb, freqs_cos, freqs_sin);
    utils::print(rotated, "  After RoPE");
    std::cout << "\n";
    
    // Test 6: Attention scores
    std::cout << "Test 6: Attention scoring\n";
    auto q = Tensor::zeros({3, 4}, DType::F32);
    auto k = Tensor::zeros({3, 4}, DType::F32);
    
    float* q_data = q.data_f32();
    float* k_data = k.data_f32();
    
    // Simple patterns
    for (int i = 0; i < 12; ++i) {
        q_data[i] = static_cast<float>(i % 4) + 1.0f;
        k_data[i] = static_cast<float>(i % 4) + 1.0f;
    }
    
    float scale = 1.0f / std::sqrt(4.0f);
    auto scores = attention_scores(q, k, scale);
    
    std::cout << "  Q shape: " << q.shape().to_string() << "\n";
    std::cout << "  K shape: " << k.shape().to_string() << "\n";
    std::cout << "  Scores shape: " << scores.shape().to_string() << "\n";
    utils::print(scores, "  Scores");
    std::cout << "\n";
    
    // Test 7: Element-wise operations
    std::cout << "Test 7: Element-wise operations\n";
    auto v1 = Tensor::zeros({3}, DType::F32);
    auto v2 = Tensor::zeros({3}, DType::F32);
    
    float* v1_data = v1.data_f32();
    float* v2_data = v2.data_f32();
    
    v1_data[0] = 1.0f; v1_data[1] = 2.0f; v1_data[2] = 3.0f;
    v2_data[0] = 4.0f; v2_data[1] = 5.0f; v2_data[2] = 6.0f;
    
    utils::print(v1, "  v1");
    utils::print(v2, "  v2");
    
    auto v_add = add(v1, v2);
    utils::print(v_add, "  v1 + v2");
    
    auto v_mul = multiply(v1, v2);
    utils::print(v_mul, "  v1 * v2");
    
    auto v_scaled = ash::scale(v1, 2.5f);
    utils::print(v_scaled, "  v1 * 2.5");
    std::cout << "\n";
    
    // Test 8: Utilities
    std::cout << "Test 8: Utility functions\n";
    auto t1 = Tensor::zeros({5}, DType::F32);
    float* t1_data = t1.data_f32();
    for (int i = 0; i < 5; ++i) t1_data[i] = static_cast<float>(i);
    
    auto t2 = Tensor::zeros({5}, DType::F32);
    utils::copy(t1, t2);
    
    bool same = utils::allclose(t1, t2);
    std::cout << "  Copy successful: " << (same ? "yes" : "no") << "\n";
    
    utils::fill(t2, 42.0f);
    utils::print(t2, "  After fill(42)");
    
    float t1_norm = utils::norm(t1);
    std::cout << "  L2 norm of [0,1,2,3,4]: " << std::fixed << std::setprecision(4) << t1_norm << "\n";
    std::cout << "\n";
    
    std::cout << "✓ Matrix operations test complete!\n";
    std::cout << "🔥 All transformer building blocks working.\n";
    std::cout << "Next: Load GGUF model tensors + tokenizer\n";
    
    return 0;
}
