#pragma once

#include "tensor.h"
#include <cmath>

namespace ash {

// Matrix multiplication: C = A @ B
// A: [m, k], B: [k, n] -> C: [m, n]
Tensor matmul(const Tensor& a, const Tensor& b);

// Matrix-vector multiplication: y = A @ x
// A: [m, k], x: [k] -> y: [m]
Tensor matvec(const Tensor& a, const Tensor& x);

// Element-wise operations
Tensor add(const Tensor& a, const Tensor& b);
Tensor multiply(const Tensor& a, const Tensor& b);
Tensor scale(const Tensor& a, float scalar);

// Activation functions
Tensor relu(const Tensor& x);
Tensor gelu(const Tensor& x);
Tensor silu(const Tensor& x); // SiLU (Swish)

// Softmax: exp(x_i) / sum(exp(x_j))
// Applied along last dimension
Tensor softmax(const Tensor& x);

// RMSNorm: Root Mean Square Layer Normalization
// x_norm = x / sqrt(mean(x^2) + eps) * weight
Tensor rmsnorm(const Tensor& x, const Tensor& weight, float eps = 1e-6f);

// RoPE: Rotary Position Embeddings
// Applies rotation to [seq_len, n_heads, head_dim] tensor
// pos: position indices [seq_len]
Tensor rope(const Tensor& x, const Tensor& freqs_cos, const Tensor& freqs_sin);

// Precompute RoPE frequencies
// Returns tuple of (cos, sin) tensors [max_seq_len, head_dim/2]
std::pair<Tensor, Tensor> precompute_rope_freqs(
    int max_seq_len,
    int head_dim,
    float theta = 10000.0f
);

// Attention scoring: Q @ K^T / sqrt(d_k)
// Q: [batch, n_heads, seq_len_q, head_dim]
// K: [batch, n_heads, seq_len_k, head_dim]
// -> scores: [batch, n_heads, seq_len_q, seq_len_k]
Tensor attention_scores(const Tensor& q, const Tensor& k, float scale);

// Apply attention mask (add large negative value to masked positions)
void apply_attention_mask(Tensor& scores, const Tensor& mask);

// Grouped-query attention (GQA) key/value projection
// Expands KV from n_kv_heads to n_q_heads by repeating
// Input: [batch, n_kv_heads, seq_len, head_dim]
// Output: [batch, n_q_heads, seq_len, head_dim]
Tensor expand_kv_for_gqa(const Tensor& kv, int n_q_heads, int n_kv_heads);

// Utilities for tensor operations
namespace utils {
    // Copy tensor data
    void copy(const Tensor& src, Tensor& dst);
    
    // Fill tensor with value
    void fill(Tensor& t, float value);
    
    // Print tensor (first few values)
    void print(const Tensor& t, const std::string& name = "tensor");
    
    // Compute L2 norm
    float norm(const Tensor& t);
    
    // Check if tensors are close (for testing)
    bool allclose(const Tensor& a, const Tensor& b, float rtol = 1e-5f, float atol = 1e-8f);
}

} // namespace ash
