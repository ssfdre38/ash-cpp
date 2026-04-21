#pragma once

#include "tensor.h"
#include "matrix_ops.h"
#include <memory>
#include <vector>
#include <unordered_map>

namespace ash {

// KV cache for efficient generation
// Stores past key/value tensors to avoid recomputation
class KVCache {
public:
    KVCache(int max_seq_len, int n_layers, int n_kv_heads, int head_dim);
    ~KVCache();
    
    // Add key/value for a layer at current position
    void update(int layer, int pos, const Tensor& key, const Tensor& value);
    
    // Get cached keys/values for a layer up to position
    std::pair<Tensor, Tensor> get(int layer, int start_pos, int end_pos);
    
    // Clear cache (for new sequences)
    void clear();
    
    // Get current sequence length
    int seq_len() const { return current_pos_; }
    
    // Reset to position (for speculative decoding, etc.)
    void reset_to(int pos);

private:
    int max_seq_len_;
    int n_layers_;
    int n_kv_heads_;
    int head_dim_;
    int current_pos_ = 0;
    
    // Cache storage: [layer][kv][seq_pos, n_kv_heads, head_dim]
    std::vector<std::pair<Tensor, Tensor>> cache_; // One per layer
};

// Multi-head attention configuration
struct AttentionConfig {
    int n_heads;           // Number of query heads
    int n_kv_heads;        // Number of key/value heads (for GQA)
    int head_dim;          // Dimension per head
    int max_seq_len;       // Maximum sequence length
    float rope_theta;      // RoPE frequency base (10000 for Gemma)
    bool use_kv_cache;     // Enable KV caching for generation
    
    int hidden_dim() const { return n_heads * head_dim; }
    int kv_dim() const { return n_kv_heads * head_dim; }
    bool is_gqa() const { return n_kv_heads < n_heads; }
};

// Multi-head attention layer
class MultiHeadAttention {
public:
    explicit MultiHeadAttention(const AttentionConfig& config);
    ~MultiHeadAttention();
    
    // Forward pass
    // x: [seq_len, hidden_dim]
    // pos: starting position in sequence (for KV cache)
    // Returns: [seq_len, hidden_dim]
    Tensor forward(
        const Tensor& x,
        const Tensor& wq,      // Query weights [hidden_dim, hidden_dim]
        const Tensor& wk,      // Key weights [hidden_dim, kv_dim]
        const Tensor& wv,      // Value weights [hidden_dim, kv_dim]
        const Tensor& wo,      // Output weights [hidden_dim, hidden_dim]
        int pos = 0,
        KVCache* kv_cache = nullptr
    );
    
    // Get config
    const AttentionConfig& config() const { return config_; }

private:
    AttentionConfig config_;
    
    // Precomputed RoPE frequencies
    Tensor rope_freqs_cos_;
    Tensor rope_freqs_sin_;
    
    // Helper: apply RoPE to queries/keys
    void apply_rope(Tensor& xq, Tensor& xk, int pos);
    
    // Helper: compute attention scores and apply softmax
    Tensor compute_attention(
        const Tensor& q,  // [seq_len, n_heads, head_dim]
        const Tensor& k,  // [seq_len_k, n_kv_heads, head_dim]
        const Tensor& v   // [seq_len_k, n_kv_heads, head_dim]
    );
    
    // Helper: repeat KV heads for GQA
    Tensor repeat_kv(const Tensor& x, int n_rep);
};

// Attention utilities
namespace attention_utils {
    // Create causal mask (upper triangular = masked)
    // Returns: [seq_len, seq_len] with 1 = mask, 0 = keep
    Tensor create_causal_mask(int seq_len);
    
    // Split tensor into heads
    // Input: [seq_len, n_heads * head_dim]
    // Output: [seq_len, n_heads, head_dim]
    Tensor split_heads(const Tensor& x, int n_heads, int head_dim);
    
    // Merge heads back
    // Input: [seq_len, n_heads, head_dim]
    // Output: [seq_len, n_heads * head_dim]
    Tensor merge_heads(const Tensor& x);
    
    // Transpose for attention (swap seq_len and n_heads dims)
    // [seq_len, n_heads, head_dim] -> [n_heads, seq_len, head_dim]
    Tensor transpose_for_scores(const Tensor& x);
}

} // namespace ash
