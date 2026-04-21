#include "attention.h"
#include "logger.h"
#include <cmath>
#include <cstring>
#include <algorithm>

namespace ash {

// =========================================================================
// KVCache Implementation
// =========================================================================

KVCache::KVCache(int max_seq_len, int n_layers, int n_kv_heads, int head_dim)
    : max_seq_len_(max_seq_len)
    , n_layers_(n_layers)
    , n_kv_heads_(n_kv_heads)
    , head_dim_(head_dim)
    , current_pos_(0) {
    
    // Allocate cache for each layer
    cache_.reserve(n_layers);
    for (int i = 0; i < n_layers; ++i) {
        TensorShape shape({max_seq_len, n_kv_heads, head_dim});
        Tensor k_cache = Tensor::zeros(shape, DType::F32);
        Tensor v_cache = Tensor::zeros(shape, DType::F32);
        cache_.emplace_back(std::move(k_cache), std::move(v_cache));
    }
    
    Logger::instance().debug("KV cache initialized: " + 
        std::to_string(n_layers) + " layers, " +
        std::to_string(max_seq_len) + " max seq len");
}

KVCache::~KVCache() = default;

void KVCache::update(int layer, int pos, const Tensor& key, const Tensor& value) {
    if (layer < 0 || layer >= n_layers_) {
        throw std::runtime_error("KVCache: layer index out of bounds");
    }
    
    if (pos < 0 || pos >= max_seq_len_) {
        throw std::runtime_error("KVCache: position out of bounds");
    }
    
    // Copy key/value into cache at position
    // key/value shape: [1, n_kv_heads, head_dim] or [n_kv_heads, head_dim]
    auto& [k_cache, v_cache] = cache_[layer];
    
    float* k_cache_data = k_cache.data_f32();
    float* v_cache_data = v_cache.data_f32();
    
    const float* k_data = key.data_f32();
    const float* v_data = value.data_f32();
    
    int offset = pos * n_kv_heads_ * head_dim_;
    int size = n_kv_heads_ * head_dim_;
    
    std::memcpy(k_cache_data + offset, k_data, size * sizeof(float));
    std::memcpy(v_cache_data + offset, v_data, size * sizeof(float));
    
    current_pos_ = std::max(current_pos_, pos + 1);
}

std::pair<Tensor, Tensor> KVCache::get(int layer, int start_pos, int end_pos) {
    if (layer < 0 || layer >= n_layers_) {
        throw std::runtime_error("KVCache: layer index out of bounds");
    }
    
    int seq_len = end_pos - start_pos;
    if (seq_len <= 0) {
        throw std::runtime_error("KVCache: invalid position range");
    }
    
    auto& [k_cache, v_cache] = cache_[layer];
    
    // Extract slice from cache
    TensorShape shape({seq_len, n_kv_heads_, head_dim_});
    Tensor k = Tensor::empty(shape, DType::F32);
    Tensor v = Tensor::empty(shape, DType::F32);
    
    float* k_data = k.data_f32();
    float* v_data = v.data_f32();
    
    const float* k_cache_data = k_cache.data_f32();
    const float* v_cache_data = v_cache.data_f32();
    
    int offset = start_pos * n_kv_heads_ * head_dim_;
    int size = seq_len * n_kv_heads_ * head_dim_;
    
    std::memcpy(k_data, k_cache_data + offset, size * sizeof(float));
    std::memcpy(v_data, v_cache_data + offset, size * sizeof(float));
    
    return {std::move(k), std::move(v)};
}

void KVCache::clear() {
    current_pos_ = 0;
}

void KVCache::reset_to(int pos) {
    if (pos < 0 || pos > current_pos_) {
        throw std::runtime_error("KVCache: invalid reset position");
    }
    current_pos_ = pos;
}

// =========================================================================
// MultiHeadAttention Implementation
// =========================================================================

MultiHeadAttention::MultiHeadAttention(const AttentionConfig& config)
    : config_(config) {
    
    // Precompute RoPE frequencies
    auto [cos_freqs, sin_freqs] = precompute_rope_freqs(
        config.max_seq_len,
        config.head_dim,
        config.rope_theta
    );
    
    rope_freqs_cos_ = std::move(cos_freqs);
    rope_freqs_sin_ = std::move(sin_freqs);
    
    Logger::instance().debug("Attention initialized: " +
        std::to_string(config.n_heads) + " heads, " +
        std::to_string(config.head_dim) + " dim per head");
}

MultiHeadAttention::~MultiHeadAttention() = default;

void MultiHeadAttention::apply_rope(Tensor& xq, Tensor& xk, int pos) {
    // Apply RoPE to queries and keys
    // Input shape: [seq_len, n_heads, head_dim]
    
    int seq_len = xq.shape().size(0);
    int n_heads = xq.shape().size(1);
    int head_dim = xq.shape().size(2);
    
    float* xq_data = xq.data_f32();
    float* xk_data = xk.data_f32();
    
    const float* cos_data = rope_freqs_cos_.data_f32();
    const float* sin_data = rope_freqs_sin_.data_f32();
    
    int half_dim = head_dim / 2;
    
    for (int seq = 0; seq < seq_len; ++seq) {
        int rope_pos = pos + seq;
        
        for (int h = 0; h < n_heads; ++h) {
            for (int i = 0; i < half_dim; ++i) {
                int idx = seq * n_heads * head_dim + h * head_dim;
                
                float cos_val = cos_data[rope_pos * half_dim + i];
                float sin_val = sin_data[rope_pos * half_dim + i];
                
                // Apply rotation to query
                float q_even = xq_data[idx + 2*i];
                float q_odd = xq_data[idx + 2*i + 1];
                xq_data[idx + 2*i] = q_even * cos_val - q_odd * sin_val;
                xq_data[idx + 2*i + 1] = q_even * sin_val + q_odd * cos_val;
                
                // Apply rotation to key
                float k_even = xk_data[idx + 2*i];
                float k_odd = xk_data[idx + 2*i + 1];
                xk_data[idx + 2*i] = k_even * cos_val - k_odd * sin_val;
                xk_data[idx + 2*i + 1] = k_even * sin_val + k_odd * cos_val;
            }
        }
    }
}

Tensor MultiHeadAttention::repeat_kv(const Tensor& x, int n_rep) {
    if (n_rep == 1) {
        // No repetition needed
        return Tensor::from_data(x.data(), x.shape(), x.dtype());
    }
    
    // Repeat KV heads: [seq_len, n_kv_heads, head_dim] -> [seq_len, n_heads, head_dim]
    int seq_len = x.shape().size(0);
    int n_kv_heads = x.shape().size(1);
    int head_dim = x.shape().size(2);
    
    Tensor out = Tensor::zeros({seq_len, n_kv_heads * n_rep, head_dim}, DType::F32);
    
    const float* x_data = x.data_f32();
    float* out_data = out.data_f32();
    
    for (int seq = 0; seq < seq_len; ++seq) {
        for (int kv_head = 0; kv_head < n_kv_heads; ++kv_head) {
            for (int rep = 0; rep < n_rep; ++rep) {
                int out_head = kv_head * n_rep + rep;
                for (int d = 0; d < head_dim; ++d) {
                    int x_idx = seq * n_kv_heads * head_dim + kv_head * head_dim + d;
                    int out_idx = seq * (n_kv_heads * n_rep) * head_dim + out_head * head_dim + d;
                    out_data[out_idx] = x_data[x_idx];
                }
            }
        }
    }
    
    return out;
}

Tensor MultiHeadAttention::compute_attention(const Tensor& q, const Tensor& k, const Tensor& v) {
    // Simplified attention computation
    // q: [seq_q, n_heads, head_dim]
    // k: [seq_k, n_heads, head_dim]
    // v: [seq_k, n_heads, head_dim]
    
    int seq_q = q.shape().size(0);
    int seq_k = k.shape().size(0);
    int n_heads = q.shape().size(1);
    int head_dim = q.shape().size(2);
    
    // For now, simplified single-head computation
    // TODO: Proper batched multi-head attention
    
    // Flatten to 2D for matmul: [seq_q, head_dim] @ [head_dim, seq_k]^T
    // This is a simplified version - proper implementation needs head-wise computation
    
    Tensor output = Tensor::zeros({seq_q, n_heads, head_dim}, DType::F32);
    
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    
    const float* q_data = q.data_f32();
    const float* k_data = k.data_f32();
    const float* v_data = v.data_f32();
    float* out_data = output.data_f32();
    
    // Compute attention for each head
    for (int h = 0; h < n_heads; ++h) {
        // Compute scores: Q @ K^T / sqrt(d_k)
        std::vector<float> scores(seq_q * seq_k);
        
        for (int i = 0; i < seq_q; ++i) {
            for (int j = 0; j < seq_k; ++j) {
                float dot = 0.0f;
                for (int d = 0; d < head_dim; ++d) {
                    int q_idx = i * n_heads * head_dim + h * head_dim + d;
                    int k_idx = j * n_heads * head_dim + h * head_dim + d;
                    dot += q_data[q_idx] * k_data[k_idx];
                }
                scores[i * seq_k + j] = dot * scale;
            }
        }
        
        // Apply softmax to each query position
        for (int i = 0; i < seq_q; ++i) {
            // Find max for numerical stability
            float max_val = scores[i * seq_k];
            for (int j = 1; j < seq_k; ++j) {
                max_val = std::max(max_val, scores[i * seq_k + j]);
            }
            
            // Compute exp and sum
            float sum = 0.0f;
            for (int j = 0; j < seq_k; ++j) {
                scores[i * seq_k + j] = std::exp(scores[i * seq_k + j] - max_val);
                sum += scores[i * seq_k + j];
            }
            
            // Normalize
            for (int j = 0; j < seq_k; ++j) {
                scores[i * seq_k + j] /= sum;
            }
        }
        
        // Compute weighted sum: scores @ V
        for (int i = 0; i < seq_q; ++i) {
            for (int d = 0; d < head_dim; ++d) {
                float val = 0.0f;
                for (int j = 0; j < seq_k; ++j) {
                    int v_idx = j * n_heads * head_dim + h * head_dim + d;
                    val += scores[i * seq_k + j] * v_data[v_idx];
                }
                int out_idx = i * n_heads * head_dim + h * head_dim + d;
                out_data[out_idx] = val;
            }
        }
    }
    
    return output;
}

Tensor MultiHeadAttention::forward(
    const Tensor& x,
    const Tensor& wq,
    const Tensor& wk,
    const Tensor& wv,
    const Tensor& wo,
    int pos,
    KVCache* kv_cache
) {
    int seq_len = x.shape().size(0);
    int hidden_dim = x.shape().size(1);
    
    // Project to Q, K, V
    // x: [seq_len, hidden_dim]
    // wq, wk, wv: [hidden_dim, ?]
    
    Tensor xq = matmul(x, wq);  // [seq_len, hidden_dim]
    Tensor xk = matmul(x, wk);  // [seq_len, kv_dim]
    Tensor xv = matmul(x, wv);  // [seq_len, kv_dim]
    
    // Reshape to heads: [seq_len, n_heads, head_dim]
    xq = attention_utils::split_heads(xq, config_.n_heads, config_.head_dim);
    xk = attention_utils::split_heads(xk, config_.n_kv_heads, config_.head_dim);
    xv = attention_utils::split_heads(xv, config_.n_kv_heads, config_.head_dim);
    
    // Apply RoPE
    apply_rope(xq, xk, pos);
    
    // Handle KV cache
    Tensor keys = std::move(xk);
    Tensor values = std::move(xv);
    
    // TODO: KV cache integration (for now just use current keys/values)
    
    // Repeat KV for GQA if needed
    if (config_.is_gqa()) {
        int n_rep = config_.n_heads / config_.n_kv_heads;
        keys = repeat_kv(keys, n_rep);
        values = repeat_kv(values, n_rep);
    }
    
    // Compute attention
    Tensor attn_output = compute_attention(xq, keys, values);
    
    // Merge heads back: [seq_len, n_heads, head_dim] -> [seq_len, hidden_dim]
    Tensor merged = attention_utils::merge_heads(attn_output);
    
    // Output projection
    Tensor output = matmul(merged, wo);
    
    return output;
}

// =========================================================================
// Attention Utilities
// =========================================================================

namespace attention_utils {

Tensor create_causal_mask(int seq_len) {
    Tensor mask = Tensor::zeros({seq_len, seq_len}, DType::F32);
    float* mask_data = mask.data_f32();
    
    // Upper triangular = 1 (masked)
    for (int i = 0; i < seq_len; ++i) {
        for (int j = i + 1; j < seq_len; ++j) {
            mask_data[i * seq_len + j] = 1.0f;
        }
    }
    
    return mask;
}

Tensor split_heads(const Tensor& x, int n_heads, int head_dim) {
    // [seq_len, n_heads * head_dim] -> [seq_len, n_heads, head_dim]
    int seq_len = x.shape().size(0);
    
    Tensor out = Tensor::empty({seq_len, n_heads, head_dim}, DType::F32);
    
    const float* x_data = x.data_f32();
    float* out_data = out.data_f32();
    
    std::memcpy(out_data, x_data, seq_len * n_heads * head_dim * sizeof(float));
    
    return out;
}

Tensor merge_heads(const Tensor& x) {
    // [seq_len, n_heads, head_dim] -> [seq_len, n_heads * head_dim]
    int seq_len = x.shape().size(0);
    int n_heads = x.shape().size(1);
    int head_dim = x.shape().size(2);
    
    Tensor out = Tensor::empty({seq_len, n_heads * head_dim}, DType::F32);
    
    const float* x_data = x.data_f32();
    float* out_data = out.data_f32();
    
    std::memcpy(out_data, x_data, seq_len * n_heads * head_dim * sizeof(float));
    
    return out;
}

Tensor transpose_for_scores(const Tensor& x) {
    // [seq_len, n_heads, head_dim] -> [n_heads, seq_len, head_dim]
    int seq_len = x.shape().size(0);
    int n_heads = x.shape().size(1);
    int head_dim = x.shape().size(2);
    
    Tensor out = Tensor::empty({n_heads, seq_len, head_dim}, DType::F32);
    
    const float* x_data = x.data_f32();
    float* out_data = out.data_f32();
    
    for (int h = 0; h < n_heads; ++h) {
        for (int s = 0; s < seq_len; ++s) {
            for (int d = 0; d < head_dim; ++d) {
                int in_idx = s * n_heads * head_dim + h * head_dim + d;
                int out_idx = h * seq_len * head_dim + s * head_dim + d;
                out_data[out_idx] = x_data[in_idx];
            }
        }
    }
    
    return out;
}

} // namespace attention_utils

} // namespace ash
