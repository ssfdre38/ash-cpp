#include "attention.h"
#include "logger.h"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <iostream>

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
    // Input shape: [seq_len, n_heads, head_dim] for xq
    //              [seq_len, n_kv_heads, head_dim] for xk (may be different for GQA)
    
    int seq_len = xq.shape().size(0);
    int n_q_heads = xq.shape().size(1);
    int n_kv_heads = xk.shape().size(1);
    int head_dim = xq.shape().size(2);
    
    float* xq_data = xq.data_f32();
    float* xk_data = xk.data_f32();
    
    const float* cos_data = rope_freqs_cos_.data_f32();
    const float* sin_data = rope_freqs_sin_.data_f32();
    
    int half_dim = head_dim / 2;
    
    for (int seq = 0; seq < seq_len; ++seq) {
        int rope_pos = pos + seq;
        
        // Apply to queries
        for (int h = 0; h < n_q_heads; ++h) {
            for (int i = 0; i < half_dim; ++i) {
                int idx = seq * n_q_heads * head_dim + h * head_dim;
                
                float cos_val = cos_data[rope_pos * half_dim + i];
                float sin_val = sin_data[rope_pos * half_dim + i];
                
                // Apply rotation to query
                float q_even = xq_data[idx + 2*i];
                float q_odd = xq_data[idx + 2*i + 1];
                xq_data[idx + 2*i] = q_even * cos_val - q_odd * sin_val;
                xq_data[idx + 2*i + 1] = q_even * sin_val + q_odd * cos_val;
            }
        }
        
        // Apply to keys (separate loop for different head count)
        for (int h = 0; h < n_kv_heads; ++h) {
            for (int i = 0; i < half_dim; ++i) {
                int idx = seq * n_kv_heads * head_dim + h * head_dim;
                
                float cos_val = cos_data[rope_pos * half_dim + i];
                float sin_val = sin_data[rope_pos * half_dim + i];
                
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

Tensor MultiHeadAttention::compute_attention(const Tensor& q, const Tensor& k, const Tensor& v, int start_pos, const std::string& arch) {
    // Simplified attention computation
    // q: [seq_q, n_heads, head_dim]
    // k: [seq_k, n_heads, head_dim]
    // v: [seq_k, n_heads, head_dim]
    // start_pos: Absolute position of first query in the full sequence
    
    int seq_q = q.shape().size(0);
    int seq_k = k.shape().size(0);
    int n_heads = q.shape().size(1);
    int head_dim = q.shape().size(2);
    
    Logger::instance().debug("compute_attention: seq_q=" + std::to_string(seq_q) + 
                           ", seq_k=" + std::to_string(seq_k) + 
                           ", n_heads=" + std::to_string(n_heads) + 
                           ", head_dim=" + std::to_string(head_dim));
    
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
                
                // Scale attention scores
                float scaled_score = dot * scale;
                
                // Gemma requirement: logit soft-capping
                if (arch == "gemma") {
                    const float SOFT_CAP = 50.0f;
                    scaled_score = SOFT_CAP * std::tanh(scaled_score / SOFT_CAP);
                }
                
                // Apply causal masking using absolute positions
                int abs_q_pos = start_pos + i;
                if (j > abs_q_pos) {
                    scores[i * seq_k + j] = -INFINITY;
                } else {
                    scores[i * seq_k + j] = scaled_score;
                }
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
        
        // Compute attention output: weighted sum of values
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
    const Tensor& bq,
    const Tensor& bk,
    const Tensor& bv,
    int layer_idx,
    int pos,
    KVCache* kv_cache
) {
    Logger::instance().debug("attention::forward layer " + std::to_string(layer_idx));
    int seq_len = x.shape().size(0);
    int hidden_dim = x.shape().size(1);
    
    Logger::instance().debug("Projecting Q, K, V...");
    
    // Project to Q, K, V using optimized transpose
    // x: [seq_len, hidden_dim] = [424, 2560]
    // wq, wk, wv stored as [out_features, in_features] in GGUF
    // We need: x @ wq^T efficiently (no temp allocation)
    
    Tensor xq = matmul_transposed(x, wq, false, true);  // x @ wq^T -> [seq_len, n_q_heads * head_dim]
    if (bq.is_allocated()) {
        // Add bias: xq += bq (broadcast across seq_len dimension)
        const float* bq_data = bq.data_f32();
        float* xq_data = xq.data_f32();
        int qkv_dim = xq.shape().size(1);  // n_q_heads * head_dim
        for (int s = 0; s < seq_len; ++s) {
            for (int d = 0; d < qkv_dim; ++d) {
                xq_data[s * qkv_dim + d] += bq_data[d];
            }
        }
    }
    Logger::instance().debug("Q projection done");
    
    Tensor xk = matmul_transposed(x, wk, false, true);  // x @ wk^T -> [seq_len, n_kv_heads * head_dim]
    if (bk.is_allocated()) {
        // Add bias: xk += bk (broadcast across seq_len dimension)
        const float* bk_data = bk.data_f32();
        float* xk_data = xk.data_f32();
        int kv_dim = xk.shape().size(1);  // n_kv_heads * head_dim
        for (int s = 0; s < seq_len; ++s) {
            for (int d = 0; d < kv_dim; ++d) {
                xk_data[s * kv_dim + d] += bk_data[d];
            }
        }
    }
    Logger::instance().debug("K projection done");
    
    Tensor xv = matmul_transposed(x, wv, false, true);// x @ wv^T -> [seq_len, n_kv_heads * head_dim]
    if (bv.is_allocated()) {
        // Add bias: xv += bv (broadcast across seq_len dimension)
        const float* bv_data = bv.data_f32();
        float* xv_data = xv.data_f32();
        int kv_dim = xv.shape().size(1);  // n_kv_heads * head_dim
        for (int s = 0; s < seq_len; ++s) {
            for (int d = 0; d < kv_dim; ++d) {
                xv_data[s * kv_dim + d] += bv_data[d];
            }
        }
    }
    Logger::instance().debug("V projection done");
    
    // Reshape to heads: [seq_len, n_heads, head_dim]
    Logger::instance().debug("Splitting heads...");
    xq = attention_utils::split_heads(xq, config_.n_heads, config_.head_dim);
    Logger::instance().debug("Q heads split");
    xk = attention_utils::split_heads(xk, config_.n_kv_heads, config_.head_dim);
    Logger::instance().debug("K heads split");
    xv = attention_utils::split_heads(xv, config_.n_kv_heads, config_.head_dim);
    Logger::instance().debug("V heads split");
    
    // Apply RoPE
    Logger::instance().debug("Applying RoPE...");
    apply_rope(xq, xk, pos);
    
    // CRITICAL FIX: KV cache integration
    // For autoregressive generation, we cache past K,V and only compute new ones
    Tensor keys, values;
    
    // Determine if this is prefill (seq_len > 1) or generation (seq_len == 1)
    bool is_generation = (seq_len == 1);
    
    bool use_cache = (kv_cache != nullptr && kv_cache->seq_len() > 0 && is_generation);
    
    if (use_cache) {
        // Generation phase: we have past K,V and are adding one new token
        // CRITICAL FIX: Use 'pos' (the position we're generating at) instead of kv_cache->seq_len()
        // because seq_len() is a global counter that increments as each layer writes!
        // We want the cache length BEFORE this generation step.
        int cache_len = pos;  // Position 5 means we have cached 0-4 (5 entries)
        
        // 1. Get past K,V from cache (positions 0 to cache_len-1)
        auto [past_k, past_v] = kv_cache->get(layer_idx, 0, cache_len);
        
        // 2. Concatenate past + new: [cache_len + 1, n_kv_heads, head_dim]
        keys = attention_utils::concat_tensors(past_k, xk, 0);
        values = attention_utils::concat_tensors(past_v, xv, 0);
        
        // 3. Store new K,V into cache
        Tensor k_slice = attention_utils::slice_tensor(xk, 0, 1, 0);
        Tensor v_slice = attention_utils::slice_tensor(xv, 0, 1, 0);
        
        kv_cache->update(layer_idx, pos, k_slice, v_slice);
    } else {
        // Prefill phase: store all K,V positions, no concat
        if (kv_cache != nullptr) {
            for (int i = 0; i < seq_len; ++i) {
                Tensor k_slice = attention_utils::slice_tensor(xk, i, i+1, 0);
                Tensor v_slice = attention_utils::slice_tensor(xv, i, i+1, 0);
                kv_cache->update(layer_idx, pos + i, k_slice, v_slice);
            }
        }
        
        // Use current K,V directly (no concat)
        keys = std::move(xk);
        values = std::move(xv);
    }
    
    // Repeat KV for GQA if needed
    if (config_.is_gqa()) {
        int n_rep = config_.n_heads / config_.n_kv_heads;
        
        keys = repeat_kv(keys, n_rep);
        values = repeat_kv(values, n_rep);
    }
    
    // CRITICAL: Assert invariants before attention computation
    if (is_generation) {
        // During generation: seq_q must be 1, seq_k must be pos+1
        if (xq.shape().size(0) != 1) {
            throw std::runtime_error("Generation: seq_q must be 1");
        }
        if (keys.shape().size(0) != pos + 1) {
            throw std::runtime_error("Generation: seq_k must be pos+1 (got " + 
                std::to_string(keys.shape().size(0)) + " expected " + std::to_string(pos + 1) + ")");
        }
    }
    
    // Assert K/V shapes match expected dimensions after repeat_kv
    if (keys.shape().size(1) != config_.n_heads) {
        throw std::runtime_error("Keys n_heads mismatch");
    }
    if (keys.shape().size(2) != config_.head_dim) {
        throw std::runtime_error("Keys head_dim mismatch");
    }
    if (values.shape().size(1) != config_.n_heads) {
        throw std::runtime_error("Values n_heads mismatch");
    }
    if (values.shape().size(2) != config_.head_dim) {
        throw std::runtime_error("Values head_dim mismatch");
    }
    
    // Compute attention
    Tensor attn_output = compute_attention(xq, keys, values, pos, config_.architecture);
    
    // Merge heads back: [seq_len, n_heads, head_dim] -> [seq_len, hidden_dim]
    Tensor merged = attention_utils::merge_heads(attn_output);
    
    // Output projection: merged @ wo^T
    // wo is stored as [hidden_dim, hidden_dim] in GGUF
    // We need: merged[seq_len, hidden_dim] @ wo^T[hidden_dim, hidden_dim] = [seq_len, hidden_dim]
    Tensor output = matmul_transposed(merged, wo, false, true);
    
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

Tensor concat_tensors(const Tensor& a, const Tensor& b, int dim) {
    // Concatenate two tensors along dimension `dim`
    // For KV cache: dim=0 (sequence dimension)
    // a: [seq_a, n_heads, head_dim]
    // b: [seq_b, n_heads, head_dim]
    // out: [seq_a + seq_b, n_heads, head_dim]
    
    if (dim != 0) {
        throw std::runtime_error("concat_tensors only supports dim=0");
    }
    
    // CRITICAL: Validate shapes match exactly (except concat dimension)
    if (a.dtype() != b.dtype()) {
        throw std::runtime_error("concat_tensors: dtype mismatch");
    }
    if (a.shape().ndim() != b.shape().ndim()) {
        throw std::runtime_error("concat_tensors: rank mismatch");
    }
    if (a.shape().size(1) != b.shape().size(1)) {
        throw std::runtime_error("concat_tensors: dimension 1 mismatch (n_heads)");
    }
    if (a.shape().size(2) != b.shape().size(2)) {
        throw std::runtime_error("concat_tensors: dimension 2 mismatch (head_dim)");
    }
    
    int seq_a = a.shape().size(0);
    int seq_b = b.shape().size(0);
    int n_heads = a.shape().size(1);
    int head_dim = a.shape().size(2);
    
    Tensor out = Tensor::empty({seq_a + seq_b, n_heads, head_dim}, DType::F32);
    
    const float* a_data = a.data_f32();
    const float* b_data = b.data_f32();
    float* out_data = out.data_f32();
    
    // Copy a first
    int a_size = seq_a * n_heads * head_dim;
    std::memcpy(out_data, a_data, a_size * sizeof(float));
    
    // Copy b after a
    int b_size = seq_b * n_heads * head_dim;
    std::memcpy(out_data + a_size, b_data, b_size * sizeof(float));
    
    return out;
}

Tensor slice_tensor(const Tensor& x, int start, int end, int dim) {
    // Extract slice [start:end) along dimension `dim`
    // For KV cache: dim=0 (sequence dimension)
    // x: [seq_len, n_heads, head_dim]
    // out: [end-start, n_heads, head_dim]
    
    if (dim != 0) {
        throw std::runtime_error("slice_tensor only supports dim=0");
    }
    
    int seq_len = x.shape().size(0);
    int n_heads = x.shape().size(1);
    int head_dim = x.shape().size(2);
    
    if (start < 0 || end > seq_len || start >= end) {
        throw std::runtime_error("slice_tensor: invalid range");
    }
    
    int slice_len = end - start;
    Tensor out = Tensor::empty({slice_len, n_heads, head_dim}, DType::F32);
    
    const float* x_data = x.data_f32();
    float* out_data = out.data_f32();
    
    int offset = start * n_heads * head_dim;
    int size = slice_len * n_heads * head_dim;
    
    std::memcpy(out_data, x_data + offset, size * sizeof(float));
    
    return out;
}

} // namespace attention_utils

} // namespace ash
