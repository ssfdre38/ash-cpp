#include "inference.h"
#include "matrix_ops.h"
#include "logger.h"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <random>
#include <chrono>

namespace ash {

InferenceEngine::InferenceEngine() {
    weights_.loaded = false;
}

InferenceEngine::~InferenceEngine() {
    // Cleanup handled by smart pointers
}

bool InferenceEngine::load_model(const std::string& gguf_path) {
    GGUFParser parser;
    if (!parser.parse(gguf_path)) {
        return false;
    }
    
    // Extract model config from parser
    config_.architecture = parser.get_architecture();
    config_.vocab_size = parser.get_vocab_size();
    config_.hidden_dim = parser.get_embedding_dim();
    config_.n_layers = parser.get_num_layers();
    config_.n_heads = parser.get_num_heads();
    config_.n_kv_heads = parser.get_num_kv_heads();
    config_.max_seq_len = parser.get_context_length();
    
    std::string prefix = config_.architecture + ".";
    config_.intermediate_dim = parser.get_uint(prefix + "feed_forward_length", 8192);
    config_.rope_theta = parser.get_float(prefix + "rope.freq_base", 10000.0f);
    config_.rms_norm_eps = parser.get_float(prefix + "attention.layer_norm_rms_epsilon", 1e-6f);
    
    config_.head_dim = config_.hidden_dim / config_.n_heads;
    
    // Load weights
    if (!load_weights_from_gguf(parser)) {
        return false;
    }
    
    // Create tokenizer from GGUF vocab
    tokenizer_ = TokenizerFactory::from_gguf(gguf_path);
    if (!tokenizer_) {
        return false;
    }
    
    // Create attention module
    attention_ = std::make_unique<MultiHeadAttention>(config_.attention_config());
    
    weights_.loaded = true;
    return true;
}

bool InferenceEngine::load_weights_from_gguf(GGUFParser& parser) {
    // Load token embeddings
    weights_.token_embeddings = parser.load_tensor("token_embd.weight");
    if (!weights_.token_embeddings.is_allocated()) {
        return false;
    }
    
    // Load layer weights
    std::string prefix = config_.architecture + ".blk.";
    weights_.layers.resize(config_.n_layers);
    
    for (size_t i = 0; i < config_.n_layers; i++) {
        std::string layer_prefix = prefix + std::to_string(i) + ".";
        LayerWeights& layer = weights_.layers[i];
        
        // Attention weights
        layer.wq = parser.load_tensor(layer_prefix + "attn_q.weight");
        layer.wk = parser.load_tensor(layer_prefix + "attn_k.weight");
        layer.wv = parser.load_tensor(layer_prefix + "attn_v.weight");
        layer.wo = parser.load_tensor(layer_prefix + "attn_output.weight");
        
        // FFN weights
        layer.w_gate = parser.load_tensor(layer_prefix + "ffn_gate.weight");
        layer.w_up = parser.load_tensor(layer_prefix + "ffn_up.weight");
        layer.w_down = parser.load_tensor(layer_prefix + "ffn_down.weight");
        
        // Layer norms
        layer.attn_norm = parser.load_tensor(layer_prefix + "attn_norm.weight");
        layer.ffn_norm = parser.load_tensor(layer_prefix + "ffn_norm.weight");
        
        // Verify critical weights loaded
        if (!layer.wq.is_allocated() || !layer.wk.is_allocated() || !layer.wv.is_allocated()) {
            return false;
        }
    }
    
    // Output norm and projection
    weights_.output_norm = parser.load_tensor(config_.architecture + ".output_norm.weight");
    weights_.output = parser.load_tensor("output.weight");
    
    // Often output weight is tied to embeddings
    if (!weights_.output.is_allocated()) {
        weights_.output = weights_.token_embeddings.clone();
    }
    
    return true;
}

Tensor InferenceEngine::forward(const std::vector<TokenID>& tokens, KVCache* kv_cache) {
    if (!weights_.loaded) {
        return Tensor();
    }
    
    int seq_len = tokens.size();
    
    // Get token embeddings: [seq_len, hidden_dim]
    Tensor x = Tensor::empty({seq_len, config_.hidden_dim}, DType::F32);
    for (int i = 0; i < seq_len; i++) {
        // Copy embedding for token[i]
        const float* emb_data = weights_.token_embeddings.data_f32() + tokens[i] * config_.hidden_dim;
        float* x_data = x.data_f32() + i * config_.hidden_dim;
        std::memcpy(x_data, emb_data, config_.hidden_dim * sizeof(float));
    }
    
    // Forward through layers
    for (int layer_idx = 0; layer_idx < (int)config_.n_layers; layer_idx++) {
        x = forward_layer(x, weights_.layers[layer_idx], layer_idx, seq_len - 1, kv_cache);
    }
    
    // Final RMSNorm
    x = rmsnorm(x, weights_.output_norm, config_.rms_norm_eps);
    
    // Output projection: [seq_len, hidden_dim] @ [hidden_dim, vocab_size]^T = [seq_len, vocab_size]
    Tensor logits = matmul(x, weights_.output);
    
    return logits;
}

Tensor InferenceEngine::forward_layer(
    const Tensor& x,
    const LayerWeights& layer,
    int layer_idx,
    int pos,
    KVCache* kv_cache
) {
    // Pre-attention RMSNorm
    Tensor x_norm = rmsnorm(x, layer.attn_norm, config_.rms_norm_eps);
    
    // Attention (projects Q,K,V internally)
    Tensor attn_out = attention_->forward(
        x_norm, 
        layer.wq, layer.wk, layer.wv, layer.wo,
        pos,
        kv_cache
    );
    
    // Residual connection
    Tensor x_attn = add(x, attn_out);
    
    // Pre-FFN RMSNorm
    Tensor x_attn_norm = rmsnorm(x_attn, layer.ffn_norm, config_.rms_norm_eps);
    
    // FFN with residual
    Tensor ffn_out = forward_ffn(x_attn_norm, layer);
    Tensor x_out = add(x_attn, ffn_out);
    
    return x_out;
}

Tensor InferenceEngine::forward_ffn(const Tensor& x, const LayerWeights& layer) {
    // SwiGLU: silu(gate(x)) * up(x) @ down
    Tensor gate = matmul(x, layer.w_gate);
    Tensor up = matmul(x, layer.w_up);
    
    // Apply SiLU to gate
    gate = silu(gate);
    
    // Element-wise multiply
    Tensor gated = multiply(gate, up);
    
    // Down projection
    Tensor out = matmul(gated, layer.w_down);
    
    return out;
}

GenerationResult InferenceEngine::generate(
    const std::string& prompt,
    const SamplingConfig& sampling
) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    GenerationResult result;
    result.tokens_generated = 0;
    result.stopped_early = false;
    
    if (!weights_.loaded) {
        result.stop_reason = "model_not_loaded";
        return result;
    }
    
    // Tokenize prompt
    std::vector<TokenID> tokens = tokenizer_->encode(prompt);
    if (tokens.empty()) {
        result.stop_reason = "tokenization_failed";
        return result;
    }
    
    // Create KV cache for this generation
    KVCache kv_cache(config_.n_layers, config_.n_kv_heads, config_.head_dim, config_.max_seq_len);
    
    // Generate tokens
    for (int i = 0; i < sampling.max_tokens; i++) {
        TokenID next_token = generate_next_token(tokens, sampling, &kv_cache);
        
        // Check stop conditions
        if (inference_utils::is_stop_token(next_token, sampling.stop_tokens)) {
            result.stopped_early = true;
            result.stop_reason = "stop_token";
            break;
        }
        
        tokens.push_back(next_token);
        result.tokens_generated++;
        
        // Check max length
        if (tokens.size() >= (size_t)config_.max_seq_len) {
            result.stopped_early = true;
            result.stop_reason = "max_length";
            break;
        }
    }
    
    // Decode tokens to text
    result.text = tokenizer_->decode(tokens);
    result.tokens = tokens;
    
    auto end_time = std::chrono::high_resolution_clock::now();
    result.generation_time_ms = std::chrono::duration<float, std::milli>(end_time - start_time).count();
    
    if (!result.stopped_early) {
        result.stop_reason = "max_tokens";
    }
    
    return result;
}

TokenID InferenceEngine::generate_next_token(
    const std::vector<TokenID>& context,
    const SamplingConfig& sampling,
    KVCache* kv_cache
) {
    // Forward pass
    Tensor logits = forward(context, kv_cache);
    
    // Get last token logits: [vocab_size]
    int seq_len = logits.shape().dims[0];
    int vocab_size = logits.shape().dims[1];
    
    Tensor last_logits = Tensor::empty({vocab_size}, DType::F32);
    const float* logits_data = logits.data_f32() + (seq_len - 1) * vocab_size;
    std::memcpy(last_logits.data_f32(), logits_data, vocab_size * sizeof(float));
    
    // Sample token
    return sample_token(last_logits, sampling);
}

TokenID InferenceEngine::sample_token(const Tensor& logits, const SamplingConfig& config) {
    if (!config.use_sampling || config.temperature == 0.0f) {
        return sample_greedy(logits);
    }
    
    // Use top-k or top-p sampling
    if (config.top_k > 0) {
        return sample_top_k(logits, config.top_k, config.temperature);
    } else if (config.top_p > 0.0f && config.top_p < 1.0f) {
        return sample_top_p(logits, config.top_p, config.temperature);
    }
    
    // Fallback to greedy
    return sample_greedy(logits);
}

TokenID InferenceEngine::sample_greedy(const Tensor& logits) {
    const float* data = logits.data_f32();
    int vocab_size = logits.num_elements();
    
    int max_idx = 0;
    float max_val = data[0];
    
    for (int i = 1; i < vocab_size; i++) {
        if (data[i] > max_val) {
            max_val = data[i];
            max_idx = i;
        }
    }
    
    return max_idx;
}

TokenID InferenceEngine::sample_top_k(const Tensor& logits, int k, float temperature) {
    // Get top-k indices
    auto top_k = inference_utils::top_k_indices(logits, k);
    
    // Extract top-k logits
    Tensor top_k_logits({k}, DType::F32);
    const float* logits_data = logits.data_f32();
    float* top_k_data = top_k_logits.data_f32();
    
    for (int i = 0; i < k; i++) {
        top_k_data[i] = logits_data[top_k[i]];
    }
    
    // Apply temperature and softmax
    Tensor probs = inference_utils::softmax_temperature(top_k_logits, temperature);
    
    // Sample from distribution
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    
    float r = dis(gen);
    float cumsum = 0.0f;
    const float* probs_data = probs.data_f32();
    
    for (int i = 0; i < k; i++) {
        cumsum += probs_data[i];
        if (r < cumsum) {
            return top_k[i];
        }
    }
    
    // Fallback to last token
    return top_k[k - 1];
}

TokenID InferenceEngine::sample_top_p(const Tensor& logits, float p, float temperature) {
    int vocab_size = logits.num_elements();
    
    // Apply temperature
    Tensor temp_logits = logits.clone();
    inference_utils::apply_temperature(temp_logits, temperature);
    
    // Compute softmax
    Tensor probs = softmax(temp_logits);
    
    // Sort indices by probability (descending)
    std::vector<std::pair<float, int>> prob_idx;
    const float* probs_data = probs.data_f32();
    for (int i = 0; i < vocab_size; i++) {
        prob_idx.push_back({probs_data[i], i});
    }
    std::sort(prob_idx.begin(), prob_idx.end(), std::greater<>());
    
    // Find nucleus (cumulative probability > p)
    float cumsum = 0.0f;
    int nucleus_size = 0;
    for (int i = 0; i < vocab_size; i++) {
        cumsum += prob_idx[i].first;
        nucleus_size++;
        if (cumsum >= p) break;
    }
    
    // Sample from nucleus
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, cumsum);
    
    float r = dis(gen);
    float sum = 0.0f;
    for (int i = 0; i < nucleus_size; i++) {
        sum += prob_idx[i].first;
        if (r < sum) {
            return prob_idx[i].second;
        }
    }
    
    // Fallback to most likely
    return prob_idx[0].second;
}

// Inference utilities implementation
namespace inference_utils {

void apply_temperature(Tensor& logits, float temperature) {
    if (temperature == 1.0f) return;
    
    float* data = logits.data_f32();
    int size = logits.num_elements();
    
    for (int i = 0; i < size; i++) {
        data[i] /= temperature;
    }
}

std::vector<int> top_k_indices(const Tensor& logits, int k) {
    int vocab_size = logits.num_elements();
    k = std::min(k, vocab_size);
    
    // Create pairs of (logit, index)
    std::vector<std::pair<float, int>> logit_idx;
    const float* data = logits.data_f32();
    for (int i = 0; i < vocab_size; i++) {
        logit_idx.push_back({data[i], i});
    }
    
    // Partial sort to get top-k
    std::partial_sort(logit_idx.begin(), logit_idx.begin() + k, logit_idx.end(),
                     [](const auto& a, const auto& b) { return a.first > b.first; });
    
    // Extract indices
    std::vector<int> indices(k);
    for (int i = 0; i < k; i++) {
        indices[i] = logit_idx[i].second;
    }
    
    return indices;
}

Tensor softmax_temperature(const Tensor& logits, float temperature) {
    Tensor temp_logits = logits.clone();
    apply_temperature(temp_logits, temperature);
    return softmax(temp_logits);
}

bool is_stop_token(TokenID token, const std::vector<TokenID>& stop_tokens) {
    return std::find(stop_tokens.begin(), stop_tokens.end(), token) != stop_tokens.end();
}

} // namespace inference_utils

} // namespace ash
