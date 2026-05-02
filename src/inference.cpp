#include "inference.h"
#include "matrix_ops.h"
#include "logger.h"
#include <iostream>
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
    
    // Default theta depends on architecture
    float default_theta = (config_.architecture == "qwen2") ? 1000000.0f : 10000.0f;
    config_.rope_theta = parser.get_float(prefix + "rope.freq_base", default_theta);
    
    config_.rms_norm_eps = parser.get_float(prefix + "attention.layer_norm_rms_epsilon", 1e-6f);
    if (config_.rms_norm_eps == 0) config_.rms_norm_eps = 1e-6f; // Sanity check
    
    config_.head_dim = config_.hidden_dim / config_.n_heads;
    
    // Debug config values
    Logger::instance().info("Model config loaded:");
    Logger::instance().info("  n_layers=" + std::to_string(config_.n_layers));
    Logger::instance().info("  n_heads=" + std::to_string(config_.n_heads));
    Logger::instance().info("  n_kv_heads=" + std::to_string(config_.n_kv_heads));
    Logger::instance().info("  hidden_dim=" + std::to_string(config_.hidden_dim));
    Logger::instance().info("  head_dim=" + std::to_string(config_.head_dim));
    Logger::instance().info("  max_seq_len=" + std::to_string(config_.max_seq_len));
    Logger::instance().info("  rope_theta=" + std::to_string(config_.rope_theta));
    
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
    std::string prefix = "blk.";  // Tensor names are just "blk.X.Y", not "arch.blk.X.Y"
    weights_.layers.resize(config_.n_layers);
    
    for (size_t i = 0; i < config_.n_layers; i++) {
        std::string layer_prefix = prefix + std::to_string(i) + ".";
        LayerWeights& layer = weights_.layers[i];
        
        // Attention weights
        layer.wq = parser.load_tensor(layer_prefix + "attn_q.weight");
        layer.wk = parser.load_tensor(layer_prefix + "attn_k.weight");
        layer.wv = parser.load_tensor(layer_prefix + "attn_v.weight");
        layer.wo = parser.load_tensor(layer_prefix + "attn_output.weight");
        
        // Attention biases (Qwen2 has these, optional for other models)
        if (parser.find_tensor(layer_prefix + "attn_q.bias")) {
            layer.bq = parser.load_tensor(layer_prefix + "attn_q.bias");
        }
        if (parser.find_tensor(layer_prefix + "attn_k.bias")) {
            layer.bk = parser.load_tensor(layer_prefix + "attn_k.bias");
        }
        if (parser.find_tensor(layer_prefix + "attn_v.bias")) {
            layer.bv = parser.load_tensor(layer_prefix + "attn_v.bias");
        }
        
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
    weights_.output_norm = parser.load_tensor("output_norm.weight");
    
    // Try to load output projection (may not exist - will use tied embeddings)
    if (parser.find_tensor("output.weight")) {
        weights_.output = parser.load_tensor("output.weight");
    }
    
    // Often output weight is tied to embeddings
    if (!weights_.output.is_allocated()) {
        weights_.output = weights_.token_embeddings.clone();
    }
    
    // CRITICAL OPTIMIZATION: Dequantize all weights once at load time
    // This is much faster than dequantizing on every forward pass
    std::cout << "🔥 Dequantizing weights..." << std::flush;
    auto deq_start = std::chrono::high_resolution_clock::now();
    
    size_t deq_count = 0;
    
    // Dequantize embeddings and output
    if (weights_.token_embeddings.dtype() != DType::F32) {
        weights_.token_embeddings = weights_.token_embeddings.dequantize();
        deq_count++;
    }
    if (weights_.output.dtype() != DType::F32) {
        weights_.output = weights_.output.dequantize();
        deq_count++;
    }
    
    // Dequantize all layer weights
    for (size_t i = 0; i < config_.n_layers; i++) {
        LayerWeights& layer = weights_.layers[i];
        
        // Attention weights
        if (layer.wq.dtype() != DType::F32) { layer.wq = layer.wq.dequantize(); deq_count++; }
        if (layer.wk.dtype() != DType::F32) { layer.wk = layer.wk.dequantize(); deq_count++; }
        if (layer.wv.dtype() != DType::F32) { layer.wv = layer.wv.dequantize(); deq_count++; }
        if (layer.wo.dtype() != DType::F32) { layer.wo = layer.wo.dequantize(); deq_count++; }
        
        // Attention biases
        if (layer.bq.is_allocated() && layer.bq.dtype() != DType::F32) { layer.bq = layer.bq.dequantize(); deq_count++; }
        if (layer.bk.is_allocated() && layer.bk.dtype() != DType::F32) { layer.bk = layer.bk.dequantize(); deq_count++; }
        if (layer.bv.is_allocated() && layer.bv.dtype() != DType::F32) { layer.bv = layer.bv.dequantize(); deq_count++; }
        
        // FFN weights
        if (layer.w_gate.dtype() != DType::F32) { layer.w_gate = layer.w_gate.dequantize(); deq_count++; }
        if (layer.w_up.dtype() != DType::F32) { layer.w_up = layer.w_up.dequantize(); deq_count++; }
        if (layer.w_down.dtype() != DType::F32) { layer.w_down = layer.w_down.dequantize(); deq_count++; }
        
        // Norms are already F32
    }
    
    auto deq_end = std::chrono::high_resolution_clock::now();
    auto deq_ms = std::chrono::duration_cast<std::chrono::milliseconds>(deq_end - deq_start).count();
    std::cout << " " << deq_count << " tensors in " << deq_ms << "ms\n";
    
    return true;
}

Tensor InferenceEngine::forward(const std::vector<TokenID>& tokens, KVCache* kv_cache, int start_pos) {
    if (!weights_.loaded) {
        return Tensor();
    }
    
    int seq_len = tokens.size();
    
    // Get token embeddings: [seq_len, hidden_dim]
    Tensor x = Tensor::empty({seq_len, config_.hidden_dim}, DType::F32);
    
    // Check if we need embedding scaling (Gemma-specific)
    // Testing WITHOUT scaling since Qwen2 shouldn't need it
    bool use_emb_scaling = (config_.architecture == "gemma");
    float emb_scale = use_emb_scaling ? std::sqrt(static_cast<float>(config_.hidden_dim)) : 1.0f;
    
    // DEBUG: Print embedding scale
    if (start_pos == 0 && seq_len == 3) {
        std::cout << "\n=== EMBEDDING SETUP ===\n";
        std::cout << "Architecture: " << config_.architecture << "\n";
        std::cout << "use_emb_scaling: " << use_emb_scaling << "\n";
        std::cout << "emb_scale: " << emb_scale << "\n";
    }
    
    for (int i = 0; i < seq_len; i++) {
        // Copy embedding for token[i]
        const float* emb_data = weights_.token_embeddings.data_f32() + tokens[i] * config_.hidden_dim;
        float* x_data = x.data_f32() + i * config_.hidden_dim;
        
        // Scale embeddings if needed (Gemma requires sqrt(hidden_dim), Qwen doesn't)
        if (use_emb_scaling) {
            for (int d = 0; d < config_.hidden_dim; d++) {
                x_data[d] = emb_data[d] * emb_scale;
            }
        } else {
            std::memcpy(x_data, emb_data, config_.hidden_dim * sizeof(float));
        }
        
        // DEBUG: Print embedding L2 norm
        if (start_pos == 0 && seq_len == 3) {
            float norm = 0.0f;
            for (int d = 0; d < config_.hidden_dim; d++) {
                norm += x_data[d] * x_data[d];
            }
            norm = std::sqrt(norm);
            std::cout << "Token " << tokens[i] << " embedding norm: " << norm 
                      << ", first 10: [";
            for (int d = 0; d < 10; d++) {
                std::cout << x_data[d];
                if (d < 9) std::cout << ", ";
            }
            std::cout << "]\n";
        }
    }
    
    // Forward through layers
    // Pass start_pos so each token i gets absolute position start_pos + i
    for (int layer_idx = 0; layer_idx < (int)config_.n_layers; layer_idx++) {
        x = forward_layer(x, weights_.layers[layer_idx], layer_idx, start_pos, kv_cache);
        
        // DEBUG: Track hidden state growth through layers
        if (start_pos == 0 && seq_len == 3 && layer_idx <= 5) {
            const float* x_data = x.data_f32();
            float norm_pos2 = 0.0f;
            for (int d = 0; d < config_.hidden_dim; d++) {
                float val = x_data[2 * config_.hidden_dim + d];  // Position 2
                norm_pos2 += val * val;
            }
            norm_pos2 = std::sqrt(norm_pos2);
            std::cout << "After layer " << layer_idx << ", position 2 L2 norm: " << norm_pos2 << "\n";
        }
        
        // DEBUG: Print layer 0 output norms
        if (layer_idx == 0 && start_pos == 0 && seq_len == 3) {
            std::cout << "\n=== LAYER 0 OUTPUT ===\n";
            for (int i = 0; i < seq_len; i++) {
                const float* x_data = x.data_f32() + i * config_.hidden_dim;
                float norm = 0.0f;
                for (int d = 0; d < config_.hidden_dim; d++) {
                    norm += x_data[d] * x_data[d];
                }
                norm = std::sqrt(norm);
                std::cout << "Position " << i << " norm: " << norm << ", first 10: [";
                for (int d = 0; d < 10; d++) {
                    std::cout << x_data[d];
                    if (d < 9) std::cout << ", ";
                }
                std::cout << "]\n";
            }
        }
    }
    
    // DEBUG: Print before final RMSNorm
    if (start_pos == 0 && seq_len == 3) {
        std::cout << "\n=== BEFORE FINAL RMSNORM ===\n";
        const float* x_data = x.data_f32();
        float norm_pos2 = 0.0f;
        for (int d = 0; d < config_.hidden_dim; d++) {
            float val = x_data[2 * config_.hidden_dim + d];  // Position 2
            norm_pos2 += val * val;
        }
        norm_pos2 = std::sqrt(norm_pos2);
        std::cout << "Position 2 L2 norm: " << norm_pos2 << "\n";
    }
    
    // Final RMSNorm
    x = rmsnorm(x, weights_.output_norm, config_.rms_norm_eps);
    
    // DEBUG: Check hidden state before output projection during generation
    if (start_pos >= 10 && start_pos <= 15 && seq_len == 1) {
        const float* x_data = x.data_f32();
        float x_min = INFINITY, x_max = -INFINITY, x_sum = 0.0f;
        for (int d = 0; d < config_.hidden_dim; d++) {
            float val = x_data[d];
            x_min = std::min(x_min, val);
            x_max = std::max(x_max, val);
            x_sum += val;
        }
        std::cout << "\n=== HIDDEN STATE BEFORE OUTPUT PROJ @ POS " << start_pos << " ===\n";
        std::cout << "Stats: min=" << x_min << ", max=" << x_max << ", mean=" << (x_sum / config_.hidden_dim) << "\n";
        std::cout << "First 10 values: ";
        for (int d = 0; d < 10; d++) {
            std::cout << x_data[d] << " ";
        }
        std::cout << "\n";
    }
    
    // DEBUG: Print after final RMSNorm
    if (start_pos == 0 && seq_len == 3) {
        std::cout << "\n=== AFTER FINAL RMSNORM ===\n";
        const float* x_data = x.data_f32();
        float norm_pos2 = 0.0f;
        for (int d = 0; d < config_.hidden_dim; d++) {
            float val = x_data[2 * config_.hidden_dim + d];  // Position 2
            norm_pos2 += val * val;
        }
        norm_pos2 = std::sqrt(norm_pos2);
        std::cout << "Position 2 L2 norm: " << norm_pos2 << "\n";
        
        // Print output_norm weight stats
        std::cout << "output_norm.weight L2: ";
        const float* norm_weights = weights_.output_norm.data_f32();
        float norm_l2 = 0.0f;
        for (int d = 0; d < config_.hidden_dim; d++) {
            norm_l2 += norm_weights[d] * norm_weights[d];
        }
        std::cout << std::sqrt(norm_l2) << "\n";
    }
    
    // Output projection: [seq_len, hidden_dim] @ [vocab_size, hidden_dim]^T = [seq_len, vocab_size]
    Tensor logits = matmul_transposed(x, weights_.output, false, true);
    
    // DEBUG: Print logits at position 2
    if (start_pos == 0 && seq_len == 3) {
        std::cout << "\n=== LOGITS AT POSITION 2 ===\n";
        const float* logits_data = logits.data_f32();
        int vocab_size = config_.vocab_size;
        
        // Find top 5
        std::vector<std::pair<float, int>> logit_pairs;
        for (int v = 0; v < vocab_size; v++) {
            logit_pairs.push_back({logits_data[2 * vocab_size + v], v});
        }
        std::sort(logit_pairs.begin(), logit_pairs.end(), std::greater<>());
        
        std::cout << "Top-5:\n";
        for (int i = 0; i < 5; i++) {
            std::cout << "  Token " << logit_pairs[i].second << ": " << logit_pairs[i].first << "\n";
        }
        
        // Check specific tokens
        std::cout << "Token 220: " << logits_data[2 * vocab_size + 220] << "\n";
        std::cout << "Token 1124: " << logits_data[2 * vocab_size + 1124] << "\n";
    }
    
    return logits;
}

Tensor InferenceEngine::forward_layer(
    const Tensor& x,
    const LayerWeights& layer,
    int layer_idx,
    int pos,
    KVCache* kv_cache
) {
    int seq_len = x.shape().dims[0];
    
    // DEBUG: Check input shape and VALUES during generation
    if (layer_idx == 0 && pos >= 10 && pos <= 15) {
        std::cout << "\n=== LAYER 0 INPUT @ POS " << pos << " ===\n";
        std::cout << "x shape: [" << x.shape().dims[0] << ", " << x.shape().dims[1] << "]\n";
        std::cout << "seq_len=" << seq_len << ", pos=" << pos << "\n";
        
        // CRITICAL: Check KV cache size to see if it's growing!
        if (kv_cache) {
            std::cout << "KV cache seq_len: " << kv_cache->seq_len() << "\n";
        } else {
            std::cout << "KV cache: nullptr\n";
        }
        
        // Print first 10 values (this should be the token embedding)
        const float* x_data = x.data_f32();
        std::cout << "First 10 embedding values: ";
        for (int i = 0; i < 10; ++i) {
            std::cout << x_data[i] << " ";
        }
        std::cout << "\n";
    }
    
    // DEBUG: Print layer 0 attn_norm weights
    if (layer_idx == 0 && pos == 0 && seq_len == 3) {
        const float* norm_weights = layer.attn_norm.data_f32();
        float norm = 0.0f;
        for (int d = 0; d < config_.hidden_dim; d++) {
            norm += norm_weights[d] * norm_weights[d];
        }
        norm = std::sqrt(norm);
        std::cout << "\n=== LAYER 0 ATTN_NORM WEIGHTS ===\n";
        std::cout << "L2 norm: " << norm << ", first 10: [";
        for (int d = 0; d < 10; d++) {
            std::cout << norm_weights[d];
            if (d < 9) std::cout << ", ";
        }
        std::cout << "]\n";
    }
    
    // Pre-attention RMSNorm
    Tensor x_norm = rmsnorm(x, layer.attn_norm, config_.rms_norm_eps);
    
    // DEBUG: Layer 1 input
    if (layer_idx == 1 && pos == 0 && seq_len == 3) {
        std::cout << "\n=== LAYER 1 INPUT ===\n";
        const float* x_data = x.data_f32();
        float norm = 0.0f;
        for (int d = 0; d < config_.hidden_dim; d++) {
            float val = x_data[2 * config_.hidden_dim + d];
            norm += val * val;
        }
        std::cout << "Position 2 L2 norm: " << std::sqrt(norm) << "\n";
    }
    
    // Attention (projects Q,K,V internally)
    Tensor attn_out = attention_->forward(
        x_norm, 
        layer.wq, layer.wk, layer.wv, layer.wo,
        layer.bq, layer.bk, layer.bv,  // Add biases
        layer_idx,
        pos,
        kv_cache
    );
    
    // DEBUG: Layer 1 attention output
    if (layer_idx == 1 && pos == 0 && seq_len == 3) {
        std::cout << "\n=== LAYER 1 ATTENTION OUTPUT ===\n";
        const float* attn_data = attn_out.data_f32();
        float norm = 0.0f;
        for (int d = 0; d < config_.hidden_dim; d++) {
            float val = attn_data[2 * config_.hidden_dim + d];
            norm += val * val;
        }
        std::cout << "Position 2 L2 norm: " << std::sqrt(norm) << "\n";
    }
    
    // DEBUG: Print attention output for layer 0
    if (layer_idx == 0 && pos == 0 && seq_len == 3) {
        std::cout << "\n=== LAYER 0 ATTENTION OUTPUT ===\n";
        for (int i = 0; i < seq_len; i++) {
            const float* attn_data = attn_out.data_f32() + i * config_.hidden_dim;
            float norm = 0.0f;
            for (int d = 0; d < config_.hidden_dim; d++) {
                norm += attn_data[d] * attn_data[d];
            }
            norm = std::sqrt(norm);
            std::cout << "Position " << i << " norm: " << norm << "\n";
        }
    }
    
    // Residual connection
    Tensor x_attn = add(x, attn_out);
    
    // DEBUG: Compare x_attn at layer 0 during generation
    if (layer_idx == 0 && pos >= 10 && pos <= 15 && seq_len == 1) {
        const float* x_attn_data = x_attn.data_f32();
        float x_attn_min = INFINITY, x_attn_max = -INFINITY, x_attn_sum = 0.0f;
        for (int d = 0; d < config_.hidden_dim; d++) {
            float val = x_attn_data[d];
            x_attn_min = std::min(x_attn_min, val);
            x_attn_max = std::max(x_attn_max, val);
            x_attn_sum += val;
        }
        std::cout << "\n=== LAYER 0 X_ATTN @ POS " << pos << " (after residual) ===\n";
        std::cout << "Stats: min=" << x_attn_min << ", max=" << x_attn_max << ", mean=" << (x_attn_sum / config_.hidden_dim) << "\n";
        std::cout << "First 5 values: ";
        for (int d = 0; d < 5; d++) {
            std::cout << x_attn_data[d] << " ";
        }
        std::cout << "\n";
    }
    
    // DEBUG: Print x_attn (before RMSNorm) for layer 0
    if (layer_idx == 0 && pos == 0 && seq_len == 3) {
        std::cout << "\n=== LAYER 0 X_ATTN (before RMSNorm) ===\n";
        for (int i = 0; i < seq_len; i++) {
            const float* x_attn_data = x_attn.data_f32() + i * config_.hidden_dim;
            float norm = 0.0f;
            for (int d = 0; d < config_.hidden_dim; d++) {
                norm += x_attn_data[d] * x_attn_data[d];
            }
            norm = std::sqrt(norm);
            std::cout << "Position " << i << " norm: " << norm << "\n";
        }
    }
    
    // Pre-FFN RMSNorm
    Tensor x_attn_norm = rmsnorm(x_attn, layer.ffn_norm, config_.rms_norm_eps);
    
    // DEBUG: Check ffn_norm weights for layer 0 and 1
    if ((layer_idx == 0 || layer_idx == 1) && pos == 0 && seq_len == 3) {
        const float* norm_weights = layer.ffn_norm.data_f32();
        float norm_weight_l2 = 0.0f;
        for (int d = 0; d < config_.hidden_dim; d++) {
            norm_weight_l2 += norm_weights[d] * norm_weights[d];
        }
        std::cout << "\n=== LAYER " << layer_idx << " FFN_NORM WEIGHT ===\n";
        std::cout << "L2 norm: " << std::sqrt(norm_weight_l2) << "\n";
        std::cout << "First 10 values: ";
        for (int d = 0; d < 10; d++) {
            std::cout << norm_weights[d];
            if (d < 9) std::cout << ", ";
        }
        std::cout << "\n";
    }
    
    // DEBUG: Layer 1 FFN input (after RMSNorm)
    if (layer_idx == 1 && pos == 0 && seq_len == 3) {
        std::cout << "\n=== LAYER 1 FFN INPUT (x_attn_norm) ===\n";
        const float* norm_data = x_attn_norm.data_f32();
        float norm = 0.0f;
        for (int d = 0; d < config_.hidden_dim; d++) {
            float val = norm_data[2 * config_.hidden_dim + d];
            norm += val * val;
        }
        std::cout << "Position 2 L2 norm: " << std::sqrt(norm) << "\n";
        
        // Also print x_attn (before RMSNorm)
        const float* x_attn_data = x_attn.data_f32();
        float x_attn_norm_val = 0.0f;
        for (int d = 0; d < config_.hidden_dim; d++) {
            float val = x_attn_data[2 * config_.hidden_dim + d];
            x_attn_norm_val += val * val;
        }
        std::cout << "x_attn (before RMSNorm) L2 norm: " << std::sqrt(x_attn_norm_val) << "\n";
    }
    
    // FFN with residual
    Tensor ffn_out = forward_ffn(x_attn_norm, layer);
    
    // DEBUG: FFN output at layer 0 during generation
    if (layer_idx == 0 && pos == 10 && seq_len == 1) {
        const float* ffn_data = ffn_out.data_f32();
        float ffn_min = INFINITY, ffn_max = -INFINITY, ffn_sum = 0.0f;
        for (int d = 0; d < config_.hidden_dim; d++) {
            float val = ffn_data[d];
            ffn_min = std::min(ffn_min, val);
            ffn_max = std::max(ffn_max, val);
            ffn_sum += val;
        }
        std::cout << "=== LAYER 0 FFN_OUT @ POS 10 ===\n";
        std::cout << "Stats: min=" << ffn_min << ", max=" << ffn_max << ", mean=" << (ffn_sum / config_.hidden_dim) << "\n";
    }
    
    // DEBUG: Layer 1 FFN output
    if (layer_idx == 1 && pos == 0 && seq_len == 3) {
        std::cout << "\n=== LAYER 1 FFN OUTPUT ===\n";
        const float* ffn_data = ffn_out.data_f32();
        float norm = 0.0f;
        for (int d = 0; d < config_.hidden_dim; d++) {
            float val = ffn_data[2 * config_.hidden_dim + d];
            norm += val * val;
        }
        std::cout << "Position 2 L2 norm: " << std::sqrt(norm) << "\n";
    }
    
    Tensor x_out = add(x_attn, ffn_out);
    
    // DEBUG: Layer 1 final output
    if (layer_idx == 1 && pos == 0 && seq_len == 3) {
        std::cout << "\n=== LAYER 1 FINAL OUTPUT ===\n";
        const float* out_data = x_out.data_f32();
        float norm = 0.0f;
        for (int d = 0; d < config_.hidden_dim; d++) {
            float val = out_data[2 * config_.hidden_dim + d];
            norm += val * val;
        }
        std::cout << "Position 2 L2 norm: " << std::sqrt(norm) << "\n";
    }
    
    return x_out;
}

Tensor InferenceEngine::forward_ffn(const Tensor& x, const LayerWeights& layer) {
    // SwiGLU: silu(gate(x)) * up(x) @ down
    // Weights are stored as [out_features, in_features] so we need to transpose
    Tensor gate = matmul_transposed(x, layer.w_gate, false, true);  // x @ w_gate^T
    Tensor up = matmul_transposed(x, layer.w_up, false, true);      // x @ w_up^T
    
    // Apply SiLU to gate
    gate = silu(gate);
    
    // Element-wise multiply
    Tensor gated = multiply(gate, up);
    
    // Down projection
    Tensor out = matmul_transposed(gated, layer.w_down, false, true);  // gated @ w_down^T
    
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
    const size_t prompt_len = tokens.size();
    
    // Build effective stop token set: caller's list + model EOS + <|im_end|> (151645)
    SamplingConfig effective_sampling = sampling;
    auto& stop = effective_sampling.stop_tokens;
    auto add_stop = [&](TokenID t) {
        if (t != 0 && std::find(stop.begin(), stop.end(), t) == stop.end())
            stop.push_back(t);
    };
    add_stop(tokenizer_->special_tokens().eos_token);
    add_stop(151645);  // <|im_end|> — Qwen2.5 chat end-of-turn token
    
    // Create KV cache for this generation
    // Constructor: KVCache(max_seq_len, n_layers, n_kv_heads, head_dim)
    KVCache kv_cache(config_.max_seq_len, config_.n_layers, config_.n_kv_heads, config_.head_dim);
    
    // PREFILL PHASE: Process entire prompt to fill KV cache
    Tensor logits = forward(tokens, &kv_cache, 0);  // start_pos = 0
    
    // Get first generated token from last position
    int seq_len = logits.shape().dims[0];
    int vocab_size = logits.shape().dims[1];
    Tensor last_logits = Tensor::empty({vocab_size}, DType::F32);
    const float* logits_data = logits.data_f32() + (seq_len - 1) * vocab_size;
    std::memcpy(last_logits.data_f32(), logits_data, vocab_size * sizeof(float));
    
    TokenID next_token = sample_token(last_logits, effective_sampling);
    
    std::cout << "." << std::flush;
    
    if (inference_utils::is_stop_token(next_token, effective_sampling.stop_tokens)) {
        result.stopped_early = true;
        result.stop_reason = "stop_token";
        result.text = tokenizer_->decode({tokens.begin() + prompt_len, tokens.end()}, true);
        result.tokens = tokens;
        auto end_time = std::chrono::high_resolution_clock::now();
        result.generation_time_ms = std::chrono::duration<float, std::milli>(end_time - start_time).count();
        return result;
    }
    
    tokens.push_back(next_token);
    result.tokens_generated++;
    Logger::instance().debug("Generated token[0]: " + std::to_string(next_token) + " = '" + tokenizer_->decode({next_token}) + "'");
    
    // GENERATION PHASE: Generate one token at a time
    for (int i = 1; i < effective_sampling.max_tokens; i++) {
        next_token = generate_next_token(tokens, effective_sampling, &kv_cache);
        
        // Simple progress indicator
        std::cout << "." << std::flush;
        
        // Check stop conditions
        if (inference_utils::is_stop_token(next_token, effective_sampling.stop_tokens)) {
            result.stopped_early = true;
            result.stop_reason = "stop_token";
            break;
        }
        
        tokens.push_back(next_token);
        result.tokens_generated++;
        Logger::instance().debug("Generated token[" + std::to_string(i) + "]: " + std::to_string(next_token) + " = '" + tokenizer_->decode({next_token}) + "'");
        
        // Check max length
        if (tokens.size() >= (size_t)config_.max_seq_len) {
            result.stopped_early = true;
            result.stop_reason = "max_length";
            break;
        }
    }
    std::cout << "\n";
    
    // Decode only generated tokens (skip prompt + special tokens like <|im_start|>)
    result.text = tokenizer_->decode({tokens.begin() + prompt_len, tokens.end()}, true);
    result.tokens = tokens;
    
    auto end_time = std::chrono::high_resolution_clock::now();
    result.generation_time_ms = std::chrono::duration<float, std::milli>(end_time - start_time).count();
    
    if (!result.stopped_early) {
        result.stop_reason = "max_tokens";
    }
    
    return result;
}

GenerationResult InferenceEngine::generate_from_tokens(
    const std::vector<TokenID>& prompt_tokens,
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
    
    if (prompt_tokens.empty()) {
        result.stop_reason = "empty_prompt";
        return result;
    }
    
    // Start with the provided tokens
    std::vector<TokenID> tokens = prompt_tokens;
    
    // Create KV cache
    KVCache kv_cache(config_.max_seq_len, config_.n_layers, config_.n_kv_heads, config_.head_dim);
    
    // PREFILL PHASE: Process entire prompt to fill KV cache
    Tensor logits = forward(tokens, &kv_cache, 0);
    
    // Get first generated token from last position
    int seq_len = logits.shape().dims[0];
    int vocab_size = logits.shape().dims[1];
    Tensor last_logits = Tensor::empty({vocab_size}, DType::F32);
    const float* logits_data = logits.data_f32() + (seq_len - 1) * vocab_size;
    std::memcpy(last_logits.data_f32(), logits_data, vocab_size * sizeof(float));
    TokenID next_token = sample_token(last_logits, sampling);
    
    std::cout << "." << std::flush;
    
    if (inference_utils::is_stop_token(next_token, sampling.stop_tokens)) {
        result.stopped_early = true;
        result.stop_reason = "stop_token";
        result.text = tokenizer_->decode(tokens);
        result.tokens = tokens;
        auto end_time = std::chrono::high_resolution_clock::now();
        result.generation_time_ms = std::chrono::duration<float, std::milli>(end_time - start_time).count();
        return result;
    }
    
    tokens.push_back(next_token);
    result.tokens_generated++;
    
    // GENERATION PHASE: Generate one token at a time
    for (int i = 1; i < sampling.max_tokens; i++) {
        next_token = generate_next_token(tokens, sampling, &kv_cache);
        
        std::cout << "." << std::flush;
        
        if (inference_utils::is_stop_token(next_token, sampling.stop_tokens)) {
            result.stopped_early = true;
            result.stop_reason = "stop_token";
            break;
        }
        
        tokens.push_back(next_token);
        result.tokens_generated++;
        
        if (tokens.size() >= (size_t)config_.max_seq_len) {
            result.stopped_early = true;
            result.stop_reason = "max_length";
            break;
        }
    }
    std::cout << "\n";
    
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
    // CRITICAL: Only pass the LAST token for generation!
    // KV cache already has all previous tokens' keys/values
    // Processing all tokens would be O(n^2) complexity
    std::vector<TokenID> new_token_vec = {context.back()};
    int start_pos = context.size() - 1;  // Position of the last token (the one we're processing)
    
    Logger::instance().debug("generate_next_token: pos=" + std::to_string(start_pos) + 
                            ", token=" + std::to_string(new_token_vec[0]) + 
                            ", kv_cache_len=" + std::to_string(kv_cache ? kv_cache->seq_len() : -1));
    
    // Forward pass with just the new token at correct position
    Tensor logits = forward(new_token_vec, kv_cache, start_pos);
    
    // Get logits: [1, vocab_size]
    int vocab_size = logits.shape().dims[1];
    
    Tensor last_logits = Tensor::empty({vocab_size}, DType::F32);
    const float* logits_data = logits.data_f32();
    std::memcpy(last_logits.data_f32(), logits_data, vocab_size * sizeof(float));
    
    // DEBUG: Print top-5 logits at every position during first 20 tokens
    if (start_pos < 20) {
        const float* logit_ptr = last_logits.data_f32();
        std::vector<std::pair<float, int>> top_logits;
        for (int i = 0; i < vocab_size; i++) {
            top_logits.push_back({logit_ptr[i], i});
        }
        std::partial_sort(top_logits.begin(), top_logits.begin() + 5, top_logits.end(),
                         [](const auto& a, const auto& b) { return a.first > b.first; });
        std::cout << "[Pos " << start_pos << "] Top-5 logits: ";
        for (int i = 0; i < 5; i++) {
            std::cout << top_logits[i].second << "(" << top_logits[i].first << ") ";
        }
        std::cout << std::endl;
    }
    
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
