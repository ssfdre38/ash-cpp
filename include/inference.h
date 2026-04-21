#pragma once

#include "tensor.h"
#include "attention.h"
#include "tokenizer.h"
#include "gguf_parser.h"
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>

namespace ash {

// Model architecture config
struct ModelConfig {
    std::string architecture;  // "gemma", "llama", etc.
    int vocab_size;
    int hidden_dim;
    int intermediate_dim;      // FFN intermediate size
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int head_dim;
    int max_seq_len;
    float rope_theta;
    float rms_norm_eps;
    
    // Derived
    AttentionConfig attention_config() const {
        AttentionConfig config;
        config.n_heads = n_heads;
        config.n_kv_heads = n_kv_heads;
        config.head_dim = head_dim;
        config.max_seq_len = max_seq_len;
        config.rope_theta = rope_theta;
        config.use_kv_cache = true;
        return config;
    }
};

// Sampling configuration
struct SamplingConfig {
    float temperature = 0.7f;   // Randomness (0 = greedy, higher = more random)
    float top_p = 0.9f;         // Nucleus sampling threshold
    int top_k = 40;             // Top-K sampling
    int max_tokens = 512;       // Max tokens to generate
    bool use_sampling = true;   // False = greedy (argmax)
    
    std::vector<TokenID> stop_tokens;  // Stop generation on these tokens
};

// Generation result
struct GenerationResult {
    std::string text;
    std::vector<TokenID> tokens;
    int tokens_generated;
    float generation_time_ms;
    bool stopped_early;  // Hit stop token or max length
    std::string stop_reason;
};

// Transformer layer weights
struct LayerWeights {
    // Attention
    Tensor wq;  // Query projection
    Tensor wk;  // Key projection
    Tensor wv;  // Value projection
    Tensor wo;  // Output projection
    
    // Feed-forward
    Tensor w_gate;  // Gate projection
    Tensor w_up;    // Up projection
    Tensor w_down;  // Down projection
    
    // Layer norms
    Tensor attn_norm;  // Pre-attention RMSNorm
    Tensor ffn_norm;   // Pre-FFN RMSNorm
};

// Full model weights
struct ModelWeights {
    Tensor token_embeddings;     // [vocab_size, hidden_dim]
    std::vector<LayerWeights> layers;
    Tensor output_norm;          // Final RMSNorm
    Tensor output;               // Output projection (often tied to embeddings)
    
    bool loaded = false;
};

// Inference engine - runs transformer forward pass
class InferenceEngine {
public:
    InferenceEngine();
    ~InferenceEngine();
    
    // Load model from GGUF file
    bool load_model(const std::string& gguf_path);
    
    // Check if model is loaded
    bool is_loaded() const { return weights_.loaded; }
    
    // Get model config
    const ModelConfig& config() const { return config_; }
    
    // Forward pass through model
    // tokens: [seq_len] input token IDs
    // Returns: [seq_len, vocab_size] logits
    Tensor forward(const std::vector<TokenID>& tokens, KVCache* kv_cache = nullptr);
    
    // Generate text from prompt
    GenerationResult generate(
        const std::string& prompt,
        const SamplingConfig& sampling = SamplingConfig()
    );
    
    // Generate next token only (for streaming)
    TokenID generate_next_token(
        const std::vector<TokenID>& context,
        const SamplingConfig& sampling,
        KVCache* kv_cache = nullptr
    );
    
    // Get tokenizer
    Tokenizer* tokenizer() { return tokenizer_.get(); }
    
    // Sampling methods (exposed for testing)
    TokenID sample_token(const Tensor& logits, const SamplingConfig& config);
    TokenID sample_greedy(const Tensor& logits);
    TokenID sample_top_k(const Tensor& logits, int k, float temperature);
    TokenID sample_top_p(const Tensor& logits, float p, float temperature);

private:
    ModelConfig config_;
    ModelWeights weights_;
    std::unique_ptr<Tokenizer> tokenizer_;
    std::unique_ptr<MultiHeadAttention> attention_;
    
    // Load weights from GGUF parser
    bool load_weights_from_gguf(GGUFParser& parser);
    
    // Forward pass through single layer
    Tensor forward_layer(
        const Tensor& x,
        const LayerWeights& layer,
        int layer_idx,
        int pos,
        KVCache* kv_cache
    );
    
    // Feed-forward network (SwiGLU)
    Tensor forward_ffn(const Tensor& x, const LayerWeights& layer);
};

// Inference utilities
namespace inference_utils {
    // Apply temperature to logits
    void apply_temperature(Tensor& logits, float temperature);
    
    // Get top-K indices
    std::vector<int> top_k_indices(const Tensor& logits, int k);
    
    // Compute softmax temperature
    Tensor softmax_temperature(const Tensor& logits, float temperature);
    
    // Check if token is stop token
    bool is_stop_token(TokenID token, const std::vector<TokenID>& stop_tokens);
}

} // namespace ash
