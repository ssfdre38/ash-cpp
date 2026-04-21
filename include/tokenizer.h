#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <memory>

namespace ash {

// Token ID type
using TokenID = int32_t;

// Special tokens
struct SpecialTokens {
    TokenID bos_token = 1;   // Beginning of sequence
    TokenID eos_token = 2;   // End of sequence
    TokenID pad_token = 0;   // Padding
    TokenID unk_token = 3;   // Unknown
};

// Tokenizer interface
class Tokenizer {
public:
    virtual ~Tokenizer() = default;
    
    // Encode text to token IDs
    virtual std::vector<TokenID> encode(const std::string& text, bool add_bos = true, bool add_eos = false) = 0;
    
    // Decode token IDs to text
    virtual std::string decode(const std::vector<TokenID>& tokens, bool skip_special = true) = 0;
    
    // Decode single token
    virtual std::string decode_token(TokenID token) = 0;
    
    // Get vocabulary size
    virtual size_t vocab_size() const = 0;
    
    // Get special tokens
    virtual const SpecialTokens& special_tokens() const = 0;
    
    // Check if token is special
    virtual bool is_special(TokenID token) const = 0;
};

// Simple SentencePiece-like tokenizer
// Simplified for Gemma models - full SentencePiece is complex
class SimpleSPTokenizer : public Tokenizer {
public:
    SimpleSPTokenizer();
    ~SimpleSPTokenizer() override;
    
    // Load vocabulary from GGUF file
    bool load_from_gguf(const std::string& gguf_path);
    
    // Load vocabulary from text file (one token per line)
    bool load_from_text(const std::string& vocab_path);
    
    // Encode/decode
    std::vector<TokenID> encode(const std::string& text, bool add_bos = true, bool add_eos = false) override;
    std::string decode(const std::vector<TokenID>& tokens, bool skip_special = true) override;
    std::string decode_token(TokenID token) override;
    
    // Vocab info
    size_t vocab_size() const override;
    const SpecialTokens& special_tokens() const override;
    bool is_special(TokenID token) const override;
    
    // Manual vocab building (for testing)
    void add_token(TokenID id, const std::string& token, float score = 0.0f);
    void set_special_tokens(const SpecialTokens& tokens);
    void mark_loaded();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    
    // Helper: greedy tokenization (simplified)
    std::vector<TokenID> tokenize_greedy(const std::string& text);
    
    // Helper: byte fallback encoding
    std::vector<TokenID> encode_bytes(const std::string& text);
};

// Tokenizer factory
class TokenizerFactory {
public:
    // Create tokenizer from GGUF model file
    static std::unique_ptr<Tokenizer> from_gguf(const std::string& gguf_path);
    
    // Create tokenizer from SentencePiece model
    static std::unique_ptr<Tokenizer> from_sentencepiece(const std::string& sp_model_path);
    
    // Create simple test tokenizer (for development)
    static std::unique_ptr<Tokenizer> create_test_tokenizer();
};

} // namespace ash
