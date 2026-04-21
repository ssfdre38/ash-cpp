#include "tokenizer.h"
#include "gguf_parser.h"
#include "logger.h"
#include <algorithm>
#include <sstream>
#include <cctype>

namespace ash {

struct SimpleSPTokenizer::Impl {
    std::unordered_map<TokenID, std::string> id_to_token;
    std::unordered_map<std::string, TokenID> token_to_id;
    std::unordered_map<TokenID, float> token_scores;
    SpecialTokens special;
    
    bool is_loaded = false;
};

SimpleSPTokenizer::SimpleSPTokenizer() : impl_(std::make_unique<Impl>()) {}
SimpleSPTokenizer::~SimpleSPTokenizer() = default;

bool SimpleSPTokenizer::load_from_gguf(const std::string& gguf_path) {
    Logger::instance().info("Loading tokenizer from GGUF: " + gguf_path);
    
    GGUFParser parser;
    if (!parser.parse(gguf_path)) {
        Logger::instance().error("Failed to parse GGUF for tokenizer");
        return false;
    }
    
    // Get vocab size
    size_t vocab_size = parser.get_vocab_size();
    Logger::instance().info("Vocab size: " + std::to_string(vocab_size));
    
    // Read tokens from metadata
    // GGUF stores vocab as arrays: tokenizer.ggml.tokens, tokenizer.ggml.scores, etc.
    GGUFMetadataValue tokens_array, scores_array;
    
    if (!parser.get_metadata("tokenizer.ggml.tokens", tokens_array)) {
        Logger::instance().warning("No tokens array in GGUF, using placeholder vocab");
        // Create minimal placeholder vocab for testing
        for (size_t i = 0; i < std::min<size_t>(1000, vocab_size); ++i) {
            add_token(static_cast<TokenID>(i), "<token_" + std::to_string(i) + ">", 0.0f);
        }
        impl_->is_loaded = true;
        return true;
    }
    
    if (parser.get_metadata("tokenizer.ggml.scores", scores_array)) {
        // Load tokens with scores
        for (size_t i = 0; i < tokens_array.array_value.size(); ++i) {
            TokenID id = static_cast<TokenID>(i);
            std::string token = tokens_array.array_value[i].string_value;
            float score = (i < scores_array.array_value.size()) ? 
                          scores_array.array_value[i].float_value : 0.0f;
            add_token(id, token, score);
        }
    } else {
        // Load tokens without scores
        for (size_t i = 0; i < tokens_array.array_value.size(); ++i) {
            TokenID id = static_cast<TokenID>(i);
            std::string token = tokens_array.array_value[i].string_value;
            add_token(id, token, 0.0f);
        }
    }
    
    // Read special tokens
    SpecialTokens special;
    special.bos_token = static_cast<TokenID>(parser.get_uint("tokenizer.ggml.bos_token_id", 1));
    special.eos_token = static_cast<TokenID>(parser.get_uint("tokenizer.ggml.eos_token_id", 2));
    special.pad_token = static_cast<TokenID>(parser.get_uint("tokenizer.ggml.padding_token_id", 0));
    special.unk_token = static_cast<TokenID>(parser.get_uint("tokenizer.ggml.unknown_token_id", 3));
    set_special_tokens(special);
    
    impl_->is_loaded = true;
    Logger::instance().info("✅ Tokenizer loaded: " + std::to_string(impl_->id_to_token.size()) + " tokens");
    
    return true;
}

bool SimpleSPTokenizer::load_from_text(const std::string& vocab_path) {
    Logger::instance().warning("Text vocab loading not yet implemented");
    return false;
}

void SimpleSPTokenizer::add_token(TokenID id, const std::string& token, float score) {
    impl_->id_to_token[id] = token;
    impl_->token_to_id[token] = id;
    impl_->token_scores[id] = score;
}

void SimpleSPTokenizer::set_special_tokens(const SpecialTokens& tokens) {
    impl_->special = tokens;
}

void SimpleSPTokenizer::mark_loaded() {
    impl_->is_loaded = true;
}

std::vector<TokenID> SimpleSPTokenizer::tokenize_greedy(const std::string& text) {
    // Simplified greedy tokenization
    // Real SentencePiece uses BPE/Unigram with scores
    
    std::vector<TokenID> result;
    size_t pos = 0;
    
    while (pos < text.length()) {
        // Try to match longest token
        size_t best_len = 0;
        TokenID best_token = impl_->special.unk_token;
        
        for (size_t len = std::min<size_t>(20, text.length() - pos); len > 0; --len) {
            std::string substr = text.substr(pos, len);
            auto it = impl_->token_to_id.find(substr);
            if (it != impl_->token_to_id.end()) {
                best_len = len;
                best_token = it->second;
                break;
            }
        }
        
        if (best_len == 0) {
            // No match - encode as byte
            unsigned char byte = static_cast<unsigned char>(text[pos]);
            std::string byte_token = "<0x" + std::to_string(byte) + ">";
            auto it = impl_->token_to_id.find(byte_token);
            if (it != impl_->token_to_id.end()) {
                result.push_back(it->second);
            } else {
                result.push_back(impl_->special.unk_token);
            }
            pos++;
        } else {
            result.push_back(best_token);
            pos += best_len;
        }
    }
    
    return result;
}

std::vector<TokenID> SimpleSPTokenizer::encode_bytes(const std::string& text) {
    // Fallback: encode each byte as token
    std::vector<TokenID> result;
    for (unsigned char c : text) {
        // Simple byte mapping (offset by 256)
        result.push_back(static_cast<TokenID>(c + 256));
    }
    return result;
}

std::vector<TokenID> SimpleSPTokenizer::encode(const std::string& text, bool add_bos, bool add_eos) {
    if (!impl_->is_loaded) {
        Logger::instance().error("Tokenizer not loaded");
        return {};
    }
    
    std::vector<TokenID> tokens;
    
    if (add_bos) {
        tokens.push_back(impl_->special.bos_token);
    }
    
    // Tokenize text
    auto text_tokens = tokenize_greedy(text);
    tokens.insert(tokens.end(), text_tokens.begin(), text_tokens.end());
    
    if (add_eos) {
        tokens.push_back(impl_->special.eos_token);
    }
    
    return tokens;
}

std::string SimpleSPTokenizer::decode(const std::vector<TokenID>& tokens, bool skip_special) {
    std::stringstream ss;
    
    for (TokenID token : tokens) {
        if (skip_special && is_special(token)) {
            continue;
        }
        
        auto it = impl_->id_to_token.find(token);
        if (it != impl_->id_to_token.end()) {
            std::string token_str = it->second;
            
            // Handle special formatting (SentencePiece uses U+2581 for spaces)
            // Just pass through for now - proper handling needs UTF-8
            
            ss << token_str;
        } else {
            ss << "<UNK>";
        }
    }
    
    return ss.str();
}

std::string SimpleSPTokenizer::decode_token(TokenID token) {
    auto it = impl_->id_to_token.find(token);
    if (it != impl_->id_to_token.end()) {
        return it->second;
    }
    return "<UNK>";
}

size_t SimpleSPTokenizer::vocab_size() const {
    return impl_->id_to_token.size();
}

const SpecialTokens& SimpleSPTokenizer::special_tokens() const {
    return impl_->special;
}

bool SimpleSPTokenizer::is_special(TokenID token) const {
    return token == impl_->special.bos_token ||
           token == impl_->special.eos_token ||
           token == impl_->special.pad_token ||
           token == impl_->special.unk_token;
}

// TokenizerFactory implementation
std::unique_ptr<Tokenizer> TokenizerFactory::from_gguf(const std::string& gguf_path) {
    auto tokenizer = std::make_unique<SimpleSPTokenizer>();
    if (tokenizer->load_from_gguf(gguf_path)) {
        return tokenizer;
    }
    return nullptr;
}

std::unique_ptr<Tokenizer> TokenizerFactory::from_sentencepiece(const std::string& sp_model_path) {
    Logger::instance().warning("SentencePiece model loading not yet implemented");
    return nullptr;
}

std::unique_ptr<Tokenizer> TokenizerFactory::create_test_tokenizer() {
    auto tokenizer = std::make_unique<SimpleSPTokenizer>();
    
    // Create minimal test vocab
    tokenizer->add_token(0, "<pad>", 0.0f);
    tokenizer->add_token(1, "<bos>", 0.0f);
    tokenizer->add_token(2, "<eos>", 0.0f);
    tokenizer->add_token(3, "<unk>", 0.0f);
    
    // Common words/subwords
    tokenizer->add_token(4, "▁the", 0.0f);
    tokenizer->add_token(5, "▁a", 0.0f);
    tokenizer->add_token(6, "▁is", 0.0f);
    tokenizer->add_token(7, "▁", 0.0f); // Space
    tokenizer->add_token(8, "hello", 0.0f);
    tokenizer->add_token(9, "world", 0.0f);
    tokenizer->add_token(10, "test", 0.0f);
    
    // Letters
    for (char c = 'a'; c <= 'z'; ++c) {
        tokenizer->add_token(11 + (c - 'a'), std::string(1, c), 0.0f);
    }
    
    SpecialTokens special;
    special.pad_token = 0;
    special.bos_token = 1;
    special.eos_token = 2;
    special.unk_token = 3;
    tokenizer->set_special_tokens(special);
    
    // Mark as loaded
    tokenizer->mark_loaded();
    
    return tokenizer;
}

} // namespace ash
