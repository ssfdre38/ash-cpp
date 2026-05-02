#include "tokenizer.h"
#include "gguf_parser.h"
#include "logger.h"
#include <algorithm>
#include <sstream>
#include <cctype>
#include <climits>

namespace ash {

struct SimpleSPTokenizer::Impl {
    std::unordered_map<TokenID, std::string> id_to_token;
    std::unordered_map<std::string, TokenID> token_to_id;
    std::unordered_map<TokenID, float> token_scores;
    SpecialTokens special;
    
    // BPE merge rules: pair -> rank (priority)
    std::unordered_map<std::string, int> bpe_merges;
    bool has_bpe = false;
    
    // Control/special tokens sorted longest-first for pre-scan before BPE
    // e.g. <|im_start|>, <|im_end|>, <|endoftext|>
    std::vector<std::pair<std::string, TokenID>> control_tokens;
    bool control_tokens_built = false;
    
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
    
    // Load BPE merges if available
    GGUFMetadataValue merges_array;
    if (parser.get_metadata("tokenizer.ggml.merges", merges_array)) {
        Logger::instance().info("Loading BPE merges: " + std::to_string(merges_array.array_value.size()) + " rules");
        
        for (size_t i = 0; i < merges_array.array_value.size(); ++i) {
            std::string merge_rule = merges_array.array_value[i].string_value;
            // Store with rank (order matters in BPE)
            impl_->bpe_merges[merge_rule] = static_cast<int>(i);
        }
        
        impl_->has_bpe = true;
        Logger::instance().info("✅ BPE enabled with " + std::to_string(impl_->bpe_merges.size()) + " merge rules");
    } else {
        Logger::instance().warning("No BPE merges found - using greedy tokenization");
        impl_->has_bpe = false;
    }
    
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

std::vector<TokenID> SimpleSPTokenizer::tokenize_bpe_segment(const std::string& text) {
    // BPE tokenize a plain text segment (no special tokens inside).
    //
    // Qwen2.5 uses GPT-2 byte-level BPE: bytes not directly in the vocab
    // are stored as unicode surrogates (space 0x20 → Ġ U+0120, etc.).
    // All BPE merge keys use the same encoding, so we must apply it in Step 1.

    // Convert a single ASCII byte to its vocab representation:
    // If the raw byte exists as a key in token_to_id, use it as-is.
    // Otherwise apply the GPT-2 byte encoding: U+0100 + byte (→ 2-byte UTF-8).
    auto byte_to_vocab_char = [&](unsigned char b) -> std::string {
        std::string raw(1, static_cast<char>(b));
        if (impl_->token_to_id.count(raw)) return raw;
        uint32_t cp = 0x0100u + b;
        std::string enc;
        if (cp <= 0x7FFu) {
            enc.push_back(static_cast<char>(0xC0u | (cp >> 6)));
            enc.push_back(static_cast<char>(0x80u | (cp & 0x3Fu)));
        } else {
            enc.push_back(static_cast<char>(0xE0u | (cp >> 12)));
            enc.push_back(static_cast<char>(0x80u | ((cp >> 6) & 0x3Fu)));
            enc.push_back(static_cast<char>(0x80u | (cp & 0x3Fu)));
        }
        return enc;
    };

    // Step 1: Split text into per-character vocab strings (GPT-2 encoded for ASCII).
    std::vector<std::string> tokens;
    for (size_t i = 0; i < text.length(); ) {
        unsigned char byte = static_cast<unsigned char>(text[i]);
        if ((byte & 0x80) == 0) {
            tokens.push_back(byte_to_vocab_char(byte));
            i += 1;
        } else if ((byte & 0xE0) == 0xC0) {
            tokens.push_back(text.substr(i, 2)); i += 2;
        } else if ((byte & 0xF0) == 0xE0) {
            tokens.push_back(text.substr(i, 3)); i += 3;
        } else if ((byte & 0xF8) == 0xF0) {
            tokens.push_back(text.substr(i, 4)); i += 4;
        } else {
            i++;
        }
    }

    // Step 2: Apply BPE merges iteratively (lowest rank = highest priority).
    // Merge keys in bpe_merges are stored as "left right" (space-separated).
    bool merged = true;
    while (merged && tokens.size() > 1) {
        merged = false;
        int best_rank = INT_MAX;
        size_t best_pos = SIZE_MAX;
        for (size_t i = 0; i + 1 < tokens.size(); i++) {
            std::string pair = tokens[i] + " " + tokens[i + 1];
            auto it = impl_->bpe_merges.find(pair);
            if (it != impl_->bpe_merges.end() && it->second < best_rank) {
                best_rank = it->second;
                best_pos = i;
            }
        }
        if (best_pos != SIZE_MAX) {
            tokens[best_pos] = tokens[best_pos] + tokens[best_pos + 1];
            tokens.erase(tokens.begin() + best_pos + 1);
            merged = true;
        }
    }

    // Step 3: Convert merged tokens to token IDs.
    std::vector<TokenID> result;
    for (const std::string& token : tokens) {
        auto it = impl_->token_to_id.find(token);
        if (it != impl_->token_to_id.end()) {
            result.push_back(it->second);
        } else {
            Logger::instance().warning("BPE: unknown token '" + token + "'");
            result.push_back(impl_->special.unk_token);
        }
    }
    return result;
}

std::vector<TokenID> SimpleSPTokenizer::tokenize_bpe(const std::string& text) {
    if (!impl_->has_bpe || impl_->bpe_merges.empty()) {
        Logger::instance().warning("No BPE merges available, using greedy tokenization");
        return tokenize_greedy(text);
    }
    
    // Build sorted control-token list once (longest first so we match greedily)
    if (!impl_->control_tokens_built) {
        for (auto& [tok_str, tok_id] : impl_->token_to_id) {
            // Control/special tokens look like <|...|> or similar
            if (tok_str.size() >= 4 && tok_str.front() == '<' && tok_str.back() == '>') {
                impl_->control_tokens.push_back({tok_str, tok_id});
            }
        }
        std::sort(impl_->control_tokens.begin(), impl_->control_tokens.end(),
                  [](const auto& a, const auto& b) { return a.first.size() > b.first.size(); });
        impl_->control_tokens_built = true;
    }
    
    // Pre-scan: split text around control tokens, BPE the segments between them
    std::vector<TokenID> result;
    size_t pos = 0;
    while (pos < text.size()) {
        // Try to match a control token at current position
        bool matched = false;
        for (auto& [tok_str, tok_id] : impl_->control_tokens) {
            if (text.compare(pos, tok_str.size(), tok_str) == 0) {
                result.push_back(tok_id);
                pos += tok_str.size();
                matched = true;
                break;
            }
        }
        if (!matched) {
            // Find end of next plain-text segment (stops at next control token)
            size_t seg_end = pos + 1;
            while (seg_end < text.size()) {
                bool at_control = false;
                for (auto& [tok_str, tok_id] : impl_->control_tokens) {
                    if (text.compare(seg_end, tok_str.size(), tok_str) == 0) {
                        at_control = true;
                        break;
                    }
                }
                if (at_control) break;
                seg_end++;
            }
            // BPE-tokenize the plain segment
            auto seg_tokens = tokenize_bpe_segment(text.substr(pos, seg_end - pos));
            result.insert(result.end(), seg_tokens.begin(), seg_tokens.end());
            pos = seg_end;
        }
    }
    return result;
}

std::vector<TokenID> SimpleSPTokenizer::tokenize_greedy(const std::string& text) {
    // Greedy longest-match tokenization
    // Skip empty tokens unless they're the only match
    
    std::vector<TokenID> result;
    size_t pos = 0;
    
    while (pos < text.length()) {
        // Try to match longest token
        size_t best_len = 0;
        TokenID best_token = impl_->special.unk_token;
        
        // Try lengths from longest to shortest (up to 50 chars for compound tokens)
        for (size_t len = std::min<size_t>(50, text.length() - pos); len > 0; --len) {
            std::string substr = text.substr(pos, len);
            auto it = impl_->token_to_id.find(substr);
            if (it != impl_->token_to_id.end()) {
                // Found a match! Use it if non-empty
                if (!substr.empty()) {
                    best_len = len;
                    best_token = it->second;
                    break;
                }
            }
        }
        
        if (best_len == 0) {
            // No match - try byte fallback tokens
            unsigned char byte = static_cast<unsigned char>(text[pos]);
            
            // Try <0xXX> format
            char hex[16];
            std::snprintf(hex, sizeof(hex), "<0x%02X>", byte);
            std::string byte_token = hex;
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
    // Fallback: encode each byte as UNK token (safer than offset mapping)
    std::vector<TokenID> result;
    for (unsigned char c : text) {
        // Use UNK token instead of out-of-bounds mapping
        result.push_back(impl_->special.unk_token);
    }
    Logger::instance().warning("Used fallback byte encoding (all UNK tokens) for: " + text.substr(0, std::min<size_t>(20, text.length())));
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
    
    // Use BPE if available, otherwise fall back to greedy
    std::vector<TokenID> text_tokens;
    if (impl_->has_bpe) {
        text_tokens = tokenize_bpe(text);
    } else {
        text_tokens = tokenize_greedy(text);
    }
    tokens.insert(tokens.end(), text_tokens.begin(), text_tokens.end());
    
    if (add_eos) {
        tokens.push_back(impl_->special.eos_token);
    }
    
    // Debug: log token IDs
    Logger::instance().debug("Encoded " + std::to_string(tokens.size()) + " tokens from text: " + text.substr(0, std::min<size_t>(50, text.length())));
    std::string token_ids_str = "Token IDs: [";
    for (size_t i = 0; i < std::min<size_t>(20, tokens.size()); ++i) {
        token_ids_str += std::to_string(tokens[i]);
        if (i < std::min<size_t>(20, tokens.size()) - 1) token_ids_str += ", ";
    }
    if (tokens.size() > 20) token_ids_str += ", ...";
    token_ids_str += "]";
    Logger::instance().debug(token_ids_str);
    
    return tokens;
}

std::string SimpleSPTokenizer::decode(const std::vector<TokenID>& tokens, bool skip_special) {
    std::string result;
    
    for (TokenID token : tokens) {
        if (skip_special && is_special(token)) {
            continue;
        }
        
        auto it = impl_->id_to_token.find(token);
        if (it != impl_->id_to_token.end()) {
            std::string token_str = it->second;
            
            // Skip control tokens (<|...|>) when skip_special is set
            if (skip_special && token_str.size() >= 4 &&
                token_str.front() == '<' && token_str.back() == '>') {
                continue;
            }
            
            // 1. Byte tokens stored as "<0xXX>"
            if (token_str.length() == 6 && token_str.substr(0, 3) == "<0x" && token_str.back() == '>') {
                try {
                    std::string hex = token_str.substr(3, 2);
                    unsigned char byte = static_cast<unsigned char>(std::stoul(hex, nullptr, 16));
                    result.push_back(static_cast<char>(byte));
                    continue;
                } catch (...) {}
            }
            
            // 2. Decode GPT-2 byte-level unicode back to raw bytes.
            // Characters in U+0100–U+0143 map back to bytes 0x00–0x43 (non-printable/space range).
            // Ġ (U+0120) → space (0x20), etc.
            std::string decoded_str;
            for (size_t i = 0; i < token_str.size(); ) {
                unsigned char b0 = static_cast<unsigned char>(token_str[i]);
                if ((b0 & 0xE0) == 0xC0 && i + 1 < token_str.size()) {
                    unsigned char b1 = static_cast<unsigned char>(token_str[i + 1]);
                    uint32_t cp = ((b0 & 0x1F) << 6) | (b1 & 0x3F);
                    if (cp >= 0x0100 && cp <= 0x0143) {
                        // GPT-2 byte encoding: find which raw byte this maps to
                        // Bytes not in {33-126, 161-172, 174-255} map to U+0100+n in order
                        int target_n = static_cast<int>(cp - 0x0100);
                        int n = 0;
                        for (int raw = 0; raw < 256; raw++) {
                            bool direct = (raw >= 33 && raw <= 126) ||
                                          (raw >= 161 && raw <= 172) ||
                                          (raw >= 174 && raw <= 255);
                            if (!direct) {
                                if (n == target_n) {
                                    decoded_str.push_back(static_cast<char>(raw));
                                    break;
                                }
                                n++;
                            }
                        }
                        i += 2;
                        continue;
                    }
                }
                decoded_str.push_back(static_cast<char>(b0));
                i++;
            }
            result.append(decoded_str);
        } else {
            result.append("<UNK>");
        }
    }
    
    return result;
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
