#include "tokenizer.h"
#include "logger.h"
#include <iostream>

using namespace ash;

int main() {
    std::cout << "🔤 Testing Tokenizer...\n\n";
    
    Logger::instance().set_min_level(LogLevel::INFO);
    
    // Create test tokenizer
    std::cout << "Creating test tokenizer with minimal vocab...\n";
    auto tokenizer = TokenizerFactory::create_test_tokenizer();
    
    if (!tokenizer) {
        std::cout << "❌ Failed to create tokenizer\n";
        return 1;
    }
    
    std::cout << "✅ Tokenizer created\n";
    std::cout << "  Vocab size: " << tokenizer->vocab_size() << "\n";
    
    auto special = tokenizer->special_tokens();
    std::cout << "  BOS token: " << special.bos_token << "\n";
    std::cout << "  EOS token: " << special.eos_token << "\n";
    std::cout << "  PAD token: " << special.pad_token << "\n";
    std::cout << "  UNK token: " << special.unk_token << "\n\n";
    
    // Test encoding
    std::cout << "Test 1: Encoding text\n";
    std::string text = "hello world test";
    auto tokens = tokenizer->encode(text, true, true);
    
    std::cout << "  Input: \"" << text << "\"\n";
    std::cout << "  Tokens: [";
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << tokens[i];
    }
    std::cout << "]\n";
    std::cout << "  Token count: " << tokens.size() << "\n\n";
    
    // Test decoding
    std::cout << "Test 2: Decoding tokens\n";
    std::string decoded = tokenizer->decode(tokens, true);
    std::cout << "  Decoded: \"" << decoded << "\"\n\n";
    
    // Test individual tokens
    std::cout << "Test 3: Individual token decode\n";
    for (size_t i = 0; i < std::min<size_t>(15, tokenizer->vocab_size()); ++i) {
        std::string token_str = tokenizer->decode_token(static_cast<TokenID>(i));
        std::cout << "  Token " << i << ": \"" << token_str << "\"\n";
    }
    std::cout << "\n";
    
    // Test special tokens
    std::cout << "Test 4: Special token handling\n";
    std::vector<TokenID> test_tokens = {
        special.bos_token,
        8, // "hello"
        special.eos_token
    };
    
    std::string with_special = tokenizer->decode(test_tokens, false);
    std::string without_special = tokenizer->decode(test_tokens, true);
    
    std::cout << "  With special: \"" << with_special << "\"\n";
    std::cout << "  Without special: \"" << without_special << "\"\n\n";
    
    // Test multiple encodings
    std::cout << "Test 5: Multiple text encodings\n";
    std::vector<std::string> test_texts = {
        "test",
        "hello",
        "world"
    };
    
    for (const auto& t : test_texts) {
        auto enc = tokenizer->encode(t, false, false);
        auto dec = tokenizer->decode(enc, true);
        std::cout << "  \"" << t << "\" -> [";
        for (size_t i = 0; i < enc.size(); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << enc[i];
        }
        std::cout << "] -> \"" << dec << "\"\n";
    }
    std::cout << "\n";
    
    std::cout << "✓ Tokenizer test complete!\n";
    std::cout << "🔥 Ready to encode/decode text for inference.\n";
    std::cout << "\nNote: This is a simplified tokenizer for testing.\n";
    std::cout << "Full GGUF tokenizer will load vocab from model file.\n";
    
    return 0;
}
