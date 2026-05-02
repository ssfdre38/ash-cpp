// Minimal generation test
#include "inference.h"
#include "logger.h"
#include <iostream>

int main(int argc, char** argv) {
    using namespace ash;
    
    if (argc < 2) {
        std::cerr << "Usage: test_gen <gguf_path>\n";
        return 1;
    }
    
    std::string model_path = argv[1];
    
    Logger::instance().set_min_level(LogLevel::INFO);  // Reduce spam
    
    std::cout << "Loading model...\n";
    InferenceEngine engine;
    if (!engine.load_model(model_path)) {
        std::cerr << "Failed to load model\n";
        return 1;
    }
    std::cout << "✅ Model loaded!\n\n";
    
    // Test generation with greedy sampling
    // Use Qwen2.5-Instruct chat template for sensible output
    std::string prompt = "<|im_start|>system\nYou are ash-code:python, a specialized AI assistant focused on Python programming. You excel at writing clean, idiomatic Python code.<|im_end|>\n<|im_start|>user\nWrite a hello world in Python<|im_end|>\n<|im_start|>assistant\n";
    std::cout << "Prompt: \"" << prompt << "\"\n\n";
    
    std::cout << "Generating...\n" << std::flush;
    
    SamplingConfig config;
    config.max_tokens = 200;  // Give enough room to complete the answer
    config.temperature = 0.0f;  // Greedy sampling
    config.use_sampling = false;
    
    std::cout << "Calling engine.generate()...\n" << std::flush;
    auto result = engine.generate(prompt, config);
    
    std::cout << "\nGenerated text:\n\"" << result.text << "\"\n";
    std::cout << "\nTokens: " << result.tokens_generated << "\n";
    std::cout << "Stop reason: " << result.stop_reason << "\n";
    
    return 0;
}
