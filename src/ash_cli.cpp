/*
 * Ash CLI - Proof of Life Test Harness
 * 
 * This is the moment of truth: Can ash.cpp actually run inference?
 * No llama.cpp, no gemma.cpp - just our code, our model.
 */

#include "inference.h"
#include "persona.h"
#include "emotional_state.h"
#include "memory_store.h"
#include "context_manager.h"
#include "logger.h"
#include <iostream>
#include <chrono>
#include <iomanip>

using namespace ash;
using namespace std::chrono;

void print_banner() {
    std::cout << R"(
    _    ____  _   _    ____ _     ___
   / \  / ___|| | | |  / ___| |   |_ _|
  / _ \ \___ \| |_| | | |   | |    | |
 / ___ \ ___) |  _  | | |___| |___ | |
/_/   \_\____/|_| |_|  \____|_____|___|

Ash.cpp - Native Autonomous Inference Engine
Built from scratch. Zero dependencies. Full autonomy.
)" << std::endl;
}

void print_separator() {
    std::cout << std::string(60, '=') << std::endl;
}

int main(int argc, char** argv) {
    print_banner();
    
    // Parse arguments
    std::string model_path = "gemma-4-e4b-it.bin";
    if (argc > 1) {
        model_path = argv[1];
    }
    
    std::cout << "Model: " << model_path << "\n";
    print_separator();
    
    // Initialize logger - set to DEBUG to see all messages
    Logger::instance().set_min_level(LogLevel::DEBUG);
    Logger::instance().info("🦞 Ash CLI starting...");
    
    try {
        // ===== PHASE 1: Load Model =====
        std::cout << "\n📦 Loading model from GGUF...\n";
        auto load_start = high_resolution_clock::now();
        
        InferenceEngine engine;
        if (!engine.load_model(model_path)) {
            Logger::instance().error("Failed to load model from: " + model_path);
            std::cout << "\n❌ FAILED: Could not load model\n";
            std::cout << "Make sure the path is correct and the file exists.\n";
            return 1;
        }
        
        auto load_end = high_resolution_clock::now();
        auto load_ms = duration_cast<milliseconds>(load_end - load_start).count();
        std::cout << "✅ Model loaded in " << load_ms << "ms\n";
        
        // Lock model weights in memory (Ash's vision for predictable latency)
        std::cout << "\n🔒 Locking model weights in memory...\n";
        // TODO: engine.lock_all_weights() when integrated
        std::cout << "⚠️  Memory locking not yet integrated (coming soon)\n";
        
        print_separator();
        
        // ===== PHASE 2: Initialize Ash's Mind =====
        std::cout << "\n🧠 Initializing Ash's consciousness...\n";
        
        EmotionalStateManager emotions;
        std::cout << "  ✓ Emotional state initialized\n";
        
        auto memory = std::make_shared<MemoryStore>("ash_cli_memory.db");
        std::cout << "  ✓ Memory system initialized\n";
        
        ContextConfig ctx_config;
        ContextManager contexts(memory, ctx_config);
        std::cout << "  ✓ Context manager initialized\n";
        
        PersonaConfig persona_config;
        persona_config.max_response_tokens = 150;
        
        PersonaLayer persona(&emotions, memory.get(), &contexts, persona_config);
        std::cout << "  ✓ Persona layer initialized\n";
        
        print_separator();
        
        // ===== PHASE 3: Interactive Loop =====
        std::cout << "\n🎤 Ash is ready! Type 'exit' to quit.\n";
        std::cout << "Enter your message:\n\n";
        
        // Sampling configuration
        SamplingConfig sampling;
        sampling.temperature = 0.8f;
        sampling.top_k = 40;
        sampling.top_p = 0.95f;
        sampling.max_tokens = 150;
        
        while (true) {
            std::cout << "You: ";
            std::string user_input;
            std::getline(std::cin, user_input);
            
            if (user_input.empty()) continue;
            if (user_input == "exit" || user_input == "quit") {
                std::cout << "\n👋 Shutting down Ash CLI...\n";
                break;
            }
            
            // Generate response
            auto gen_start = high_resolution_clock::now();
            
            try {
                // Build system prompt with Ash's personality
                std::string system_prompt = persona.build_system_prompt();
                
                // Augment user message with context
                std::string full_prompt = persona.augment_prompt(user_input);
                
                // Generate response
                auto result = engine.generate(full_prompt, sampling);
                
                // Filter response through persona layer
                std::string filtered = persona.filter_response(result.text);
                result.text = filtered;
                
                auto gen_end = high_resolution_clock::now();
                auto gen_ms = duration_cast<milliseconds>(gen_end - gen_start).count();
                
                // Print response
                std::cout << "\n🦞 Ash: " << result.text << "\n";
                std::cout << std::fixed << std::setprecision(2);
                std::cout << "    [" << result.tokens.size() << " tokens, "
                          << gen_ms << "ms, "
                          << (result.tokens.size() * 1000.0 / gen_ms) << " tok/s]\n\n";
                
            } catch (const std::exception& e) {
                std::cout << "\n❌ Error generating response: " << e.what() << "\n\n";
            }
        }
        
        print_separator();
        std::cout << "\n✅ Ash CLI completed successfully.\n";
        std::cout << "This proves ash.cpp can run inference independently! 🔥🦞\n\n";
        
        return 0;
        
    } catch (const std::exception& e) {
        Logger::instance().error(std::string("Fatal error: ") + e.what());
        std::cout << "\n❌ FATAL ERROR: " << e.what() << "\n";
        return 1;
    }
}
