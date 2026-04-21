#include "model_loader.h"
#include "logger.h"
#include <iostream>

using namespace ash;

int main(int argc, char* argv[]) {
    Logger::instance().set_log_file("model_test.log");
    Logger::instance().set_min_level(LogLevel::DEBUG);
    
    Logger::instance().info("=== Ash Model Loader Test ===");
    
    // Check command line args
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <path_to_model.gguf>" << std::endl;
        std::cout << std::endl;
        std::cout << "Recommended model: " << ModelLoader::get_recommended_model() << std::endl;
        std::cout << std::endl;
        std::cout << "Download from: https://huggingface.co/lmstudio-community/gemma-4-turbo-GGUF" << std::endl;
        std::cout << "Choose: gemma-4-turbo-Q4_K_M.gguf (~3GB)" << std::endl;
        return 1;
    }
    
    std::string model_path = argv[1];
    Logger::instance().info("Model path: " + model_path);
    
    // Create model loader
    ModelLoader loader;
    
    // Set progress callback
    loader.set_progress_callback([](size_t downloaded, size_t total) {
        if (total > 0) {
            int percent = (downloaded * 100) / total;
            Logger::instance().info("Download progress: " + 
                std::to_string(percent) + "%");
        }
    });
    
    // Load the model
    Logger::instance().info("Loading model...");
    auto model = loader.load_local(model_path);
    
    if (!model) {
        Logger::instance().error("❌ Failed to load model");
        return 1;
    }
    
    // Get model info
    auto info = model->get_info();
    
    std::cout << "\n🔥 Model loaded successfully!\n" << std::endl;
    std::cout << "Model Information:" << std::endl;
    std::cout << "  Name: " << info.name << std::endl;
    std::cout << "  Architecture: " << info.architecture << std::endl;
    std::cout << "  Format: " << (info.format == ModelFormat::GGUF ? "GGUF" : "Unknown") << std::endl;
    std::cout << "  Parameters: " << (info.parameter_count / 1000000000) << "B" << std::endl;
    std::cout << "  File size: " << (info.file_size_bytes / (1024*1024)) << " MB" << std::endl;
    std::cout << "  Context length: " << info.context_length << std::endl;
    std::cout << "  Vocab size: " << info.vocab_size << std::endl;
    std::cout << "  Layers: " << info.num_layers << std::endl;
    std::cout << "  Heads: " << info.num_heads << std::endl;
    std::cout << "  Memory usage: " << (model->get_memory_usage() / (1024*1024)) << " MB" << std::endl;
    
    // Get quant type name
    std::string quant_name;
    switch (info.quant_type) {
        case QuantizationType::Q2_K: quant_name = "Q2_K (2-bit)"; break;
        case QuantizationType::Q4_K_M: quant_name = "Q4_K_M (4-bit)"; break;
        case QuantizationType::Q5_K_M: quant_name = "Q5_K_M (5-bit)"; break;
        case QuantizationType::Q8_0: quant_name = "Q8_0 (8-bit)"; break;
        case QuantizationType::F16: quant_name = "F16 (16-bit)"; break;
        default: quant_name = "Unknown"; break;
    }
    std::cout << "  Quantization: " << quant_name << std::endl;
    
    // Register in model registry
    ModelRegistry::instance().register_model("gemma-4-turbo", std::move(model));
    
    // List registered models
    auto models = ModelRegistry::instance().list_models();
    std::cout << "\nRegistered models:" << std::endl;
    for (const auto& name : models) {
        std::cout << "  - " << name << std::endl;
    }
    
    std::cout << "\nTotal memory: " << 
        (ModelRegistry::instance().get_total_memory_usage() / (1024*1024)) << " MB" << std::endl;
    
    std::cout << "\n🦞 Model ready for inference!" << std::endl;
    std::cout << "Note: Actual inference engine not yet implemented." << std::endl;
    std::cout << "Next steps: Integrate llama.cpp or gemma.cpp backend.\n" << std::endl;
    
    // Cleanup
    ModelRegistry::instance().unload_all();
    
    return 0;
}
