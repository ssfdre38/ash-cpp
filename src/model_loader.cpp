#include "model_loader.h"
#include "logger.h"
#include <fstream>
#include <filesystem>
#include <unordered_map>
#include <cstring>
#include <algorithm>

namespace fs = std::filesystem;

namespace ash {

// GGUF magic number
static const uint32_t GGUF_MAGIC = 0x46554747; // "GGUF"
static const uint32_t GGUF_VERSION = 3;

// SafeTensors magic
static const char* SAFETENSORS_MAGIC = "{\"__metadata__\"";

// =========================================================================
// ModelLoader Implementation
// =========================================================================

struct ModelLoader::Impl {
    ProgressCallback progress_callback;
};

ModelLoader::ModelLoader() : impl_(std::make_unique<Impl>()) {}
ModelLoader::~ModelLoader() = default;

std::unique_ptr<IModel> ModelLoader::load_local(
    const std::string& file_path,
    ModelFormat format
) {
    Logger::instance().info("Loading model from: " + file_path);
    
    // Check file exists
    if (!fs::exists(file_path)) {
        Logger::instance().error("Model file not found: " + file_path);
        return nullptr;
    }
    
    // Auto-detect format if needed
    if (format == ModelFormat::AUTO_DETECT) {
        format = detect_format(file_path);
        std::string format_name = (format == ModelFormat::GGUF ? "GGUF" : "Unknown");
        Logger::instance().info("Detected format: " + format_name);
    }
    
    // Create appropriate model instance
    std::unique_ptr<IModel> model;
    switch (format) {
        case ModelFormat::GGUF: {
            auto gguf = std::make_unique<GGUFModel>();
            if (gguf->load(file_path)) {
                model = std::move(gguf);
            }
            break;
        }
        case ModelFormat::SAFETENSORS: {
            auto st = std::make_unique<SafeTensorsModel>();
            if (st->load(file_path)) {
                model = std::move(st);
            }
            break;
        }
        default:
            Logger::instance().error("Unsupported model format");
            return nullptr;
    }
    
    if (model) {
        auto info = model->get_info();
        Logger::instance().info("✅ Model loaded: " + info.name + 
            " (" + std::to_string(info.file_size_bytes / (1024*1024)) + " MB)");
    }
    
    return model;
}

std::unique_ptr<IModel> ModelLoader::load_from_url(
    const std::string& url,
    const std::string& cache_dir,
    ModelFormat format
) {
    Logger::instance().info("Downloading model from: " + url);
    
    // Create cache directory
    fs::create_directories(cache_dir);
    
    // Extract filename from URL
    std::string filename = url.substr(url.find_last_of('/') + 1);
    std::string cache_path = cache_dir + "/" + filename;
    
    // Check if already cached
    if (fs::exists(cache_path)) {
        Logger::instance().info("Using cached model: " + cache_path);
        return load_local(cache_path, format);
    }
    
    // Download file
    bool success = ModelDownloader::download_file(url, cache_path, impl_->progress_callback);
    if (!success) {
        Logger::instance().error("Failed to download model");
        return nullptr;
    }
    
    // Load the downloaded model
    return load_local(cache_path, format);
}

std::unique_ptr<IModel> ModelLoader::load_from_huggingface(
    const std::string& repo_id,
    const std::string& filename,
    const std::string& cache_dir
) {
    Logger::instance().info("Loading from Hugging Face: " + repo_id);
    
    // TODO: Implement HF Hub API integration
    // For now, just log and return nullptr
    Logger::instance().warning("Hugging Face integration not yet implemented");
    Logger::instance().info("Manual download: https://huggingface.co/" + repo_id);
    
    return nullptr;
}

ModelFormat ModelLoader::detect_format(const std::string& file_path) {
    std::ifstream file(file_path, std::ios::binary);
    if (!file) {
        return ModelFormat::AUTO_DETECT;
    }
    
    // Read first 4 bytes for magic number
    uint32_t magic = 0;
    file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    
    if (magic == GGUF_MAGIC) {
        return ModelFormat::GGUF;
    }
    
    // Check for SafeTensors JSON header
    file.seekg(0);
    char buffer[32];
    file.read(buffer, sizeof(buffer));
    if (std::strncmp(buffer, SAFETENSORS_MAGIC, std::strlen(SAFETENSORS_MAGIC)) == 0) {
        return ModelFormat::SAFETENSORS;
    }
    
    // Check file extension as fallback
    std::string ext = fs::path(file_path).extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    
    if (ext == ".gguf") return ModelFormat::GGUF;
    if (ext == ".safetensors") return ModelFormat::SAFETENSORS;
    if (ext == ".bin") return ModelFormat::PYTORCH_BIN;
    
    return ModelFormat::AUTO_DETECT;
}

QuantizationType ModelLoader::detect_quantization(const std::string& filename) {
    // Convert to lowercase for matching
    std::string lower = filename;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    
    // Check for quantization type in filename
    if (lower.find("q2_k") != std::string::npos) return QuantizationType::Q2_K;
    if (lower.find("q3_k_s") != std::string::npos) return QuantizationType::Q3_K_S;
    if (lower.find("q3_k_m") != std::string::npos) return QuantizationType::Q3_K_M;
    if (lower.find("q4_0") != std::string::npos) return QuantizationType::Q4_0;
    if (lower.find("q4_k_s") != std::string::npos) return QuantizationType::Q4_K_S;
    if (lower.find("q4_k_m") != std::string::npos) return QuantizationType::Q4_K_M;
    if (lower.find("q5_k_s") != std::string::npos) return QuantizationType::Q5_K_S;
    if (lower.find("q5_k_m") != std::string::npos) return QuantizationType::Q5_K_M;
    if (lower.find("q6_k") != std::string::npos) return QuantizationType::Q6_K;
    if (lower.find("q8_0") != std::string::npos) return QuantizationType::Q8_0;
    if (lower.find("f16") != std::string::npos) return QuantizationType::F16;
    if (lower.find("f32") != std::string::npos) return QuantizationType::F32;
    
    return QuantizationType::UNKNOWN;
}

void ModelLoader::set_progress_callback(ProgressCallback callback) {
    impl_->progress_callback = callback;
}

// =========================================================================
// GGUFModel Implementation
// =========================================================================

struct GGUFModel::Impl {
    bool loaded = false;
    ModelInfo info;
    void* mmap_ptr = nullptr;
    size_t mmap_size = 0;
};

GGUFModel::GGUFModel() : impl_(std::make_unique<Impl>()) {}
GGUFModel::~GGUFModel() {
    unload();
}

bool GGUFModel::load(const std::string& source_path) {
    Logger::instance().info("Loading GGUF model: " + source_path);
    
    std::ifstream file(source_path, std::ios::binary);
    if (!file) {
        Logger::instance().error("Failed to open GGUF file");
        return false;
    }
    
    // Read GGUF header
    uint32_t magic, version;
    file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    file.read(reinterpret_cast<char*>(&version), sizeof(version));
    
    if (magic != GGUF_MAGIC) {
        Logger::instance().error("Invalid GGUF magic number");
        return false;
    }
    
    Logger::instance().info("GGUF version: " + std::to_string(version));
    
    // Get file size
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0);
    
    // Fill in model info
    impl_->info.name = fs::path(source_path).stem().string();
    impl_->info.version = std::to_string(version);
    impl_->info.architecture = "gemma"; // TODO: Parse from GGUF metadata
    impl_->info.format = ModelFormat::GGUF;
    impl_->info.quant_type = ModelLoader::detect_quantization(source_path);
    impl_->info.file_size_bytes = file_size;
    impl_->info.path = source_path;
    
    // Default Gemma 4 config (TODO: parse from GGUF)
    impl_->info.context_length = 8192;
    impl_->info.vocab_size = 256000;
    impl_->info.embedding_dim = 3072;
    impl_->info.num_layers = 42;
    impl_->info.num_heads = 16;
    impl_->info.parameter_count = 9000000000; // 9B
    
    impl_->loaded = true;
    
    Logger::instance().info("GGUF model loaded successfully");
    Logger::instance().debug("  Size: " + std::to_string(file_size / (1024*1024)) + " MB");
    Logger::instance().debug("  Quant: " + std::to_string(static_cast<int>(impl_->info.quant_type)));
    
    return true;
}

bool GGUFModel::load_mmap(const std::string& file_path) {
    // TODO: Implement memory-mapped loading for large models
    // This is more efficient than reading entire file into RAM
    Logger::instance().warning("Memory-mapped loading not yet implemented");
    return load(file_path);
}

void GGUFModel::unload() {
    if (impl_->loaded) {
        Logger::instance().info("Unloading GGUF model: " + impl_->info.name);
        // TODO: Cleanup mmap if used
        impl_->loaded = false;
    }
}

bool GGUFModel::is_loaded() const {
    return impl_->loaded;
}

ModelInfo GGUFModel::get_info() const {
    return impl_->info;
}

size_t GGUFModel::get_memory_usage() const {
    // TODO: Track actual memory usage
    return impl_->loaded ? impl_->info.file_size_bytes : 0;
}

// =========================================================================
// SafeTensorsModel Implementation (stub)
// =========================================================================

struct SafeTensorsModel::Impl {
    bool loaded = false;
    ModelInfo info;
};

SafeTensorsModel::SafeTensorsModel() : impl_(std::make_unique<Impl>()) {}
SafeTensorsModel::~SafeTensorsModel() {
    unload();
}

bool SafeTensorsModel::load(const std::string& source_path) {
    Logger::instance().warning("SafeTensors format not yet fully implemented");
    return false;
}

void SafeTensorsModel::unload() {
    impl_->loaded = false;
}

bool SafeTensorsModel::is_loaded() const {
    return impl_->loaded;
}

ModelInfo SafeTensorsModel::get_info() const {
    return impl_->info;
}

size_t SafeTensorsModel::get_memory_usage() const {
    return 0;
}

// =========================================================================
// ModelRegistry Implementation
// =========================================================================

struct ModelRegistry::Impl {
    std::unordered_map<std::string, std::unique_ptr<IModel>> models;
    std::mutex mutex;
};

ModelRegistry& ModelRegistry::instance() {
    static ModelRegistry instance;
    return instance;
}

ModelRegistry::ModelRegistry() : impl_(std::make_unique<Impl>()) {}
ModelRegistry::~ModelRegistry() {
    unload_all();
}

void ModelRegistry::register_model(const std::string& name, std::unique_ptr<IModel> model) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->models[name] = std::move(model);
    Logger::instance().info("Model registered: " + name);
}

IModel* ModelRegistry::get_model(const std::string& name) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    auto it = impl_->models.find(name);
    return it != impl_->models.end() ? it->second.get() : nullptr;
}

void ModelRegistry::unload_model(const std::string& name) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->models.erase(name);
    Logger::instance().info("Model unloaded: " + name);
}

void ModelRegistry::unload_all() {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    Logger::instance().info("Unloading all models (" + 
        std::to_string(impl_->models.size()) + ")");
    impl_->models.clear();
}

std::vector<std::string> ModelRegistry::list_models() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    std::vector<std::string> names;
    for (const auto& [name, _] : impl_->models) {
        names.push_back(name);
    }
    return names;
}

size_t ModelRegistry::get_total_memory_usage() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    size_t total = 0;
    for (const auto& [_, model] : impl_->models) {
        total += model->get_memory_usage();
    }
    return total;
}

// =========================================================================
// ModelDownloader Implementation (stub)
// =========================================================================

bool ModelDownloader::download_file(
    const std::string& url,
    const std::string& output_path,
    ModelLoader::ProgressCallback progress
) {
    Logger::instance().warning("HTTP download not yet implemented");
    Logger::instance().info("Please download manually from: " + url);
    Logger::instance().info("Save to: " + output_path);
    return false;
}

bool ModelDownloader::download_from_huggingface(
    const std::string& repo_id,
    const std::string& filename,
    const std::string& output_path,
    const std::string& token,
    ModelLoader::ProgressCallback progress
) {
    Logger::instance().warning("Hugging Face API not yet implemented");
    std::string hf_url = "https://huggingface.co/" + repo_id;
    if (!filename.empty()) {
        hf_url += "/blob/main/" + filename;
    }
    Logger::instance().info("Please download manually from: " + hf_url);
    return false;
}

bool ModelDownloader::is_cached(const std::string& cache_dir, const std::string& filename) {
    return fs::exists(cache_dir + "/" + filename);
}

std::string ModelDownloader::get_cache_path(const std::string& cache_dir, const std::string& filename) {
    return cache_dir + "/" + filename;
}

} // namespace ash
