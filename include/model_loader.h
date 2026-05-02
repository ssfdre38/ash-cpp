#pragma once

#include <string>
#include <memory>
#include <vector>
#include <cstdint>
#include <functional>

namespace ash {

// Supported model formats
enum class ModelFormat {
    GGUF,           // llama.cpp format (quantized)
    SAFETENSORS,    // Hugging Face PyTorch format
    PYTORCH_BIN,    // Legacy PyTorch .bin format
    AUTO_DETECT     // Try to detect from file
};

// Model source types
enum class ModelSource {
    LOCAL_FILE,     // Load from disk
    HTTP_DOWNLOAD,  // Download from URL
    HUGGINGFACE,    // Download from HF hub
    BLOB_STORAGE    // Custom blob storage (future)
};

// Quantization levels (for GGUF)
enum class QuantizationType {
    Q2_K,      // 2-bit (smallest, lowest quality)
    Q3_K_S,    // 3-bit small
    Q3_K_M,    // 3-bit medium
    Q4_0,      // 4-bit (legacy)
    Q4_K_S,    // 4-bit small
    Q4_K_M,    // 4-bit medium (RECOMMENDED)
    Q5_K_S,    // 5-bit small
    Q5_K_M,    // 5-bit medium
    Q6_K,      // 6-bit
    Q8_0,      // 8-bit
    F16,       // 16-bit float
    F32,       // 32-bit float (full precision)
    UNKNOWN
};

// Model type for ash-forge routing
enum class ModelType {
    ROUTER,      // Scout agent (ash-core) - routes queries
    SPECIALIST,  // Domain specialist (ash-python, ash-javascript, etc.)
    PERSONALITY, // Full personality (ash-chat) - autonomous, emotional
    UNKNOWN
};

// Model metadata
struct ModelInfo {
    std::string name;
    std::string version;
    std::string architecture;  // "gemma", "llama", etc.
    ModelFormat format;
    QuantizationType quant_type;
    size_t parameter_count;    // e.g., 9B for Gemma 4
    size_t file_size_bytes;
    std::string path;          // Local file path once loaded
    
    // Model-specific config
    uint32_t context_length;   // Max context window
    uint32_t vocab_size;
    uint32_t embedding_dim;
    uint32_t num_layers;
    uint32_t num_heads;
    
    // Ash-Forge multi-model extensions
    ModelType type = ModelType::UNKNOWN;
    std::vector<std::string> categories;  // ["python", "pip", "django"] for specialists
    bool always_loaded = false;           // true for ash-core router
    int priority = 50;                    // Higher = keep loaded longer (0-100)
    
    // Usage tracking (for LRU eviction)
    mutable time_t last_used = 0;         // Timestamp of last use
    mutable size_t query_count = 0;       // Number of queries handled
};

// Abstract model interface
class IModel {
public:
    virtual ~IModel() = default;
    
    // Load model from source
    virtual bool load(const std::string& source_path) = 0;
    
    // Unload model from memory
    virtual void unload() = 0;
    
    // Check if model is loaded
    virtual bool is_loaded() const = 0;
    
    // Get model metadata
    virtual ModelInfo get_info() const = 0;
    
    // Get model memory usage
    virtual size_t get_memory_usage() const = 0;
};

// Model loader - handles different formats and sources
class ModelLoader {
public:
    ModelLoader();
    ~ModelLoader();
    
    // Load model from local file
    std::unique_ptr<IModel> load_local(
        const std::string& file_path,
        ModelFormat format = ModelFormat::AUTO_DETECT
    );
    
    // Download and load model from URL
    std::unique_ptr<IModel> load_from_url(
        const std::string& url,
        const std::string& cache_dir = "./models/cache",
        ModelFormat format = ModelFormat::AUTO_DETECT
    );
    
    // Load model from Hugging Face Hub
    std::unique_ptr<IModel> load_from_huggingface(
        const std::string& repo_id,          // e.g., "google/gemma-4-turbo"
        const std::string& filename = "",    // Specific file, or auto-select
        const std::string& cache_dir = "./models/cache"
    );
    
    // Auto-detect model format from file
    static ModelFormat detect_format(const std::string& file_path);
    
    // Parse quantization type from filename
    static QuantizationType detect_quantization(const std::string& filename);
    
    // Get recommended model for Ash
    static std::string get_recommended_model() {
        return "lmstudio-community/gemma-4-turbo-Q4_K_M-GGUF";
    }
    
    // Progress callback for downloads
    using ProgressCallback = std::function<void(size_t downloaded, size_t total)>;
    void set_progress_callback(ProgressCallback callback);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// GGUF model implementation
class GGUFModel : public IModel {
public:
    GGUFModel();
    ~GGUFModel() override;
    
    bool load(const std::string& source_path) override;
    void unload() override;
    bool is_loaded() const override;
    ModelInfo get_info() const override;
    size_t get_memory_usage() const override;
    
    // GGUF-specific: memory-mapped loading for efficiency
    bool load_mmap(const std::string& file_path);
    
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// SafeTensors model implementation (future)
class SafeTensorsModel : public IModel {
public:
    SafeTensorsModel();
    ~SafeTensorsModel() override;
    
    bool load(const std::string& source_path) override;
    void unload() override;
    bool is_loaded() const override;
    ModelInfo get_info() const override;
    size_t get_memory_usage() const override;
    
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Model registry - tracks loaded models (enhanced for ash-forge multi-model)
class ModelRegistry {
public:
    static ModelRegistry& instance();
    
    // ===== Original API (preserved) =====
    // Register a loaded model
    void register_model(const std::string& name, std::unique_ptr<IModel> model);
    
    // Get a loaded model by name
    IModel* get_model(const std::string& name);
    
    // Unload a specific model
    void unload_model(const std::string& name);
    
    // Unload all models
    void unload_all();
    
    // List all loaded models
    std::vector<std::string> list_models() const;
    
    // Get total memory usage
    size_t get_total_memory_usage() const;
    
    // ===== Ash-Forge Multi-Model Extensions =====
    
    // Discover available models in directory
    std::vector<ModelInfo> discover_models(const std::string& models_dir);
    
    // Load model from file path (auto-creates IModel instance)
    bool load_model(const std::string& name, const std::string& file_path);
    
    // Check if model is currently loaded
    bool is_loaded(const std::string& name) const;
    
    // Get model info without loading (from discovery)
    ModelInfo get_model_info(const std::string& name) const;
    
    // Memory management
    size_t get_memory_budget() const { return memory_budget_; }
    void set_memory_budget(size_t bytes) { memory_budget_ = bytes; }
    bool fits_in_budget(size_t additional_bytes) const;
    
    // LRU tracking
    void mark_model_used(const std::string& name);
    std::string get_lru_model() const;  // Get least recently used (for eviction)
    
    // Automatic memory management
    bool ensure_space(size_t required_bytes);  // Evict models if needed
    void cleanup_cold_models(int max_age_seconds = 300);  // Unload old models
    
private:
    ModelRegistry();
    ~ModelRegistry();
    
    struct Impl;
    std::unique_ptr<Impl> impl_;
    
    size_t memory_budget_ = 12ull * 1024 * 1024 * 1024;  // 12GB default
};

// Model download utilities
class ModelDownloader {
public:
    // Download file from URL with progress
    static bool download_file(
        const std::string& url,
        const std::string& output_path,
        ModelLoader::ProgressCallback progress = nullptr
    );
    
    // Download from Hugging Face Hub
    static bool download_from_huggingface(
        const std::string& repo_id,
        const std::string& filename,
        const std::string& output_path,
        const std::string& token = "",  // Optional HF token
        ModelLoader::ProgressCallback progress = nullptr
    );
    
    // Check if file exists in cache
    static bool is_cached(const std::string& cache_dir, const std::string& filename);
    
    // Get cache path for a file
    static std::string get_cache_path(const std::string& cache_dir, const std::string& filename);
};

// Blob storage interface (for future custom storage)
class IBlobStorage {
public:
    virtual ~IBlobStorage() = default;
    
    // Check if blob exists
    virtual bool exists(const std::string& blob_id) = 0;
    
    // Get blob metadata
    virtual size_t get_size(const std::string& blob_id) = 0;
    
    // Download blob to local file
    virtual bool download(const std::string& blob_id, const std::string& output_path) = 0;
    
    // Upload local file as blob
    virtual bool upload(const std::string& file_path, const std::string& blob_id) = 0;
};

} // namespace ash
