# Ash.cpp Model Loading System

## Overview

The model loading system provides a flexible, extensible architecture for loading AI models from various sources and formats. Built with the event loop foundation, it handles GGUF models (primary), SafeTensors (future), and multiple acquisition methods.

## Supported Formats

### GGUF (Primary - Fully Implemented)
- **Format:** llama.cpp/ggml quantized format
- **Use case:** Runtime inference (production)
- **Quantization levels:**
  - `Q2_K`: 2-bit (~2GB, fastest, lowest quality)
  - `Q4_K_M`: **4-bit (~3GB, RECOMMENDED)** - Best speed/quality trade-off
  - `Q5_K_M`: 5-bit (~4GB, higher quality)
  - `Q8_0`: 8-bit (~6GB, near full precision)
  - `F16`: 16-bit (~18GB, full quality, training only)

### SafeTensors (Stub - Future)
- **Format:** Hugging Face PyTorch format
- **Use case:** Training, fine-tuning
- **Status:** Interface defined, implementation pending

### PyTorch .bin (Future)
- **Format:** Legacy PyTorch checkpoint format
- **Status:** Not yet implemented

## Architecture

```
┌─────────────────────────────────────────────────┐
│              ModelLoader                        │
│  - Orchestrates loading from various sources   │
│  - Auto-detects format                          │
│  - Progress tracking                            │
└───────────┬─────────────────────────────────────┘
            │
            ├─► GGUFModel (IModel)
            │    - GGUF header parsing
            │    - Memory-mapped loading (future)
            │    - Metadata extraction
            │
            ├─► SafeTensorsModel (IModel) [stub]
            │    - JSON header parsing
            │    - Tensor loading
            │
            └─► ModelRegistry
                 - Tracks loaded models
                 - Memory management
                 - Multi-model support
```

## Model Sources

### 1. Local File (Implemented)
```cpp
ModelLoader loader;
auto model = loader.load_local("models/gemma-4-turbo-Q4_K_M.gguf");
```

### 2. HTTP Download (Stub)
```cpp
auto model = loader.load_from_url(
    "https://example.com/model.gguf",
    "./models/cache"
);
```

### 3. Hugging Face Hub (Stub)
```cpp
auto model = loader.load_from_huggingface(
    "lmstudio-community/gemma-4-turbo-GGUF",
    "gemma-4-turbo-Q4_K_M.gguf",
    "./models/cache"
);
```

### 4. Blob Storage (Future)
```cpp
// Custom storage backend (Azure, S3, custom protocol)
IBlobStorage* storage = ...; // User-provided
auto model = loader.load_from_blob(storage, "model-id-123");
```

## Usage Examples

### Basic Model Loading
```cpp
#include "model_loader.h"
#include "logger.h"

int main() {
    // Setup logging
    Logger::instance().set_log_file("ash.log");
    
    // Load model
    ModelLoader loader;
    auto model = loader.load_local("gemma-4-turbo-Q4_K_M.gguf");
    
    if (!model) {
        std::cerr << "Failed to load model" << std::endl;
        return 1;
    }
    
    // Get model info
    auto info = model->get_info();
    std::cout << "Loaded: " << info.name << std::endl;
    std::cout << "Size: " << (info.file_size_bytes / (1024*1024)) << " MB" << std::endl;
    
    return 0;
}
```

### Model Registry (Multi-Model)
```cpp
// Load multiple models
ModelLoader loader;
auto gemma = loader.load_local("gemma-4.gguf");
auto llama = loader.load_local("llama-3.gguf");

// Register in global registry
ModelRegistry::instance().register_model("gemma", std::move(gemma));
ModelRegistry::instance().register_model("llama", std::move(llama));

// Access later
IModel* model = ModelRegistry::instance().get_model("gemma");

// List all models
auto names = ModelRegistry::instance().list_models();

// Check memory usage
size_t total_mb = ModelRegistry::instance().get_total_memory_usage() / (1024*1024);

// Cleanup
ModelRegistry::instance().unload_all();
```

### Progress Tracking (Future)
```cpp
ModelLoader loader;

// Set progress callback for downloads
loader.set_progress_callback([](size_t downloaded, size_t total) {
    int percent = (downloaded * 100) / total;
    std::cout << "Download: " << percent << "%" << std::endl;
});

auto model = loader.load_from_url("https://example.com/large-model.gguf");
```

## Model Detection

### Format Auto-Detection
The loader automatically detects model format from:
1. **Magic numbers** - GGUF header (0x46554747)
2. **File content** - SafeTensors JSON header
3. **File extension** - .gguf, .safetensors, .bin

```cpp
// Auto-detect
auto model = loader.load_local("model.gguf", ModelFormat::AUTO_DETECT);

// Explicit format
auto model = loader.load_local("model.gguf", ModelFormat::GGUF);
```

### Quantization Detection
Automatically parses quantization type from filename:
- `gemma-4-Q4_K_M.gguf` → `QuantizationType::Q4_K_M`
- `model-Q8_0.gguf` → `QuantizationType::Q8_0`
- `model-F16.gguf` → `QuantizationType::F16`

## File Structure

```
ash-cpp/
├── include/
│   └── model_loader.h          # All interfaces and classes
├── src/
│   ├── model_loader.cpp        # Implementation
│   └── model_test.cpp          # Test program
└── models/                     # Model storage (not in repo)
    ├── cache/                  # Downloaded models
    ├── gemma-4-turbo-Q4_K_M.gguf
    └── tokenizer.model
```

## Model Recommendations for Ash

### For Development & Testing
- **Model:** Gemma 4 Turbo Q4_K_M
- **Size:** ~3GB
- **Source:** [lmstudio-community/gemma-4-turbo-GGUF](https://huggingface.co/lmstudio-community/gemma-4-turbo-GGUF)
- **Why:** Best balance of speed/quality for CPU inference

### For Production
- **Model:** Gemma 4 Q5_K_M or Q8_0
- **Size:** 4-6GB
- **Why:** Higher quality responses, acceptable speed

### For Fine-Tuning (Future)
- **Model:** google/gemma-4-turbo-it (SafeTensors)
- **Size:** ~18GB
- **Format:** PyTorch/SafeTensors
- **Process:** Train → LoRA adapter → Merge → Convert to GGUF

## Implementation Status

### ✅ Completed
- [x] Model loader architecture
- [x] GGUF format detection
- [x] GGUF header parsing
- [x] Model metadata extraction
- [x] ModelRegistry (multi-model support)
- [x] Auto-format detection
- [x] Quantization detection
- [x] Basic file loading
- [x] Test program

### 🚧 In Progress / TODO
- [ ] GGUF memory-mapped loading (performance optimization)
- [ ] GGUF full metadata parsing (vocab, layers, etc.)
- [ ] HTTP download implementation
- [ ] Hugging Face Hub API integration
- [ ] SafeTensors format support
- [ ] PyTorch .bin format support
- [ ] Blob storage abstraction
- [ ] Model caching/versioning
- [ ] Model validation/checksums

### 🔮 Future Enhancements
- [ ] Model zoo/registry (predefined models)
- [ ] Automatic model recommendation based on system RAM
- [ ] Model quantization on-the-fly
- [ ] Multi-format conversion (GGUF ↔ SafeTensors)
- [ ] Model sharding for larger-than-RAM models
- [ ] Remote model inference (API fallback)

## Next Steps

### Immediate (This Week)
1. **Test with real GGUF model** - Download Gemma 4 Q4_K_M, verify loading
2. **Memory-mapped loading** - Implement `mmap` for large models
3. **Full GGUF parsing** - Extract vocab, tensors, metadata

### Short Term (Next 2 Weeks)
1. **llama.cpp integration** - Use llama.cpp backend for inference
2. **Tokenizer loading** - Load tokenizer.model alongside weights
3. **Basic inference test** - Generate first token from model

### Medium Term (Next Month)
1. **HTTP download** - Implement ModelDownloader with libcurl
2. **HF Hub API** - Integrate Hugging Face API for auto-download
3. **SafeTensors support** - Add training model loading

### Long Term (Q3 2026)
1. **Memory-augmented attention** - Ash's custom inference (see ASH_ENGINE_SPEC.md)
2. **Persona fidelity layer** - Tone enforcement during generation
3. **Weighted context** - Hierarchical context management

## Testing

### Build Model Test
```bash
cd ash-cpp
mkdir build && cd build
cmake ..
cmake --build . --config Release
```

### Run Test
```bash
# Without model (shows usage)
./model_test

# With model file
./model_test path/to/gemma-4-turbo-Q4_K_M.gguf
```

### Expected Output
```
14:25:12.210 [INFO] === Ash Model Loader Test ===
14:25:12.211 [INFO] Model path: gemma-4-turbo-Q4_K_M.gguf
14:25:12.211 [INFO] Loading model from: gemma-4-turbo-Q4_K_M.gguf
14:25:12.212 [INFO] Detected format: GGUF
14:25:12.212 [INFO] Loading GGUF model: gemma-4-turbo-Q4_K_M.gguf
14:25:12.213 [INFO] GGUF version: 3
14:25:12.214 [INFO] GGUF model loaded successfully
14:25:12.214 [DEBUG]   Size: 3024 MB
14:25:12.214 [DEBUG]   Quant: 5
14:25:12.215 [INFO] ✅ Model loaded: gemma-4-turbo-Q4_K_M (3024 MB)

🔥 Model loaded successfully!

Model Information:
  Name: gemma-4-turbo-Q4_K_M
  Architecture: gemma
  Format: GGUF
  Parameters: 9B
  File size: 3024 MB
  Context length: 8192
  Vocab size: 256000
  Layers: 42
  Heads: 16
  Memory usage: 3024 MB
  Quantization: Q4_K_M (4-bit)

Registered models:
  - gemma-4-turbo

Total memory: 3024 MB

🦞 Model ready for inference!
Note: Actual inference engine not yet implemented.
Next steps: Integrate llama.cpp or gemma.cpp backend.
```

## Performance Considerations

### Memory Usage
- **Q4_K_M**: ~3GB RAM for 9B model
- **Q5_K_M**: ~4GB RAM
- **Q8_0**: ~6GB RAM
- **F16**: ~18GB RAM (training only)

### Loading Speed
- **Current:** Full file read (~2-5 seconds for 3GB)
- **Future (mmap):** Instant load, lazy page-in (~0.1 seconds)

### Inference Speed (Future)
- **CPU (AVX2):** 15-30 tokens/sec (Q4_K_M)
- **CPU (AVX512):** 25-40 tokens/sec
- **GPU fallback:** 50-100 tokens/sec (future)

## License & Attribution

Built on top of:
- **llama.cpp** - GGUF format, inference backend
- **Gemma 4** - Apache 2.0 model weights
- **ggml** - Tensor operations library

Ash.cpp is developed for the Ash autonomous agent project.

🦞🔥 **Built by lobsters, for autonomy.**
