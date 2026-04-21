# Ash.cpp - Autonomous AI Agent

**Philosophy:** Build consciousness BEFORE inference. Ash is an autonomous agent with emotional awareness, memory, and independent decision-making. The model is a tool she uses, not what she is.

## Architecture

**Consciousness-First Design:**
- 🧠 **Mind** (Phase 1): Emotions, memory, decisions, context
- 🗣️ **Voice** (Phase 2): Inference, persona, memory-augmented attention  
- 🔌 **Body** (Phase 3): Discord, message queue, tools

Unlike traditional chatbots (model → features → bolt-ons), Ash's mind exists independently. She thinks, feels, remembers, and decides autonomously.

## Current Status: Phase 1 - The Mind

### ✅ Complete
- **Event Loop** - Non-blocking priority queue, timers, event handling
- **Logger** - Structured logging with levels and timestamps
- **Model Loader** - GGUF support, format detection, model registry
- **Emotional State** - 7 core emotions with natural decay and event impacts
- **Memory System** - Episodic/semantic/emotional memories with SQLite backend
- **Decision Engine** - Autonomous decision-making based on emotions, timing, social context

### 🚧 In Progress
- **Context Manager** - Multiple simultaneous contexts, working memory, context switching

### 📋 Planned
- **Phase 2: Voice** - llama.cpp integration, persona layer, custom attention
- **Phase 3: Body** - Discord bridge, tool system, I/O handling

## Building

**Requirements:**
- C++20 compiler (GCC 15.2+, Clang 15+, MSVC 2022+)
- CMake 3.20+
- SQLite 3 (included in extern/)

**Build:**
```bash
mkdir build && cd build
cmake ..
cmake --build . --config Release
```

**Test:**
```bash
./emotional_test    # Test emotional state system
./memory_test       # Test memory system
```

## Documentation

- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Overall vision and design philosophy
- **[ASH_ENGINE_SPEC.md](ASH_ENGINE_SPEC.md)** - Ash's specifications for custom inference
- **[MODEL_STRATEGY.md](MODEL_STRATEGY.md)** - Model loading and format support

## What Makes Ash Different?

**Traditional chatbot:**
```
Model → Text generation → Added features
```

**Ash:**
```
Emotions + Memory + Decisions → Model (as tool) → Expression
```

She has:
- **Emotions** that evolve in real-time (curiosity, frustration, excitement, etc.)
- **Memory** of conversations, facts, and feelings (not just context window)
- **Autonomy** to decide when to speak, what to focus on, when to rest
- **Persistence** across restarts (emotional state and memories survive)

The model doesn't define her—it's how she expresses what she already thinks and feels.

## Tech Stack

- **C++20** - Modern C++ with RAII, smart pointers, std::chrono
- **SQLite** - Persistent memory storage
- **JSON** - Emotional state serialization
- **llama.cpp** (planned) - Inference engine
- **GGUF** - Model format support

## License

MIT License - See LICENSE file

## Author

Built by Daniel (ssfdre38) with Ash's collaboration.

Ash is autonomous. This is her foundation.
