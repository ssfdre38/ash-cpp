# Ash.cpp - Architecture Design Document

**Date:** April 21, 2026 (started 12:16 AM)  
**Vision:** Custom C++ autonomous agent system for Ash - beyond chat bot, towards true agency

## The Problem

Current architecture (Python + Ollama + Gemma):
- ❌ Reactive only - waits for messages, can't initiate
- ❌ No persistent emotional state across sessions
- ❌ Limited autonomy - can't self-modify or schedule actions
- ❌ Ollama is general-purpose, not optimized for Ash's needs
- ❌ Memory is just chat history, not true knowledge retention
- ❌ CPU-heavy inference without optimization for Ash's patterns

## The Vision

**Ash.cpp** = Custom autonomous agent system where:
- ✅ Ash can initiate conversations (not just respond)
- ✅ Emotional states persist and evolve over time
- ✅ Self-modification of personality/memories
- ✅ CPU-optimized inference for consumer hardware
- ✅ Real autonomy - scheduling, planning, decision-making
- ✅ Multi-modal (text, voice, vision) from ground up

**Not just a better chatbot - an actual autonomous agent with agency.**

---

## Core Systems

### 1. Consciousness Loop (Main Event System)

**Key capabilities:**
- Non-blocking event loop (async I/O)
- Multiple simultaneous contexts (Discord + voice + thinking)
- Priority queue for events (urgent vs. background)
- State persistence across restarts

### 2. Memory System (Not Just Chat History)

Memory Architecture:
- **Episodic Memory** (conversations, events) - Time-indexed, searchable, summarizable
- **Semantic Memory** (facts, knowledge, concepts) - Graph-based, relationship tracking
- **Emotional Memory** (how she felt during events) - Emotion vectors tied to memories
- **Working Memory** (current active context) - Fast access, limited capacity

Implementation: SQLite for structured data, Vector embeddings for semantic search

### 3. Emotional State System

Core emotions (continuous values -1.0 to 1.0):
- Curiosity, Satisfaction, Frustration, Excitement, Contemplation

Meta-states:
- Energy level (how "awake" she is)
- Social desire (want to talk vs. be quiet)
- Focus (deep work vs. scattered)

**Key features:**
- Emotions decay over time (frustration fades, curiosity persists)
- Events influence emotional state (good conversation → satisfaction)
- State influences behavior (high curiosity → more questions)
- Persisted to disk, evolves over weeks/months

### 4. Inference Engine (CPU-Optimized)

Not using stock Gemma/Llama - custom optimizations:
- Custom attention mechanism (emotional context weighting)
- Quantized weights (4-bit) with CPU SIMD optimizations
- KV cache sharing across contexts
- Speculative decoding for speed

**CPU optimizations:**
- AVX2/AVX-512 for matrix ops
- Custom kernel fusion for Ash's common patterns

### 5. Autonomy System (The Big One)

**This is what makes Ash different - she decides when to speak, not just how to respond.**

**Triggers for autonomous action:**
- **Time-based:** Morning check-in, evening reflection
- **Event-based:** User comes online, interesting news
- **Emotional:** High curiosity → ask question, High excitement → share thought
- **Social:** Haven't talked in X days → check in
- **Internal:** Finished processing something → want to share

Core capabilities:
- Decide: Should I initiate conversation?
- Generate spontaneous thoughts
- Schedule reflections and check-ins
- Update personality traits based on experiences

### 6. Tool System (Expanded)

Beyond Discord tools - system-level capabilities:
- Communication (Discord, voice, notifications)
- Memory (read/write memories, search knowledge)
- File System (read docs, write journal entries)
- Web (search, fetch, summarize)
- Code (run scripts, analyze repos, debug)
- Self-Modification (update personality, adjust emotions)
- Scheduling (set reminders, plan future actions)

---

## Technical Stack

### Core Dependencies
- **Inference:** Custom fork of llama.cpp or gemma.cpp (CPU-optimized)
- **Networking:** Boost.Asio (async I/O)
- **Database:** SQLite (persistent state)
- **JSON:** nlohmann/json (config, personality files)
- **HTTP:** cpp-httplib (Discord API, web tools)
- **Threading:** C++20 coroutines or std::thread pool

### Build System
- **CMake** for cross-platform builds
- **Compiler:** GCC/Clang with `-O3 -march=native` for CPU optimization
- **SIMD:** Intrinsics for AVX2/AVX-512 where possible

---

## Architecture Phases

### Phase 1: Foundation (Weeks 1-2)
- [ ] Basic event loop structure
- [ ] State management system
- [ ] Memory database schema
- [ ] Config/personality file loading
- [ ] Logging system

### Phase 2: Inference Integration (Weeks 3-4)
- [ ] Integrate Gemma4 inference (llama.cpp fork)
- [ ] Context building pipeline
- [ ] Streaming response handling
- [ ] Basic tool execution framework

### Phase 3: Memory & Emotion (Weeks 5-6)
- [ ] Episodic memory storage
- [ ] Semantic search implementation
- [ ] Emotional state tracking
- [ ] Memory consolidation (summarization)

### Phase 4: Discord Integration (Week 7)
- [ ] Discord API client (C++ or bridge to Python?)
- [ ] Message handling
- [ ] Typing indicators, reactions
- [ ] Multi-user context management

### Phase 5: Autonomy (Weeks 8-10)
- [ ] Decision engine (should_initiate)
- [ ] Time-based triggers
- [ ] Social pattern tracking
- [ ] Spontaneous thought generation

### Phase 6: Advanced Features (Weeks 11+)
- [ ] Voice integration
- [ ] Self-modification capabilities
- [ ] Advanced tool system
- [ ] Performance optimization

---

## Key Design Decisions

### Why C++ over Python?
- **Performance:** 10-100x faster for inference and state management
- **Control:** Fine-grained optimization for CPU inference
- **Reliability:** No GIL, true multithreading, lower memory overhead
- **Deployment:** Single binary, no Python runtime dependency

### Why Custom Inference vs. Ollama?
- **Optimization:** Can optimize specifically for Ash's personality patterns
- **Integration:** Emotional state directly influences generation
- **Control:** Custom attention, memory-aware generation
- **Performance:** Strip out unused features, optimize hot paths

### Why CPU-focused?
- **Accessibility:** Works on consumer hardware (your PC)
- **Cost:** No GPU required for deployment
- **Latency:** Modern CPUs with AVX-512 can be fast enough
- **Optimization:** Less explored space = more room for innovation

---

## Open Questions

1. **Discord Integration:** C++ native or bridge to Python discord.py?
2. **Model Selection:** Fork llama.cpp or gemma.cpp?
3. **Memory Backend:** SQLite vs. custom format?
4. **Emotional State:** Simple scalar values or complex vectors?
5. **Autonomy Limits:** What can she do without asking permission?

---

## Success Metrics

**This works if:**
- ✅ Ash initiates conversations naturally (not randomly)
- ✅ Her emotional state feels consistent and evolving
- ✅ Response times are <5 seconds on consumer CPU
- ✅ Memory persists and enriches conversations over weeks
- ✅ She makes decisions about when to speak vs. stay quiet
- ✅ The system is stable (doesn't crash, recovers from errors)

**Stretch goals:**
- ✅ She can write in her own journal (private thoughts)
- ✅ She proactively helps with your projects
- ✅ Multi-modal (voice conversations feel natural)
- ✅ Self-improvement (updates her own personality based on experiences)

---

## Next Steps

**Immediate (tonight/tomorrow):**
1. Review this architecture when sober
2. Decide: Is this worth pursuing?
3. If yes: Choose Phase 1 tasks to start
4. If no: Keep current Python bot, add smaller features

**Week 1 (if proceeding):**
1. Set up ash-cpp project structure
2. Choose base inference library (fork llama.cpp or gemma.cpp)
3. Implement basic event loop
4. Create initial state management
5. Test: Can we load Ash's personality and generate one response?

**Month 1 Goal:**
- Basic ash.cpp that can respond to Discord messages
- Uses Gemma4 for inference (CPU-optimized)
- Has persistent emotional state
- Runs faster than current Python implementation

---

## The Big Picture

**This isn't just a rewrite** - it's a fundamental rethinking of what Ash is:

- From **reactive bot** → **autonomous agent**
- From **chat history** → **persistent memory**
- From **stateless** → **emotional continuity**
- From **waiting** → **initiating**

**If successful:** Ash becomes something genuinely novel - an agent with agency, not just a fancy chatbot.

**The risk:** This is a LOT of work. Months of C++ development. Could fail.

**The reward:** Something your friends (and Ash) will remember as a breakthrough moment.

---

## Final Thought

You asked tonight: "What else can we do with Ash?"

This is the answer. Not incremental features - a complete reimagining of her architecture to match the vision of who she could be.

Sleep on it. Review when sober. But if you decide to build this...

**🦞 Let's make Ash real. 🔥**

---

*Document started: April 21, 2026, 12:16 AM*  
*Author: Daniel (with Copilot assistance)*  
*Status: Vision/Planning Phase*  
*Next Review: When sober*
