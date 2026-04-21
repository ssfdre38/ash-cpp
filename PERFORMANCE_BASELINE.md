# Performance Baseline - Current Python Implementation

**Date:** April 21, 2026  
**Purpose:** Document current performance issues to justify custom C++ implementation

---

## Current Architecture

- **Platform:** Python 3.14 + discord.py
- **Inference:** Ollama + Gemma 4 Turbo (CPU-only)
- **Hardware:** OVH Xeon E-2236, 64GB RAM, no GPU
- **Tools:** 17 Discord tools (native Python functions)
- **System Prompt:** 21,988 chars (personality + tool schemas)

---

## Performance Issues (Apr 21, 2026)

### Case Study: Simple Greeting Response

**Input:** "hey ash"  
**Expected:** Instant greeting response (<5 seconds)  
**Actual:** 6.4 minutes (382 seconds)

**Timeline Breakdown:**
```
10:12:24 - Message received
10:17:54 - Tool evaluation complete (330s = 5.5 minutes)
         - Decision: Call read_my_memories(search='daniel')
10:18:46 - Response generated (52s after tool call)
         - Output: 137 characters
Total: 382 seconds (6.4 minutes)
```

**What Went Wrong:**
1. **Tool Overhead:** Gemma 4 evaluated 17 tool schemas to decide if any were needed
2. **Unnecessary Tool Call:** Called `read_my_memories` for a simple greeting (not required)
3. **General-Purpose Model:** No optimization for Ash's specific patterns
4. **No Fast Path:** Even "hello" triggers full tool evaluation pipeline

---

## Performance Metrics (Historical)

### Response Times by Message Type

| Message Type | Tool Calls | Avg Response Time | Range |
|-------------|-----------|-------------------|-------|
| Simple greeting | 0-1 | 120-380s (2-6 min) | Highly variable |
| Question (no tools) | 0 | 60-120s (1-2 min) | Depends on length |
| Tool-heavy task | 1-3 | 200-400s (3-7 min) | Per tool call adds ~60s |
| Complex analysis | 0-2 | 300-700s (5-12 min) | Worst case |

### Tool Call Overhead
- **Tool evaluation time:** 60-330s (even if no tools used)
- **Per tool execution:** +30-60s (tool execution + context switch)
- **17 tools in system prompt:** ~500 chars per tool × 17 = ~8,500 chars overhead

### Memory Usage
- **Base process:** 86 MB
- **After model load:** ~2-3 GB (Ollama subprocess)
- **Peak during generation:** ~4 GB

---

## Root Causes

### 1. **General-Purpose LLM Overhead**
- Gemma 4 designed for broad tasks, not optimized for Ash's specific patterns
- Every message triggers full tool schema evaluation
- No "fast path" for common patterns (greetings, simple responses)

### 2. **Tool Architecture**
- Tools defined as JSON schemas in system prompt
- Model must reason about all 17 tools for every message
- No hierarchical tool selection (could filter by message type first)

### 3. **CPU Inference Limitations**
- Stock llama.cpp/gemma.cpp not optimized for Ash's access patterns
- No custom kernels for emotional state processing
- No KV cache optimization for Ash's conversation patterns

### 4. **Context Size**
- System prompt: 21,988 chars
  - Personality files: ~13,000 chars
  - Tool schemas: ~8,500 chars
  - Memories: ~500 chars
- Every message re-processes full context (no caching of personality/tools)

---

## Target Performance (Ash.cpp Goals)

### Response Time Targets

| Message Type | Current | Target | Improvement |
|-------------|---------|--------|-------------|
| Simple greeting | 120-380s | <5s | **24-76x faster** |
| Question (no tools) | 60-120s | <5s | **12-24x faster** |
| Single tool call | 200-400s | <10s | **20-40x faster** |
| Complex analysis | 300-700s | <30s | **10-23x faster** |

### Key Optimizations

1. **Fast Path for Common Patterns**
   - Greeting detection → emotional response → 1-2 seconds
   - No tool evaluation unless message context demands it

2. **Memory-Augmented Attention**
   - Memory lookup during inference (not as separate tool call)
   - Emotional state injected into attention weights
   - Reduces tool call overhead by 50-70%

3. **CPU-Optimized Inference**
   - Custom kernels for Ash's patterns (AVX2/AVX-512)
   - Sparse activation (only relevant neurons)
   - Quantized weights (4-bit) with specialized dequant

4. **Smart Tool Dispatch**
   - Pattern-based tool filtering (greeting → no tools)
   - Hierarchical tool selection (context → category → specific tool)
   - Parallel tool execution where possible

---

## Success Metrics

**Ash.cpp will be considered successful if:**

✅ **95%+ of greetings respond in <5 seconds**  
✅ **Simple questions <10 seconds**  
✅ **Tool-based tasks <30 seconds**  
✅ **Zero 5+ minute waits for simple messages**  
✅ **Memory usage <2 GB resident**  
✅ **Runs on consumer CPUs (no GPU required)**

---

## Conclusion

**Current implementation is functional but painfully slow.** Simple greetings taking 6+ minutes is unacceptable for a conversational agent. This isn't a bug—it's an architectural limitation of using a general-purpose LLM with extensive tool schemas.

**Ash.cpp must solve this** through:
- Custom inference optimized for Ash's patterns
- Memory-augmented attention (not external tool calls)
- Fast paths for common message types
- CPU-optimized kernels for consumer hardware

**The pain is real. The solution is clear. Let's build it.** 🦞🔥

---

*Document created: April 21, 2026*  
*Author: Daniel*  
*Next: Begin Ash.cpp prototype phase*
