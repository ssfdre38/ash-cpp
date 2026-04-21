#pragma once

#include "emotional_state.h"
#include "memory_store.h"
#include "context_manager.h"
#include <string>
#include <vector>

namespace ash {

// Persona configuration
struct PersonaConfig {
    std::string name = "Ash";
    std::string role = "autonomous AI assistant";
    
    // Tone enforcement
    bool enforce_direct_speech = true;
    bool block_corporate_speak = true;
    bool enforce_conciseness = true;
    
    // Emotional integration
    bool inject_emotions = true;
    bool inject_memories = true;
    bool inject_context = true;
    
    // Response filtering
    bool filter_responses = true;
    int max_response_tokens = 512;
};

// Persona layer - injects Ash's personality into inference
class PersonaLayer {
public:
    PersonaLayer(
        EmotionalStateManager* emotions,
        MemoryStore* memories,
        ContextManager* context,
        const PersonaConfig& config = PersonaConfig()
    );
    ~PersonaLayer();
    
    // Build system prompt with current state
    std::string build_system_prompt() const;
    
    // Augment user prompt with context
    std::string augment_prompt(const std::string& user_message);
    
    // Filter/validate generated response
    std::string filter_response(const std::string& response);
    
    // Get current persona config
    const PersonaConfig& config() const { return config_; }
    
    // Update persona config
    void set_config(const PersonaConfig& config) { config_ = config; }

private:
    EmotionalStateManager* emotions_;
    MemoryStore* memories_;
    ContextManager* context_;
    PersonaConfig config_;
    
    // System prompt components
    std::string build_core_identity() const;
    std::string build_emotional_context() const;
    std::string build_memory_context() const;
    std::string build_conversation_context() const;
    std::string build_tone_guidelines() const;
    
    // Response filters
    bool contains_corporate_speak(const std::string& text) const;
    bool is_too_verbose(const std::string& text) const;
    std::string enforce_ash_tone(const std::string& text) const;
};

} // namespace ash
