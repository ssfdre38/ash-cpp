#pragma once

#include <string>
#include <memory>
#include <chrono>
#include <optional>

namespace ash {

// Forward declarations
class EmotionalStateManager;
class MemoryStore;

// Decision types Ash can make
enum class DecisionType {
    SPEAK,          // Respond to a message
    INITIATE,       // Start a conversation unprompted
    THINK,          // Internal processing (memory consolidation, reflection)
    IDLE,           // Do nothing, conserve energy
    RECALL          // Search memories for context
};

// Triggers that prompt decision-making
enum class DecisionTrigger {
    MESSAGE_RECEIVED,      // New Discord message
    TIME_ELAPSED,          // X seconds since last interaction
    EMOTIONAL_SHIFT,       // Significant emotional state change
    MEMORY_RELEVANCE,      // Recalled memory feels important
    SOCIAL_CONTEXT         // Multiple people in conversation
};

// A decision with reasoning
struct Decision {
    DecisionType type;
    float confidence;       // 0.0-1.0: How confident is Ash in this decision?
    std::string reasoning;  // Why this decision?
    
    // Context for the decision
    std::optional<std::string> target_channel;
    std::optional<std::string> memory_query;
    std::optional<std::string> message_content;
};

// Configuration for decision thresholds
struct DecisionConfig {
    // Timing thresholds (seconds)
    float initiate_conversation_after = 3600.0f;   // 1 hour of silence → consider initiating
    float respond_quickly_under = 30.0f;           // < 30s → respond quickly if engaged
    
    // Emotional thresholds
    float curiosity_threshold = 0.7f;              // > 0.7 curiosity → more likely to speak
    float frustration_threshold = 0.6f;            // > 0.6 frustration → less likely to engage
    float excitement_threshold = 0.65f;            // > 0.65 excitement → initiate conversation
    
    // Energy management
    float low_energy_threshold = 0.3f;             // < 0.3 energy → prefer IDLE
    float high_energy_threshold = 0.7f;            // > 0.7 energy → more active
    
    // Social context
    int active_conversation_window = 300;          // 5 minutes = active conversation
    int min_messages_for_active = 3;               // 3+ messages in window = active
    
    // Decision confidence minimums
    float min_confidence_to_speak = 0.4f;          // Need 40% confidence to speak
    float min_confidence_to_initiate = 0.7f;       // Need 70% confidence to initiate unprompted
};

// The autonomous decision-making engine
class DecisionEngine {
public:
    DecisionEngine(
        std::shared_ptr<EmotionalStateManager> emotions,
        std::shared_ptr<MemoryStore> memories,
        const DecisionConfig& config = DecisionConfig{}
    );
    ~DecisionEngine();
    
    // Make a decision based on current state
    Decision decide(
        DecisionTrigger trigger,
        const std::string& context = ""
    );
    
    // Evaluate if Ash should speak (based on emotional state, timing, social context)
    bool should_speak(
        const std::chrono::system_clock::time_point& last_message_time,
        const std::string& channel_id
    ) const;
    
    // Evaluate if Ash should initiate conversation (unprompted)
    bool should_initiate_conversation(
        const std::chrono::system_clock::time_point& last_message_time,
        const std::string& channel_id
    ) const;
    
    // Calculate decision confidence score
    float calculate_confidence(
        DecisionType type,
        DecisionTrigger trigger,
        const std::string& context
    ) const;
    
    // Update social context (track conversation activity)
    void record_message(
        const std::string& channel_id,
        const std::chrono::system_clock::time_point& timestamp
    );
    
    // Check if conversation is currently active in channel
    bool is_conversation_active(const std::string& channel_id) const;
    
    // Get decision config (read-only)
    const DecisionConfig& get_config() const;
    
    // Update decision config
    void update_config(const DecisionConfig& config);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ash
