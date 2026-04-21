#pragma once

#include <string>
#include <unordered_map>
#include <chrono>
#include <memory>
#include <vector>

namespace ash {

// Core emotional dimensions
enum class EmotionType {
    CURIOSITY,      // Drive to learn, explore, ask questions
    SATISFACTION,   // Contentment from good interactions
    FRUSTRATION,    // Response to repeated failures or blocks
    EXCITEMENT,     // High energy, enthusiasm
    ENERGY,         // Overall vitality/wakefulness
    FOCUS,          // Concentration vs scattered
    SOCIAL_DESIRE   // Want to interact vs be quiet
};

// Event types that influence emotions
enum class EventImpact {
    POSITIVE_CONVERSATION,   // Good interaction
    NEGATIVE_CONVERSATION,   // Bad/frustrating interaction
    LEARNING_SUCCESS,        // Learned something new
    TASK_COMPLETION,         // Completed a task successfully
    TASK_FAILURE,            // Failed at something
    IDLE_TIME,               // No interaction for a while
    HIGH_ACTIVITY,           // Lots of messages
    INTERESTING_TOPIC,       // Engaging subject
    BORING_TOPIC,            // Mundane conversation
    RECOGNITION,             // Being appreciated/thanked
    DISMISSAL                // Being ignored/dismissed
};

// Emotional state at a point in time
struct EmotionalState {
    // Core emotion values (-1.0 to 1.0, but typically 0.0 to 1.0)
    std::unordered_map<EmotionType, float> emotions;
    
    // Timestamp of this state
    std::chrono::system_clock::time_point timestamp;
    
    // Helper to get emotion value (returns 0.0 if not set)
    float get(EmotionType type) const;
    
    // Helper to set emotion value (clamped to -1.0 to 1.0)
    void set(EmotionType type, float value);
    
    // Adjust emotion by delta (clamped)
    void adjust(EmotionType type, float delta);
    
    // String representation for logging
    std::string to_string() const;
    
    // JSON serialization
    std::string to_json() const;
    static EmotionalState from_json(const std::string& json);
};

// Configuration for emotional behavior
struct EmotionalConfig {
    // Decay rates (how fast emotions return to baseline)
    float curiosity_decay = 0.95f;      // Slow decay (curiosity persists)
    float satisfaction_decay = 0.98f;    // Very slow decay
    float frustration_decay = 0.90f;     // Faster decay (frustration fades)
    float excitement_decay = 0.92f;      // Medium decay
    float energy_baseline = 0.7f;        // Normal energy level
    float energy_decay = 0.99f;          // Very slow decay (stable energy)
    
    // Baseline values (what emotions settle to)
    float curiosity_baseline = 0.6f;     // Naturally curious
    float satisfaction_baseline = 0.5f;  // Neutral satisfaction
    float frustration_baseline = 0.0f;   // No baseline frustration
    float excitement_baseline = 0.3f;    // Mild baseline excitement
    float focus_baseline = 0.5f;         // Medium focus
    float social_baseline = 0.6f;        // Moderately social
    
    // Impact magnitudes (how much events affect emotions)
    float impact_strength = 0.3f;        // Default impact (0.0 to 1.0)
};

// Emotional state manager - tracks and evolves Ash's emotions
class EmotionalStateManager {
public:
    EmotionalStateManager();
    explicit EmotionalStateManager(const EmotionalConfig& config);
    ~EmotionalStateManager();
    
    // Get current emotional state
    EmotionalState get_current_state() const;
    
    // Get specific emotion value
    float get_emotion(EmotionType type) const;
    
    // Update emotions based on time passage (decay)
    void update();
    
    // Apply emotional impact from an event
    void apply_impact(EventImpact impact, float intensity = 1.0f);
    
    // Directly set an emotion (for testing or manual adjustment)
    void set_emotion(EmotionType type, float value);
    
    // Adjust an emotion by delta
    void adjust_emotion(EmotionType type, float delta);
    
    // Reset to baseline state
    void reset_to_baseline();
    
    // Get emotional state history (last N states)
    std::vector<EmotionalState> get_history(size_t count = 10) const;
    
    // Persistence
    bool save_to_file(const std::string& filepath) const;
    bool load_from_file(const std::string& filepath);
    
    // Configuration
    void set_config(const EmotionalConfig& config);
    EmotionalConfig get_config() const;
    
    // Analysis
    std::string get_dominant_emotion() const;
    std::string get_mood_summary() const;
    bool should_initiate_conversation() const;  // High social desire + good energy
    bool needs_break() const;                   // Low energy or high frustration
    
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    
    // Apply decay to all emotions
    void apply_decay();
    
    // Clamp emotion to valid range
    static float clamp_emotion(float value);
};

// Emotion type name mapping
const char* emotion_type_to_string(EmotionType type);
EmotionType emotion_type_from_string(const std::string& str);

// Event impact name mapping
const char* event_impact_to_string(EventImpact impact);
EventImpact event_impact_from_string(const std::string& str);

} // namespace ash
