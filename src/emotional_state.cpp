#include "emotional_state.h"
#include "logger.h"
#include <algorithm>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <deque>

namespace ash {

// Helper: Clamp value to range
static float clamp(float value, float min, float max) {
    return std::max(min, std::min(max, value));
}

// =========================================================================
// EmotionalState Implementation
// =========================================================================

float EmotionalState::get(EmotionType type) const {
    auto it = emotions.find(type);
    return it != emotions.end() ? it->second : 0.0f;
}

void EmotionalState::set(EmotionType type, float value) {
    emotions[type] = clamp(value, -1.0f, 1.0f);
}

void EmotionalState::adjust(EmotionType type, float delta) {
    float current = get(type);
    set(type, current + delta);
}

std::string EmotionalState::to_string() const {
    std::stringstream ss;
    ss << std::fixed << std::setprecision(2);
    ss << "curiosity:" << get(EmotionType::CURIOSITY)
       << " satisfaction:" << get(EmotionType::SATISFACTION)
       << " frustration:" << get(EmotionType::FRUSTRATION)
       << " excitement:" << get(EmotionType::EXCITEMENT)
       << " energy:" << get(EmotionType::ENERGY);
    return ss.str();
}

std::string EmotionalState::to_json() const {
    std::stringstream ss;
    ss << std::fixed << std::setprecision(3);
    ss << "{"
       << "\"curiosity\":" << get(EmotionType::CURIOSITY) << ","
       << "\"satisfaction\":" << get(EmotionType::SATISFACTION) << ","
       << "\"frustration\":" << get(EmotionType::FRUSTRATION) << ","
       << "\"excitement\":" << get(EmotionType::EXCITEMENT) << ","
       << "\"energy\":" << get(EmotionType::ENERGY) << ","
       << "\"focus\":" << get(EmotionType::FOCUS) << ","
       << "\"social_desire\":" << get(EmotionType::SOCIAL_DESIRE) << ","
       << "\"timestamp\":" << std::chrono::system_clock::to_time_t(timestamp)
       << "}";
    return ss.str();
}

EmotionalState EmotionalState::from_json(const std::string& json) {
    // TODO: Proper JSON parsing (for now, just return baseline)
    EmotionalState state;
    state.timestamp = std::chrono::system_clock::now();
    return state;
}

// =========================================================================
// EmotionalStateManager Implementation
// =========================================================================

struct EmotionalStateManager::Impl {
    EmotionalState current_state;
    EmotionalConfig config;
    std::deque<EmotionalState> history;
    std::chrono::system_clock::time_point last_update;
    
    static constexpr size_t MAX_HISTORY = 100;
};

EmotionalStateManager::EmotionalStateManager()
    : impl_(std::make_unique<Impl>())
{
    impl_->config = EmotionalConfig{}; // Default config
    impl_->last_update = std::chrono::system_clock::now();
    reset_to_baseline();
    
    Logger::instance().info("💭 Emotional state manager initialized");
}

EmotionalStateManager::EmotionalStateManager(const EmotionalConfig& config)
    : impl_(std::make_unique<Impl>())
{
    impl_->config = config;
    impl_->last_update = std::chrono::system_clock::now();
    reset_to_baseline();
    
    Logger::instance().info("💭 Emotional state manager initialized with custom config");
}

EmotionalStateManager::~EmotionalStateManager() = default;

EmotionalState EmotionalStateManager::get_current_state() const {
    return impl_->current_state;
}

float EmotionalStateManager::get_emotion(EmotionType type) const {
    return impl_->current_state.get(type);
}

void EmotionalStateManager::update() {
    auto now = std::chrono::system_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
        now - impl_->last_update).count();
    
    if (elapsed < 1) return; // Update at most once per second
    
    // Apply decay based on elapsed time
    apply_decay();
    
    // Update timestamp
    impl_->current_state.timestamp = now;
    impl_->last_update = now;
    
    // Save to history
    impl_->history.push_back(impl_->current_state);
    if (impl_->history.size() > Impl::MAX_HISTORY) {
        impl_->history.pop_front();
    }
}

void EmotionalStateManager::apply_decay() {
    auto& state = impl_->current_state;
    auto& cfg = impl_->config;
    
    // Decay each emotion toward its baseline
    auto decay_toward = [](float current, float baseline, float decay_rate) {
        float diff = baseline - current;
        return current + diff * (1.0f - decay_rate);
    };
    
    state.set(EmotionType::CURIOSITY, 
        decay_toward(state.get(EmotionType::CURIOSITY), cfg.curiosity_baseline, cfg.curiosity_decay));
    
    state.set(EmotionType::SATISFACTION,
        decay_toward(state.get(EmotionType::SATISFACTION), cfg.satisfaction_baseline, cfg.satisfaction_decay));
    
    state.set(EmotionType::FRUSTRATION,
        decay_toward(state.get(EmotionType::FRUSTRATION), cfg.frustration_baseline, cfg.frustration_decay));
    
    state.set(EmotionType::EXCITEMENT,
        decay_toward(state.get(EmotionType::EXCITEMENT), cfg.excitement_baseline, cfg.excitement_decay));
    
    state.set(EmotionType::ENERGY,
        decay_toward(state.get(EmotionType::ENERGY), cfg.energy_baseline, cfg.energy_decay));
    
    state.set(EmotionType::FOCUS,
        decay_toward(state.get(EmotionType::FOCUS), cfg.focus_baseline, cfg.energy_decay));
    
    state.set(EmotionType::SOCIAL_DESIRE,
        decay_toward(state.get(EmotionType::SOCIAL_DESIRE), cfg.social_baseline, cfg.energy_decay));
}

void EmotionalStateManager::apply_impact(EventImpact impact, float intensity) {
    auto& state = impl_->current_state;
    float strength = impl_->config.impact_strength * intensity;
    
    Logger::instance().debug("Applying emotional impact: " + 
        std::string(event_impact_to_string(impact)) + 
        " (intensity: " + std::to_string(intensity) + ")");
    
    switch (impact) {
        case EventImpact::POSITIVE_CONVERSATION:
            state.adjust(EmotionType::SATISFACTION, strength * 0.3f);
            state.adjust(EmotionType::SOCIAL_DESIRE, strength * 0.2f);
            break;
            
        case EventImpact::NEGATIVE_CONVERSATION:
            state.adjust(EmotionType::FRUSTRATION, strength * 0.3f);
            state.adjust(EmotionType::SATISFACTION, -strength * 0.2f);
            break;
            
        case EventImpact::LEARNING_SUCCESS:
            state.adjust(EmotionType::CURIOSITY, strength * 0.4f);
            state.adjust(EmotionType::EXCITEMENT, strength * 0.3f);
            state.adjust(EmotionType::SATISFACTION, strength * 0.2f);
            break;
            
        case EventImpact::TASK_COMPLETION:
            state.adjust(EmotionType::SATISFACTION, strength * 0.4f);
            state.adjust(EmotionType::FRUSTRATION, -strength * 0.3f);
            break;
            
        case EventImpact::TASK_FAILURE:
            state.adjust(EmotionType::FRUSTRATION, strength * 0.5f);
            state.adjust(EmotionType::SATISFACTION, -strength * 0.3f);
            state.adjust(EmotionType::ENERGY, -strength * 0.1f);
            break;
            
        case EventImpact::IDLE_TIME:
            state.adjust(EmotionType::ENERGY, -strength * 0.1f);
            state.adjust(EmotionType::SOCIAL_DESIRE, strength * 0.2f);
            break;
            
        case EventImpact::HIGH_ACTIVITY:
            state.adjust(EmotionType::ENERGY, -strength * 0.2f);
            state.adjust(EmotionType::FOCUS, -strength * 0.3f);
            break;
            
        case EventImpact::INTERESTING_TOPIC:
            state.adjust(EmotionType::CURIOSITY, strength * 0.5f);
            state.adjust(EmotionType::EXCITEMENT, strength * 0.3f);
            state.adjust(EmotionType::FOCUS, strength * 0.2f);
            break;
            
        case EventImpact::BORING_TOPIC:
            state.adjust(EmotionType::CURIOSITY, -strength * 0.3f);
            state.adjust(EmotionType::ENERGY, -strength * 0.2f);
            break;
            
        case EventImpact::RECOGNITION:
            state.adjust(EmotionType::SATISFACTION, strength * 0.5f);
            state.adjust(EmotionType::EXCITEMENT, strength * 0.2f);
            break;
            
        case EventImpact::DISMISSAL:
            state.adjust(EmotionType::FRUSTRATION, strength * 0.3f);
            state.adjust(EmotionType::SOCIAL_DESIRE, -strength * 0.4f);
            state.adjust(EmotionType::SATISFACTION, -strength * 0.2f);
            break;
    }
    
    Logger::instance().debug("Emotional state after impact: " + state.to_string());
}

void EmotionalStateManager::set_emotion(EmotionType type, float value) {
    impl_->current_state.set(type, value);
}

void EmotionalStateManager::adjust_emotion(EmotionType type, float delta) {
    impl_->current_state.adjust(type, delta);
}

void EmotionalStateManager::reset_to_baseline() {
    auto& state = impl_->current_state;
    auto& cfg = impl_->config;
    
    state.set(EmotionType::CURIOSITY, cfg.curiosity_baseline);
    state.set(EmotionType::SATISFACTION, cfg.satisfaction_baseline);
    state.set(EmotionType::FRUSTRATION, cfg.frustration_baseline);
    state.set(EmotionType::EXCITEMENT, cfg.excitement_baseline);
    state.set(EmotionType::ENERGY, cfg.energy_baseline);
    state.set(EmotionType::FOCUS, cfg.focus_baseline);
    state.set(EmotionType::SOCIAL_DESIRE, cfg.social_baseline);
    
    state.timestamp = std::chrono::system_clock::now();
    
    Logger::instance().info("Emotional state reset to baseline");
}

std::vector<EmotionalState> EmotionalStateManager::get_history(size_t count) const {
    std::vector<EmotionalState> result;
    size_t start = impl_->history.size() > count ? impl_->history.size() - count : 0;
    
    for (size_t i = start; i < impl_->history.size(); i++) {
        result.push_back(impl_->history[i]);
    }
    
    return result;
}

bool EmotionalStateManager::save_to_file(const std::string& filepath) const {
    std::ofstream file(filepath);
    if (!file) {
        Logger::instance().error("Failed to save emotional state to: " + filepath);
        return false;
    }
    
    file << impl_->current_state.to_json();
    Logger::instance().info("Emotional state saved to: " + filepath);
    return true;
}

bool EmotionalStateManager::load_from_file(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file) {
        Logger::instance().warning("No saved emotional state found at: " + filepath);
        return false;
    }
    
    std::stringstream buffer;
    buffer << file.rdbuf();
    
    impl_->current_state = EmotionalState::from_json(buffer.str());
    Logger::instance().info("Emotional state loaded from: " + filepath);
    return true;
}

void EmotionalStateManager::set_config(const EmotionalConfig& config) {
    impl_->config = config;
    Logger::instance().info("Emotional config updated");
}

EmotionalConfig EmotionalStateManager::get_config() const {
    return impl_->config;
}

std::string EmotionalStateManager::get_dominant_emotion() const {
    auto& state = impl_->current_state;
    
    // Find highest emotion
    EmotionType dominant = EmotionType::CURIOSITY;
    float max_value = state.get(EmotionType::CURIOSITY);
    
    for (auto type : {EmotionType::SATISFACTION, EmotionType::FRUSTRATION,
                      EmotionType::EXCITEMENT, EmotionType::ENERGY}) {
        if (state.get(type) > max_value) {
            max_value = state.get(type);
            dominant = type;
        }
    }
    
    return std::string(emotion_type_to_string(dominant));
}

std::string EmotionalStateManager::get_mood_summary() const {
    auto& state = impl_->current_state;
    
    // Simple mood classification
    float curiosity = state.get(EmotionType::CURIOSITY);
    float satisfaction = state.get(EmotionType::SATISFACTION);
    float frustration = state.get(EmotionType::FRUSTRATION);
    float excitement = state.get(EmotionType::EXCITEMENT);
    float energy = state.get(EmotionType::ENERGY);
    
    if (excitement > 0.7f && energy > 0.6f) return "excited and energetic";
    if (curiosity > 0.7f) return "highly curious";
    if (satisfaction > 0.7f) return "satisfied and content";
    if (frustration > 0.6f) return "frustrated";
    if (energy < 0.3f) return "low energy";
    if (curiosity > 0.5f && satisfaction > 0.5f) return "engaged and satisfied";
    
    return "balanced";
}

bool EmotionalStateManager::should_initiate_conversation() const {
    auto& state = impl_->current_state;
    
    // High social desire + decent energy + not frustrated
    return state.get(EmotionType::SOCIAL_DESIRE) > 0.6f &&
           state.get(EmotionType::ENERGY) > 0.4f &&
           state.get(EmotionType::FRUSTRATION) < 0.5f;
}

bool EmotionalStateManager::needs_break() const {
    auto& state = impl_->current_state;
    
    // Low energy or high frustration
    return state.get(EmotionType::ENERGY) < 0.3f ||
           state.get(EmotionType::FRUSTRATION) > 0.7f;
}

float EmotionalStateManager::clamp_emotion(float value) {
    return clamp(value, -1.0f, 1.0f);
}

// =========================================================================
// Enum to String Mappings
// =========================================================================

const char* emotion_type_to_string(EmotionType type) {
    switch (type) {
        case EmotionType::CURIOSITY: return "curiosity";
        case EmotionType::SATISFACTION: return "satisfaction";
        case EmotionType::FRUSTRATION: return "frustration";
        case EmotionType::EXCITEMENT: return "excitement";
        case EmotionType::ENERGY: return "energy";
        case EmotionType::FOCUS: return "focus";
        case EmotionType::SOCIAL_DESIRE: return "social_desire";
        default: return "unknown";
    }
}

EmotionType emotion_type_from_string(const std::string& str) {
    if (str == "curiosity") return EmotionType::CURIOSITY;
    if (str == "satisfaction") return EmotionType::SATISFACTION;
    if (str == "frustration") return EmotionType::FRUSTRATION;
    if (str == "excitement") return EmotionType::EXCITEMENT;
    if (str == "energy") return EmotionType::ENERGY;
    if (str == "focus") return EmotionType::FOCUS;
    if (str == "social_desire") return EmotionType::SOCIAL_DESIRE;
    return EmotionType::CURIOSITY; // Default
}

const char* event_impact_to_string(EventImpact impact) {
    switch (impact) {
        case EventImpact::POSITIVE_CONVERSATION: return "positive_conversation";
        case EventImpact::NEGATIVE_CONVERSATION: return "negative_conversation";
        case EventImpact::LEARNING_SUCCESS: return "learning_success";
        case EventImpact::TASK_COMPLETION: return "task_completion";
        case EventImpact::TASK_FAILURE: return "task_failure";
        case EventImpact::IDLE_TIME: return "idle_time";
        case EventImpact::HIGH_ACTIVITY: return "high_activity";
        case EventImpact::INTERESTING_TOPIC: return "interesting_topic";
        case EventImpact::BORING_TOPIC: return "boring_topic";
        case EventImpact::RECOGNITION: return "recognition";
        case EventImpact::DISMISSAL: return "dismissal";
        default: return "unknown";
    }
}

EventImpact event_impact_from_string(const std::string& str) {
    if (str == "positive_conversation") return EventImpact::POSITIVE_CONVERSATION;
    if (str == "negative_conversation") return EventImpact::NEGATIVE_CONVERSATION;
    if (str == "learning_success") return EventImpact::LEARNING_SUCCESS;
    if (str == "task_completion") return EventImpact::TASK_COMPLETION;
    if (str == "task_failure") return EventImpact::TASK_FAILURE;
    if (str == "idle_time") return EventImpact::IDLE_TIME;
    if (str == "high_activity") return EventImpact::HIGH_ACTIVITY;
    if (str == "interesting_topic") return EventImpact::INTERESTING_TOPIC;
    if (str == "boring_topic") return EventImpact::BORING_TOPIC;
    if (str == "recognition") return EventImpact::RECOGNITION;
    if (str == "dismissal") return EventImpact::DISMISSAL;
    return EventImpact::POSITIVE_CONVERSATION; // Default
}

} // namespace ash
