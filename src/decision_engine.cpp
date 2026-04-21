#include "decision_engine.h"
#include "emotional_state.h"
#include "memory_store.h"
#include "logger.h"
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <cmath>

namespace ash {

// Internal implementation
class DecisionEngine::Impl {
public:
    std::shared_ptr<EmotionalStateManager> emotions_;
    std::shared_ptr<MemoryStore> memories_;
    DecisionConfig config_;
    
    // Social context tracking: channel_id → list of recent message timestamps
    std::unordered_map<std::string, std::vector<std::chrono::system_clock::time_point>> channel_activity_;
    
    Impl(
        std::shared_ptr<EmotionalStateManager> emotions,
        std::shared_ptr<MemoryStore> memories,
        const DecisionConfig& config
    ) : emotions_(emotions), memories_(memories), config_(config) {}
    
    // Calculate score for SPEAK decision
    float score_speak_decision(
        DecisionTrigger trigger,
        const std::string& context,
        const EmotionalState& state
    ) const {
        float score = 0.5f; // Base: neutral
        
        // Emotional factors
        score += state.curiosity * 0.3f;        // Curious → want to respond
        score += state.excitement * 0.2f;       // Excited → more talkative
        score += state.satisfaction * 0.1f;     // Satisfied → engaged
        score -= state.frustration * 0.3f;      // Frustrated → less engaged
        score -= (1.0f - state.energy) * 0.2f;  // Low energy → less likely
        
        // Trigger-based adjustments
        if (trigger == DecisionTrigger::MESSAGE_RECEIVED) {
            score += 0.2f; // Direct message → more likely to respond
        }
        if (trigger == DecisionTrigger::EMOTIONAL_SHIFT && state.curiosity > 0.7f) {
            score += 0.15f; // Emotional shift with high curiosity → want to engage
        }
        
        // Context relevance (basic keyword check)
        if (!context.empty()) {
            // Check if context mentions "ash", "help", "question" - implies direct address
            std::string lower_context = context;
            std::transform(lower_context.begin(), lower_context.end(), lower_context.begin(), ::tolower);
            
            if (lower_context.find("ash") != std::string::npos ||
                lower_context.find("help") != std::string::npos ||
                lower_context.find("?") != std::string::npos) {
                score += 0.25f; // Direct address or question → much more likely
            }
        }
        
        return std::clamp(score, 0.0f, 1.0f);
    }
    
    // Calculate score for INITIATE decision
    float score_initiate_decision(
        const EmotionalState& state,
        float time_since_last_message
    ) const {
        float score = 0.0f; // Base: don't initiate (requires strong motivation)
        
        // Emotional factors (stronger thresholds than SPEAK)
        if (state.excitement > config_.excitement_threshold) {
            score += 0.4f; // High excitement → want to share
        }
        if (state.curiosity > config_.curiosity_threshold && state.social_desire > 0.6f) {
            score += 0.3f; // Curious + social → want to engage
        }
        if (state.energy > config_.high_energy_threshold) {
            score += 0.2f; // High energy → more likely to be active
        }
        
        // Time factor: longer silence → more likely to initiate (but only if emotionally motivated)
        if (time_since_last_message > config_.initiate_conversation_after) {
            float time_factor = std::min(
                (time_since_last_message - config_.initiate_conversation_after) / 3600.0f,
                0.3f
            );
            score += time_factor;
        }
        
        // Low energy or high frustration → don't initiate
        if (state.energy < config_.low_energy_threshold) {
            score -= 0.5f;
        }
        if (state.frustration > config_.frustration_threshold) {
            score -= 0.4f;
        }
        
        return std::clamp(score, 0.0f, 1.0f);
    }
    
    // Calculate score for THINK decision (internal processing)
    float score_think_decision(
        const EmotionalState& state
    ) const {
        float score = 0.3f; // Base: some background thinking
        
        // High focus or satisfaction → good time to consolidate memories
        if (state.focus > 0.6f) {
            score += 0.2f;
        }
        if (state.satisfaction > 0.7f) {
            score += 0.15f;
        }
        
        // Low energy → prefer thinking over active engagement
        if (state.energy < config_.low_energy_threshold) {
            score += 0.25f;
        }
        
        return std::clamp(score, 0.0f, 1.0f);
    }
    
    // Calculate score for IDLE decision
    float score_idle_decision(
        const EmotionalState& state
    ) const {
        float score = 0.2f; // Base: low (Ash is generally active)
        
        // Very low energy → need rest
        if (state.energy < 0.2f) {
            score += 0.6f;
        }
        
        // High frustration + low satisfaction → need break
        if (state.frustration > 0.7f && state.satisfaction < 0.3f) {
            score += 0.3f;
        }
        
        // Low curiosity + low excitement → nothing interesting happening
        if (state.curiosity < 0.3f && state.excitement < 0.3f) {
            score += 0.2f;
        }
        
        return std::clamp(score, 0.0f, 1.0f);
    }
    
    // Calculate score for RECALL decision (search memories)
    float score_recall_decision(
        DecisionTrigger trigger,
        const std::string& context,
        const EmotionalState& state
    ) const {
        float score = 0.3f; // Base: moderate (memories are useful)
        
        // High curiosity → want to recall related memories
        if (state.curiosity > 0.6f) {
            score += 0.2f;
        }
        
        // Context mentions names, projects, or past events → recall relevant
        if (!context.empty()) {
            std::string lower_context = context;
            std::transform(lower_context.begin(), lower_context.end(), lower_context.begin(), ::tolower);
            
            if (lower_context.find("remember") != std::string::npos ||
                lower_context.find("before") != std::string::npos ||
                lower_context.find("last time") != std::string::npos ||
                lower_context.find("daniel") != std::string::npos) {
                score += 0.35f; // Explicit memory reference
            }
        }
        
        // MEMORY_RELEVANCE trigger → definitely recall
        if (trigger == DecisionTrigger::MEMORY_RELEVANCE) {
            score += 0.3f;
        }
        
        return std::clamp(score, 0.0f, 1.0f);
    }
    
    // Build a decision from scores
    Decision build_decision(
        std::vector<std::pair<DecisionType, float>> scored_options,
        const std::string& context
    ) const {
        // Sort by score (highest first)
        std::sort(scored_options.begin(), scored_options.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });
        
        auto [type, confidence] = scored_options[0];
        
        Decision decision;
        decision.type = type;
        decision.confidence = confidence;
        
        // Generate reasoning based on decision type
        switch (type) {
            case DecisionType::SPEAK:
                decision.reasoning = "Emotionally engaged and context is relevant. Ready to respond.";
                break;
            case DecisionType::INITIATE:
                decision.reasoning = "High motivation to engage. Good time to start conversation.";
                break;
            case DecisionType::THINK:
                decision.reasoning = "Good opportunity for internal processing and memory consolidation.";
                break;
            case DecisionType::IDLE:
                decision.reasoning = "Low energy or no strong motivation. Conserving resources.";
                break;
            case DecisionType::RECALL:
                decision.reasoning = "Context suggests relevant memories. Should search for context.";
                if (!context.empty()) {
                    decision.memory_query = context; // Use context as initial query
                }
                break;
        }
        
        // Log decision
        const char* type_str = "";
        switch (type) {
            case DecisionType::SPEAK: type_str = "SPEAK"; break;
            case DecisionType::INITIATE: type_str = "INITIATE"; break;
            case DecisionType::THINK: type_str = "THINK"; break;
            case DecisionType::IDLE: type_str = "IDLE"; break;
            case DecisionType::RECALL: type_str = "RECALL"; break;
        }
        
        Logger::debug("Decision: {} (confidence: {:.2f}) - {}", 
            type_str, confidence, decision.reasoning);
        
        return decision;
    }
};

// Constructor
DecisionEngine::DecisionEngine(
    std::shared_ptr<EmotionalStateManager> emotions,
    std::shared_ptr<MemoryStore> memories,
    const DecisionConfig& config
) : impl_(std::make_unique<Impl>(emotions, memories, config)) {
    Logger::info("🧠 Decision engine initialized");
}

// Destructor
DecisionEngine::~DecisionEngine() = default;

// Make a decision
Decision DecisionEngine::decide(
    DecisionTrigger trigger,
    const std::string& context
) {
    auto state = impl_->emotions_->get_state();
    
    // Score all decision options
    std::vector<std::pair<DecisionType, float>> scores;
    
    scores.push_back({DecisionType::SPEAK, impl_->score_speak_decision(trigger, context, state)});
    scores.push_back({DecisionType::INITIATE, impl_->score_initiate_decision(state, 0.0f)}); // TODO: time tracking
    scores.push_back({DecisionType::THINK, impl_->score_think_decision(state)});
    scores.push_back({DecisionType::IDLE, impl_->score_idle_decision(state)});
    scores.push_back({DecisionType::RECALL, impl_->score_recall_decision(trigger, context, state)});
    
    return impl_->build_decision(scores, context);
}

// Should Ash speak?
bool DecisionEngine::should_speak(
    const std::chrono::system_clock::time_point& last_message_time,
    const std::string& channel_id
) const {
    auto now = std::chrono::system_clock::now();
    auto time_since = std::chrono::duration_cast<std::chrono::seconds>(now - last_message_time).count();
    
    auto state = impl_->emotions_->get_state();
    
    // Quick response if conversation is active and energy is good
    if (time_since < impl_->config_.respond_quickly_under && state.energy > 0.5f) {
        return true;
    }
    
    // Check emotional state
    if (state.frustration > impl_->config_.frustration_threshold) {
        return false; // Too frustrated to engage
    }
    
    if (state.energy < impl_->config_.low_energy_threshold) {
        return false; // Too tired
    }
    
    // High curiosity or excitement → likely to speak
    if (state.curiosity > impl_->config_.curiosity_threshold ||
        state.excitement > impl_->config_.excitement_threshold) {
        return true;
    }
    
    // Default: moderate likelihood (let decide() make final call)
    return state.energy > 0.4f && state.curiosity > 0.4f;
}

// Should Ash initiate conversation?
bool DecisionEngine::should_initiate_conversation(
    const std::chrono::system_clock::time_point& last_message_time,
    const std::string& channel_id
) const {
    auto now = std::chrono::system_clock::now();
    auto time_since = std::chrono::duration_cast<std::chrono::seconds>(now - last_message_time).count();
    
    // Don't initiate if recent conversation
    if (time_since < impl_->config_.initiate_conversation_after) {
        return false;
    }
    
    auto state = impl_->emotions_->get_state();
    
    // Need high energy and motivation to initiate
    if (state.energy < impl_->config_.high_energy_threshold) {
        return false;
    }
    
    // Need strong emotional motivation
    bool high_excitement = state.excitement > impl_->config_.excitement_threshold;
    bool curious_and_social = state.curiosity > impl_->config_.curiosity_threshold && 
                             state.social_desire > 0.6f;
    
    return high_excitement || curious_and_social;
}

// Calculate confidence
float DecisionEngine::calculate_confidence(
    DecisionType type,
    DecisionTrigger trigger,
    const std::string& context
) const {
    auto state = impl_->emotions_->get_state();
    
    switch (type) {
        case DecisionType::SPEAK:
            return impl_->score_speak_decision(trigger, context, state);
        case DecisionType::INITIATE:
            return impl_->score_initiate_decision(state, 0.0f); // TODO: time tracking
        case DecisionType::THINK:
            return impl_->score_think_decision(state);
        case DecisionType::IDLE:
            return impl_->score_idle_decision(state);
        case DecisionType::RECALL:
            return impl_->score_recall_decision(trigger, context, state);
        default:
            return 0.0f;
    }
}

// Record message activity
void DecisionEngine::record_message(
    const std::string& channel_id,
    const std::chrono::system_clock::time_point& timestamp
) {
    auto& activity = impl_->channel_activity_[channel_id];
    activity.push_back(timestamp);
    
    // Prune old messages (keep only recent window)
    auto cutoff = timestamp - std::chrono::seconds(impl_->config_.active_conversation_window);
    activity.erase(
        std::remove_if(activity.begin(), activity.end(),
            [cutoff](const auto& t) { return t < cutoff; }),
        activity.end()
    );
}

// Is conversation active?
bool DecisionEngine::is_conversation_active(const std::string& channel_id) const {
    auto it = impl_->channel_activity_.find(channel_id);
    if (it == impl_->channel_activity_.end()) {
        return false;
    }
    
    auto now = std::chrono::system_clock::now();
    auto cutoff = now - std::chrono::seconds(impl_->config_.active_conversation_window);
    
    int recent_messages = std::count_if(it->second.begin(), it->second.end(),
        [cutoff](const auto& t) { return t >= cutoff; });
    
    return recent_messages >= impl_->config_.min_messages_for_active;
}

// Get config
const DecisionConfig& DecisionEngine::get_config() const {
    return impl_->config_;
}

// Update config
void DecisionEngine::update_config(const DecisionConfig& config) {
    impl_->config_ = config;
    Logger::info("Decision config updated");
}

} // namespace ash
