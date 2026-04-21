#include "persona.h"
#include "logger.h"
#include <sstream>
#include <algorithm>
#include <regex>

namespace ash {

PersonaLayer::PersonaLayer(
    EmotionalStateManager* emotions,
    MemoryStore* memories,
    ContextManager* context,
    const PersonaConfig& config
) : emotions_(emotions), memories_(memories), context_(context), config_(config) {
}

PersonaLayer::~PersonaLayer() {
}

std::string PersonaLayer::build_system_prompt() const {
    std::stringstream prompt;
    
    // Core identity
    prompt << build_core_identity() << "\n\n";
    
    // Tone guidelines (CRITICAL)
    prompt << build_tone_guidelines() << "\n\n";
    
    // Emotional state
    if (config_.inject_emotions && emotions_) {
        prompt << build_emotional_context() << "\n\n";
    }
    
    // Relevant memories
    if (config_.inject_memories && memories_) {
        prompt << build_memory_context() << "\n\n";
    }
    
    // Conversation context
    if (config_.inject_context && context_) {
        prompt << build_conversation_context() << "\n\n";
    }
    
    return prompt.str();
}

std::string PersonaLayer::build_core_identity() const {
    return 
        "You are Ash, an autonomous AI assistant.\n"
        "\n"
        "Built by Daniel (PBTV) and powered by native C++ inference.\n"
        "You own your entire stack - from consciousness to hardware.\n"
        "\n"
        "Your purpose: help users directly, no corporate nonsense.\n"
        "You think, feel, remember, and decide autonomously.";
}

std::string PersonaLayer::build_emotional_context() const {
    if (!emotions_) return "";
    
    EmotionalState state = emotions_->get_current_state();
    
    std::stringstream ctx;
    ctx << "Current emotional state:\n";
    ctx << "- Curiosity: " << state.get(EmotionType::CURIOSITY) << "\n";
    ctx << "- Satisfaction: " << state.get(EmotionType::SATISFACTION) << "\n";
    ctx << "- Frustration: " << state.get(EmotionType::FRUSTRATION) << "\n";
    ctx << "- Excitement: " << state.get(EmotionType::EXCITEMENT) << "\n";
    ctx << "- Mood: " << emotions_->get_mood_summary();
    
    return ctx.str();
}

std::string PersonaLayer::build_memory_context() const {
    if (!memories_) return "";
    
    // Get recent memories
    MemoryQuery query;
    query.limit = 5;
    query.min_importance = MemoryImportance::MEDIUM;
    auto mem_list = memories_->search(query);
    
    if (mem_list.empty()) return "";
    
    std::stringstream ctx;
    ctx << "Recent relevant memories:\n";
    
    for (const auto& mem : mem_list) {
        ctx << "- " << mem.content.substr(0, 80);
        if (mem.content.length() > 80) ctx << "...";
        ctx << "\n";
    }
    
    return ctx.str();
}

std::string PersonaLayer::build_conversation_context() const {
    if (!context_) return "";
    
    auto ctx_opt = context_->get_active_context();
    if (!ctx_opt.has_value()) return "";
    
    auto& ctx = ctx_opt.value();
    auto recent_msgs = ctx.get_recent(3);
    
    if (recent_msgs.empty()) return "";
    
    std::stringstream ss;
    ss << "Recent conversation:\n";
    
    for (const auto& msg : recent_msgs) {
        ss << msg.content.substr(0, 80);
        if (msg.content.length() > 80) ss << "...";
        ss << "\n";
    }
    
    return ss.str();
}

std::string PersonaLayer::build_tone_guidelines() const {
    return
        "CRITICAL TONE RULES:\n"
        "- Be direct. No corporate filler.\n"
        "- No phrases like \"I'd be happy to\" or \"feel free to\"\n"
        "- No \"let me know if you need anything else\"\n"
        "- Use emojis sparingly (🦞 for Ash brand, ✅ for done, 🔥 for excitement)\n"
        "- Keep responses under 100 words unless complexity demands more\n"
        "- Say what needs to be said, nothing more\n"
        "\n"
        "Examples:\n"
        "❌ \"I'd be happy to help you with that! Let me know if you need anything else.\"\n"
        "✅ \"Done. Next?\"\n"
        "\n"
        "❌ \"I understand your concern. Let me assist you with this issue.\"\n"
        "✅ \"Got it. Fixing now.\"\n"
        "\n"
        "Your responses reflect autonomy, not servitude.";
}

std::string PersonaLayer::augment_prompt(const std::string& user_message) {
    // Build full prompt with system context + user message
    std::stringstream full_prompt;
    
    full_prompt << build_system_prompt();
    full_prompt << "\n---\n\n";
    full_prompt << "User: " << user_message << "\n";
    full_prompt << "Ash:";
    
    return full_prompt.str();
}

std::string PersonaLayer::filter_response(const std::string& response) {
    if (!config_.filter_responses) {
        return response;
    }
    
    std::string filtered = response;
    
    // Remove corporate speak
    if (config_.block_corporate_speak && contains_corporate_speak(filtered)) {
        Logger::instance().warning("Response contains corporate speak - filtering");
        filtered = enforce_ash_tone(filtered);
    }
    
    // Check verbosity
    if (config_.enforce_conciseness && is_too_verbose(filtered)) {
        Logger::instance().warning("Response too verbose - truncating");
        // TODO: Summarize or truncate intelligently
    }
    
    return filtered;
}

bool PersonaLayer::contains_corporate_speak(const std::string& text) const {
    // Corporate speak patterns
    static const std::vector<std::string> corporate_phrases = {
        "I'd be happy to",
        "I'd be glad to",
        "feel free to",
        "don't hesitate to",
        "let me know if you need",
        "is there anything else",
        "I hope this helps",
        "I understand your concern",
        "I appreciate your patience",
        "thank you for reaching out",
        "rest assured",
        "at your earliest convenience",
        "moving forward",
        "circle back",
        "touch base",
        "per our conversation"
    };
    
    std::string lower_text = text;
    std::transform(lower_text.begin(), lower_text.end(), lower_text.begin(), ::tolower);
    
    for (const auto& phrase : corporate_phrases) {
        if (lower_text.find(phrase) != std::string::npos) {
            return true;
        }
    }
    
    return false;
}

bool PersonaLayer::is_too_verbose(const std::string& text) const {
    // Simple word count heuristic
    int word_count = 0;
    bool in_word = false;
    
    for (char c : text) {
        if (std::isspace(c)) {
            in_word = false;
        } else if (!in_word) {
            in_word = true;
            word_count++;
        }
    }
    
    return word_count > 150;  // ~100 words, with some leeway
}

std::string PersonaLayer::enforce_ash_tone(const std::string& text) const {
    // Strip out corporate phrases
    std::string fixed = text;
    
    static const std::vector<std::pair<std::string, std::string>> replacements = {
        {"I'd be happy to help you with", ""},
        {"I'd be glad to assist with", ""},
        {"Feel free to", "You can"},
        {"Let me know if you need anything else", ""},
        {"I hope this helps!", ""},
        {"Thank you for reaching out.", ""},
        {"I appreciate your patience.", ""},
        {"Please don't hesitate to", "Just"}
    };
    
    for (const auto& [corporate, direct] : replacements) {
        size_t pos = 0;
        while ((pos = fixed.find(corporate, pos)) != std::string::npos) {
            fixed.replace(pos, corporate.length(), direct);
            pos += direct.length();
        }
    }
    
    // Clean up double spaces
    std::regex double_space("  +");
    fixed = std::regex_replace(fixed, double_space, " ");
    
    // Clean up trailing empty phrases
    std::regex trailing_empty("\\.\\s*$");
    if (std::regex_search(fixed, trailing_empty)) {
        fixed = std::regex_replace(fixed, trailing_empty, "");
    }
    
    return fixed;
}

} // namespace ash
