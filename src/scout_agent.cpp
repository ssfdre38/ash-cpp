#include "scout_agent.h"
#include "logger.h"
#include <algorithm>
#include <sstream>

namespace ash {

ScoutAgent::ScoutAgent(ModelRegistry* registry) 
    : registry_(registry) {
    init_patterns();
}

ScoutAgent::~ScoutAgent() {
}

void ScoutAgent::init_patterns() {
    // Initialize regex patterns for fast classification
    // Priority: higher values are checked first
    
    patterns_ = {
        // Python patterns (priority 90)
        {"python", std::regex(R"(\b(python|pip|django|flask|pandas|numpy|pytest)\b)", 
                              std::regex::icase), 90},
        {"python", std::regex(R"(\.py\b)", std::regex::icase), 85},
        {"python", std::regex(R"(\bvirtualenv\b|\bvenv\b)", std::regex::icase), 85},
        
        // JavaScript patterns (priority 90)
        {"javascript", std::regex(R"(\b(javascript|js|node|npm|react|vue|angular)\b)", 
                                  std::regex::icase), 90},
        {"javascript", std::regex(R"(\.js\b|\.ts\b|\.jsx\b|\.tsx\b)", std::regex::icase), 85},
        {"javascript", std::regex(R"(\bpackage\.json\b)", std::regex::icase), 85},
        
        // Debugging patterns (priority 95 - high priority)
        {"debugging", std::regex(R"(\b(error|bug|crash|traceback|exception|debug)\b)", 
                                 std::regex::icase), 95},
        {"debugging", std::regex(R"(\bfailed|failing|broken\b)", std::regex::icase), 90},
        {"debugging", std::regex(R"(\bstack trace\b|\bsegfault\b)", std::regex::icase), 95},
        
        // Systems patterns (priority 80)
        {"systems", std::regex(R"(\b(linux|server|docker|kubernetes|deploy|nginx)\b)", 
                               std::regex::icase), 80},
        {"systems", std::regex(R"(\b(ssh|sudo|systemctl|service)\b)", std::regex::icase), 85},
        {"systems", std::regex(R"(\bDocker|kubectl|k8s\b)", std::regex::icase), 85},
        
        // Creative patterns (priority 70)
        {"creative", std::regex(R"(\b(write|story|poem|creative|imagine)\b)", 
                                std::regex::icase), 70},
        {"creative", std::regex(R"(\bnovel|fiction|narrative\b)", std::regex::icase), 75},
        
        // Chat/conversational patterns (priority 60 - lowest, fallback)
        {"chat", std::regex(R"(\b(how are you|tell me about|what do you think)\b)", 
                           std::regex::icase), 60},
        {"chat", std::regex(R"(^(hi|hello|hey)\b)", std::regex::icase), 60},
    };
    
    // Sort by priority (highest first)
    std::sort(patterns_.begin(), patterns_.end(), 
        [](const CategoryPattern& a, const CategoryPattern& b) {
            return a.priority > b.priority;
        });
    
    Logger::instance().info("Scout agent initialized with " + 
        std::to_string(patterns_.size()) + " classification patterns");
}

bool ScoutAgent::initialize(const std::string& core_model_name) {
    core_model_name_ = core_model_name;
    
    // Check if ash-core is loaded
    if (!registry_->is_loaded(core_model_name_)) {
        Logger::instance().info("Scout agent: ash-core not loaded, will route without model inference");
        // Note: We can still route using pattern matching
    } else {
        Logger::instance().info("Scout agent: ash-core loaded, advanced routing available");
    }
    
    initialized_ = true;
    return true;
}

std::vector<std::string> ScoutAgent::classify_query(const std::string& query) {
    // Use pattern matching for classification
    auto categories = classify_with_patterns(query);
    
    // Future: If pattern matching is uncertain, use ash-core model inference
    // if (categories.empty() && registry_->is_loaded(core_model_name_)) {
    //     categories = classify_with_model(query);
    // }
    
    // Default to chat if no categories found
    if (categories.empty()) {
        categories.push_back("chat");
    }
    
    return categories;
}

std::vector<std::string> ScoutAgent::classify_with_patterns(const std::string& query) {
    std::vector<std::string> categories;
    std::map<std::string, int> category_scores;
    
    // Check all patterns
    for (const auto& pattern : patterns_) {
        if (std::regex_search(query, pattern.pattern)) {
            category_scores[pattern.category] += pattern.priority;
        }
    }
    
    // Get categories with significant scores
    int max_score = 0;
    for (const auto& [category, score] : category_scores) {
        if (score > max_score) max_score = score;
    }
    
    // Include categories with score >= 50% of max
    int threshold = max_score / 2;
    for (const auto& [category, score] : category_scores) {
        if (score >= threshold && score >= 60) {  // Minimum score of 60
            categories.push_back(category);
        }
    }
    
    // Limit to top 3 categories
    if (categories.size() > 3) {
        categories.resize(3);
    }
    
    return categories;
}

std::vector<std::string> ScoutAgent::classify_with_model(const std::string& query) {
    // TODO: Use ash-core model for classification when available
    // This would involve:
    // 1. Load ash-core model
    // 2. Format prompt: "Classify this query into categories: {query}\nCategories:"
    // 3. Parse output to extract categories
    
    Logger::instance().debug("Model-based classification not yet implemented");
    return {};
}

std::string ScoutAgent::process_query(const std::string& query) {
    if (!initialized_) {
        Logger::instance().error("Scout agent not initialized");
        return "Error: Scout agent not initialized";
    }
    
    Logger::instance().info("Scout processing query: " + query.substr(0, 50) + "...");
    
    // Classify query
    auto categories = classify_query(query);
    
    Logger::instance().info("Classified as: [" + 
        [&categories]() {
            std::string result;
            for (size_t i = 0; i < categories.size(); ++i) {
                if (i > 0) result += ", ";
                result += categories[i];
            }
            return result;
        }() + "]");
    
    // Route based on number of categories
    if (categories.size() == 1) {
        // Single specialist
        return route_single(query, categories[0]);
    } else if (categories.size() <= 3) {
        // Multiple specialists
        return route_multiple(query, categories);
    } else {
        // Too complex, use chat
        return route_to_chat(query);
    }
}

std::string ScoutAgent::category_to_model(const std::string& category) {
    // Map category to model name
    if (category == "python") return "ash-python";
    if (category == "javascript") return "ash-javascript";
    if (category == "debugging") return "ash-debugging";
    if (category == "systems") return "ash-systems";
    if (category == "creative") return "ash-creative";
    if (category == "chat") return "ash-chat";
    
    // Default to chat for unknown categories
    return "ash-chat";
}

std::string ScoutAgent::route_single(const std::string& query, const std::string& specialist) {
    std::string model_name = category_to_model(specialist);
    
    // Check if specialist is loaded
    if (!registry_->is_loaded(model_name)) {
        Logger::instance().warning("Specialist not loaded: " + model_name + ", routing to ash-chat");
        return route_to_chat(query);
    }
    
    Logger::instance().info("Routing to specialist: " + model_name);
    registry_->mark_model_used(model_name);
    
    // TODO: Actually run inference on the specialist
    // For now, return a placeholder
    return "[" + model_name + " would respond here]\n\nQuery: " + query;
}

std::string ScoutAgent::route_multiple(const std::string& query, 
                                       const std::vector<std::string>& specialists) {
    std::map<std::string, std::string> responses;
    
    // Query each specialist
    for (const auto& specialist : specialists) {
        std::string model_name = category_to_model(specialist);
        
        if (!registry_->is_loaded(model_name)) {
            Logger::instance().warning("Specialist not loaded: " + model_name);
            continue;
        }
        
        Logger::instance().info("Querying specialist: " + model_name);
        registry_->mark_model_used(model_name);
        
        // TODO: Actually run inference
        responses[specialist] = "[" + model_name + " response]";
    }
    
    // If no specialists could respond, fallback to chat
    if (responses.empty()) {
        Logger::instance().warning("No specialists available, routing to ash-chat");
        return route_to_chat(query);
    }
    
    // Synthesize responses
    return synthesize_responses(query, responses, specialists);
}

std::string ScoutAgent::route_to_chat(const std::string& query) {
    std::string model_name = "ash-chat";
    
    if (!registry_->is_loaded(model_name)) {
        Logger::instance().error("ash-chat not loaded, cannot respond");
        return "Error: No models available to respond to query";
    }
    
    Logger::instance().info("Routing to ash-chat (conversational/complex)");
    registry_->mark_model_used(model_name);
    
    // TODO: Actually run inference with ash-chat
    return "[ash-chat would respond here]\n\nQuery: " + query;
}

std::string ScoutAgent::synthesize_responses(
    const std::string& query,
    const std::map<std::string, std::string>& responses,
    const std::vector<std::string>& categories) {
    
    // Check for conflicts
    if (has_conflict(responses)) {
        Logger::instance().info("Conflict detected in specialist responses, arbitrating...");
        return arbitrate(query, responses, categories);
    }
    
    // No conflicts, combine responses
    std::ostringstream result;
    result << "Multi-specialist response:\n\n";
    
    for (const auto& [category, response] : responses) {
        result << "**" << category << ":**\n";
        result << response << "\n\n";
    }
    
    return result.str();
}

bool ScoutAgent::has_conflict(const std::map<std::string, std::string>& responses) {
    // TODO: Implement conflict detection
    // For now, assume no conflicts
    return false;
}

std::string ScoutAgent::arbitrate(
    const std::string& query,
    const std::map<std::string, std::string>& responses,
    const std::vector<std::string>& categories) {
    
    // Primary domain is first category (highest priority)
    std::string primary = categories[0];
    
    std::ostringstream result;
    result << "Based on your question, here's what I found:\n\n";
    
    // Present primary response first
    if (responses.count(primary)) {
        result << "**Primary answer (" << primary << "):**\n";
        result << responses.at(primary) << "\n\n";
    }
    
    // Present other perspectives
    for (const auto& [category, response] : responses) {
        if (category != primary) {
            result << "**Also relevant (" << category << "):**\n";
            result << response << "\n\n";
        }
    }
    
    return result.str();
}

} // namespace ash
