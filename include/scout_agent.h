#pragma once

#include "model_loader.h"
#include "inference.h"
#include <string>
#include <vector>
#include <map>
#include <regex>

namespace ash {

// Scout Agent - Query classification and routing for ash-forge
// Uses ash-core model to route queries to appropriate specialists
class ScoutAgent {
public:
    ScoutAgent(ModelRegistry* registry);
    ~ScoutAgent();
    
    // Initialize scout (loads ash-core router model)
    bool initialize(const std::string& core_model_name = "ash-core");
    
    // Check if initialized
    bool is_initialized() const { return initialized_; }
    
    // Main routing entry point
    std::string process_query(const std::string& query);
    
    // Classify query into categories (public for testing)
    std::vector<std::string> classify_query(const std::string& query);
    
private:
    ModelRegistry* registry_;
    std::string core_model_name_;
    bool initialized_ = false;
    
    // Pattern matching rules for fast classification
    struct CategoryPattern {
        std::string category;
        std::regex pattern;
        int priority;  // Higher = checked first
    };
    std::vector<CategoryPattern> patterns_;
    
    // Initialize pattern matching rules
    void init_patterns();
    
    // Classification methods
    std::vector<std::string> classify_with_patterns(const std::string& query);
    std::vector<std::string> classify_with_model(const std::string& query);  // Future: use ash-core
    
    // Routing strategies
    std::string route_single(const std::string& query, const std::string& specialist);
    std::string route_multiple(const std::string& query, const std::vector<std::string>& specialists);
    std::string route_to_chat(const std::string& query);  // Fallback to ash-chat
    
    // Response synthesis
    std::string synthesize_responses(
        const std::string& query,
        const std::map<std::string, std::string>& responses,
        const std::vector<std::string>& categories
    );
    
    // Conflict detection and arbitration
    bool has_conflict(const std::map<std::string, std::string>& responses);
    std::string arbitrate(
        const std::string& query,
        const std::map<std::string, std::string>& responses,
        const std::vector<std::string>& categories
    );
    
    // Helper to map category to model name
    std::string category_to_model(const std::string& category);
};

} // namespace ash
