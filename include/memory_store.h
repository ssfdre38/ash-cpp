#pragma once

#include <string>
#include <vector>
#include <memory>
#include <chrono>
#include <optional>
#include <unordered_map>

namespace ash {

// Memory types
enum class MemoryType {
    EPISODIC,   // Events, conversations, experiences (what happened)
    SEMANTIC,   // Facts, knowledge, concepts (what I know)
    EMOTIONAL   // Feelings during events (how I felt)
};

// Memory importance (affects retrieval priority)
enum class MemoryImportance {
    LOW = 0,
    MEDIUM = 1,
    HIGH = 2,
    CRITICAL = 3
};

// Memory metadata
struct MemoryMetadata {
    int64_t id = -1;
    MemoryType type;
    MemoryImportance importance;
    std::chrono::system_clock::time_point timestamp;
    std::string source;  // "discord", "internal", "system"
    std::vector<std::string> tags;
    std::unordered_map<std::string, std::string> extra;  // Flexible metadata
};

// Memory content
struct Memory {
    MemoryMetadata metadata;
    std::string content;
    
    // Optional fields
    std::optional<float> emotional_valence;  // -1.0 (negative) to 1.0 (positive)
    std::optional<std::string> related_to;   // Link to other memory IDs
    std::optional<int> access_count;         // How many times retrieved
    std::optional<std::chrono::system_clock::time_point> last_accessed;
    
    // Helper for JSON serialization
    std::string to_json() const;
    static Memory from_json(const std::string& json);
};

// Search query parameters
struct MemoryQuery {
    std::optional<MemoryType> type;
    std::optional<MemoryImportance> min_importance;
    std::optional<std::string> content_contains;
    std::optional<std::vector<std::string>> tags;
    std::optional<std::chrono::system_clock::time_point> after;
    std::optional<std::chrono::system_clock::time_point> before;
    std::optional<std::string> source;
    
    size_t limit = 10;  // Max results to return
    bool order_by_recency = true;  // true = newest first, false = oldest first
};

// Memory store interface
class MemoryStore {
public:
    MemoryStore();
    explicit MemoryStore(const std::string& db_path);
    ~MemoryStore();
    
    // Initialize database (creates tables if needed)
    bool initialize();
    
    // Store new memory
    int64_t store(const Memory& memory);
    
    // Retrieve memory by ID
    std::optional<Memory> get(int64_t id);
    
    // Search memories
    std::vector<Memory> search(const MemoryQuery& query);
    
    // Update existing memory
    bool update(int64_t id, const Memory& memory);
    
    // Delete memory
    bool remove(int64_t id);
    
    // Statistics
    size_t count_all() const;
    size_t count_by_type(MemoryType type) const;
    
    // Recent memories
    std::vector<Memory> get_recent(size_t count = 10);
    
    // Important memories
    std::vector<Memory> get_important(MemoryImportance min_importance = MemoryImportance::HIGH);
    
    // Memories by tag
    std::vector<Memory> get_by_tag(const std::string& tag);
    
    // Memory consolidation (summarize old memories)
    void consolidate_old_memories(std::chrono::hours age_threshold = std::chrono::hours(24 * 30));
    
    // Prune low-importance old memories
    void prune_memories(MemoryImportance max_importance = MemoryImportance::LOW,
                       std::chrono::hours age_threshold = std::chrono::hours(24 * 90));
    
    // Track access (increments access_count, updates last_accessed)
    void track_access(int64_t id);
    
    // Get memory statistics
    struct Stats {
        size_t total_memories;
        size_t episodic_count;
        size_t semantic_count;
        size_t emotional_count;
        size_t high_importance;
        size_t critical_importance;
        std::chrono::system_clock::time_point oldest_memory;
        std::chrono::system_clock::time_point newest_memory;
    };
    Stats get_stats() const;
    
    // Vacuum database (reclaim space)
    void vacuum();
    
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    
    // Helper: Execute SQL and return affected rows
    int execute_sql(const std::string& sql);
    
    // Helper: Serialize tags to comma-separated string
    static std::string serialize_tags(const std::vector<std::string>& tags);
    static std::vector<std::string> deserialize_tags(const std::string& tags_str);
};

// Memory builder helper (fluent API)
class MemoryBuilder {
public:
    MemoryBuilder();
    
    MemoryBuilder& episodic();
    MemoryBuilder& semantic();
    MemoryBuilder& emotional();
    
    MemoryBuilder& importance(MemoryImportance imp);
    MemoryBuilder& low_importance();
    MemoryBuilder& medium_importance();
    MemoryBuilder& high_importance();
    MemoryBuilder& critical_importance();
    
    MemoryBuilder& content(const std::string& text);
    MemoryBuilder& source(const std::string& src);
    MemoryBuilder& tag(const std::string& t);
    MemoryBuilder& tags(const std::vector<std::string>& t);
    
    MemoryBuilder& valence(float val);  // -1.0 to 1.0
    MemoryBuilder& related_to(const std::string& memory_id);
    
    MemoryBuilder& metadata(const std::string& key, const std::string& value);
    
    Memory build() const;
    
private:
    Memory memory_;
};

// Enum to string conversions
const char* memory_type_to_string(MemoryType type);
MemoryType memory_type_from_string(const std::string& str);

const char* memory_importance_to_string(MemoryImportance imp);
MemoryImportance memory_importance_from_string(const std::string& str);

} // namespace ash
