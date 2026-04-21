#pragma once

#include <string>
#include <memory>
#include <chrono>
#include <vector>
#include <optional>

namespace ash {

// Forward declarations
class MemoryStore;
class EmotionalStateManager;

// Context types
enum class ContextType {
    DISCORD_CHANNEL,    // Discord conversation in a channel
    DISCORD_DM,         // Direct message conversation
    INTERNAL_THOUGHT,   // Ash's internal processing/reflection
    PLANNING,           // Active planning or problem-solving
    MEMORY_RECALL       // Searching/consolidating memories
};

// Message in a context
struct ContextMessage {
    std::string author;                                     // Who sent it ("ash" for Ash's messages)
    std::string content;                                    // Message content
    std::chrono::system_clock::time_point timestamp;        // When
    std::optional<std::string> message_id;                  // Platform-specific ID
    
    // Helper to format for display
    std::string to_string() const;
};

// A context represents a single conversation or thought process
struct Context {
    std::string id;                                         // Unique context ID
    ContextType type;                                       // What kind of context
    std::string name;                                       // Display name (e.g., "ash-chat", "internal-reflection")
    
    // Conversation history (working memory - last N messages)
    std::vector<ContextMessage> messages;
    int max_messages = 50;                                  // Max messages to keep in working memory
    
    // Timing
    std::chrono::system_clock::time_point created;
    std::chrono::system_clock::time_point last_activity;
    
    // Metadata
    std::vector<std::string> participants;                  // Who's in this conversation
    std::optional<std::string> platform_channel_id;         // Discord channel ID, etc.
    
    // State
    bool is_active = true;                                  // Is this context currently active?
    int priority = 0;                                       // Higher priority = more attention
    
    // Add message to context
    void add_message(const ContextMessage& msg);
    
    // Get recent messages (last N)
    std::vector<ContextMessage> get_recent(int count = 10) const;
    
    // Get all messages as formatted string
    std::string get_history_text() const;
    
    // Serialization
    std::string to_json() const;
    static Context from_json(const std::string& json);
};

// Configuration for context management
struct ContextConfig {
    int max_active_contexts = 10;                          // Max simultaneous active contexts
    int max_working_memory_messages = 50;                  // Max messages per context
    float inactive_context_timeout = 3600.0f;              // Seconds before context considered inactive
    bool auto_archive_inactive = true;                     // Automatically archive old contexts
    int context_switch_delay_ms = 100;                     // Delay when switching contexts
};

// The context manager - tracks multiple simultaneous conversations
class ContextManager {
public:
    explicit ContextManager(
        std::shared_ptr<MemoryStore> memories,
        const ContextConfig& config = ContextConfig{}
    );
    ~ContextManager();
    
    // Create a new context
    std::string create_context(
        ContextType type,
        const std::string& name,
        const std::vector<std::string>& participants = {}
    );
    
    // Get context by ID
    std::optional<Context> get_context(const std::string& context_id) const;
    
    // Add message to context
    bool add_message(
        const std::string& context_id,
        const ContextMessage& message
    );
    
    // Switch active context
    bool switch_to_context(const std::string& context_id);
    
    // Get current active context
    std::optional<Context> get_active_context() const;
    
    // List all contexts (optionally filter by type or active status)
    std::vector<Context> list_contexts(
        std::optional<ContextType> type_filter = std::nullopt,
        std::optional<bool> active_only = std::nullopt
    ) const;
    
    // Archive a context (move to long-term memory, mark inactive)
    bool archive_context(const std::string& context_id);
    
    // Restore archived context
    bool restore_context(const std::string& context_id);
    
    // Update context priority (higher = more attention)
    void set_priority(const std::string& context_id, int priority);
    
    // Check if context is active
    bool is_active(const std::string& context_id) const;
    
    // Get context by platform channel ID (e.g., Discord channel)
    std::optional<std::string> get_context_by_channel(const std::string& channel_id) const;
    
    // Automatic maintenance (archive old contexts, clean up)
    void maintenance();
    
    // Get config
    const ContextConfig& get_config() const;
    
    // Update config
    void update_config(const ContextConfig& config);
    
    // Persistence
    bool save_to_file(const std::string& filepath) const;
    bool load_from_file(const std::string& filepath);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ash
