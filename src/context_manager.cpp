#include "context_manager.h"
#include "memory_store.h"
#include "logger.h"
#include <unordered_map>
#include <algorithm>
#include <thread>
#include <sstream>
#include <iomanip>
#include <ctime>

namespace ash {

// ContextMessage helpers
std::string ContextMessage::to_string() const {
    auto time_t = std::chrono::system_clock::to_time_t(timestamp);
    std::stringstream ss;
    ss << "[" << std::put_time(std::localtime(&time_t), "%H:%M:%S") << "] ";
    ss << author << ": " << content;
    return ss.str();
}

// Context implementation
void Context::add_message(const ContextMessage& msg) {
    messages.push_back(msg);
    last_activity = msg.timestamp;
    
    // Trim to max_messages
    if (messages.size() > static_cast<size_t>(max_messages)) {
        messages.erase(messages.begin(), messages.begin() + (messages.size() - max_messages));
    }
}

std::vector<ContextMessage> Context::get_recent(int count) const {
    if (count >= static_cast<int>(messages.size())) {
        return messages;
    }
    return std::vector<ContextMessage>(
        messages.end() - count,
        messages.end()
    );
}

std::string Context::get_history_text() const {
    std::stringstream ss;
    for (const auto& msg : messages) {
        ss << msg.to_string() << "\n";
    }
    return ss.str();
}

std::string Context::to_json() const {
    // Basic JSON serialization (simplified - could use a JSON library)
    std::stringstream ss;
    ss << "{";
    ss << "\"id\":\"" << id << "\",";
    ss << "\"type\":" << static_cast<int>(type) << ",";
    ss << "\"name\":\"" << name << "\",";
    ss << "\"is_active\":" << (is_active ? "true" : "false") << ",";
    ss << "\"priority\":" << priority;
    ss << "}";
    return ss.str();
}

Context Context::from_json(const std::string& json) {
    // Simplified parsing - in production would use proper JSON library
    Context ctx;
    // TODO: Implement proper JSON parsing
    return ctx;
}

// Internal implementation
class ContextManager::Impl {
public:
    std::shared_ptr<MemoryStore> memories_;
    ContextConfig config_;
    
    // Active contexts map: context_id → Context
    std::unordered_map<std::string, Context> contexts_;
    
    // Current active context ID
    std::optional<std::string> active_context_id_;
    
    // Channel ID to context ID mapping
    std::unordered_map<std::string, std::string> channel_to_context_;
    
    Impl(std::shared_ptr<MemoryStore> memories, const ContextConfig& config)
        : memories_(memories), config_(config) {}
    
    // Generate unique context ID
    std::string generate_context_id(ContextType type, const std::string& name) {
        static int counter = 0;
        std::stringstream ss;
        ss << "ctx_";
        switch (type) {
            case ContextType::DISCORD_CHANNEL: ss << "discord_"; break;
            case ContextType::DISCORD_DM: ss << "dm_"; break;
            case ContextType::INTERNAL_THOUGHT: ss << "thought_"; break;
            case ContextType::PLANNING: ss << "plan_"; break;
            case ContextType::MEMORY_RECALL: ss << "recall_"; break;
        }
        ss << name << "_" << ++counter;
        return ss.str();
    }
    
    // Archive context to long-term memory
    void archive_to_memory(const Context& ctx) {
        // Store conversation as episodic memory
        std::string conversation_summary = "Conversation in " + ctx.name + " with " + 
            std::to_string(ctx.messages.size()) + " messages";
        
        auto memory = MemoryBuilder()
            .episodic()
            .high_importance()
            .content(conversation_summary)
            .tag("context:" + ctx.id)
            .tag("conversation")
            .source("context_manager")
            .build();
        
        memories_->store(memory);
        
        // Store individual important messages
        for (const auto& msg : ctx.messages) {
            if (msg.author != "ash") { // Store messages from others
                auto msg_memory = MemoryBuilder()
                    .episodic()
                    .medium_importance()
                    .content(msg.author + " said: " + msg.content)
                    .tag("context:" + ctx.id)
                    .tag("message")
                    .source(msg.author)
                    .build();
                
                memories_->store(msg_memory);
            }
        }
        
        Logger::instance().info("Archived context " + ctx.id + " to memory (" + 
            std::to_string(ctx.messages.size()) + " messages)");
    }
};

// Constructor
ContextManager::ContextManager(
    std::shared_ptr<MemoryStore> memories,
    const ContextConfig& config
) : impl_(std::make_unique<Impl>(memories, config)) {
    Logger::instance().info("🗂️ Context manager initialized");
}

// Destructor
ContextManager::~ContextManager() = default;

// Create context
std::string ContextManager::create_context(
    ContextType type,
    const std::string& name,
    const std::vector<std::string>& participants
) {
    Context ctx;
    ctx.id = impl_->generate_context_id(type, name);
    ctx.type = type;
    ctx.name = name;
    ctx.participants = participants;
    ctx.created = std::chrono::system_clock::now();
    ctx.last_activity = ctx.created;
    ctx.is_active = true;
    ctx.priority = 0;
    ctx.max_messages = impl_->config_.max_working_memory_messages;
    
    impl_->contexts_[ctx.id] = ctx;
    
    Logger::instance().info("Created context: " + ctx.id + " (" + name + ")");
    
    return ctx.id;
}

// Get context
std::optional<Context> ContextManager::get_context(const std::string& context_id) const {
    auto it = impl_->contexts_.find(context_id);
    if (it != impl_->contexts_.end()) {
        return it->second;
    }
    return std::nullopt;
}

// Add message
bool ContextManager::add_message(
    const std::string& context_id,
    const ContextMessage& message
) {
    auto it = impl_->contexts_.find(context_id);
    if (it == impl_->contexts_.end()) {
        return false;
    }
    
    it->second.add_message(message);
    Logger::instance().debug("Added message to context " + context_id + " from " + message.author);
    
    return true;
}

// Switch context
bool ContextManager::switch_to_context(const std::string& context_id) {
    auto it = impl_->contexts_.find(context_id);
    if (it == impl_->contexts_.end()) {
        return false;
    }
    
    impl_->active_context_id_ = context_id;
    Logger::instance().info("Switched to context: " + context_id);
    
    // Simulate context switch delay (allows state to settle)
    if (impl_->config_.context_switch_delay_ms > 0) {
        std::this_thread::sleep_for(
            std::chrono::milliseconds(impl_->config_.context_switch_delay_ms)
        );
    }
    
    return true;
}

// Get active context
std::optional<Context> ContextManager::get_active_context() const {
    if (!impl_->active_context_id_) {
        return std::nullopt;
    }
    return get_context(*impl_->active_context_id_);
}

// List contexts
std::vector<Context> ContextManager::list_contexts(
    std::optional<ContextType> type_filter,
    std::optional<bool> active_only
) const {
    std::vector<Context> result;
    
    for (const auto& [id, ctx] : impl_->contexts_) {
        // Apply filters
        if (type_filter && ctx.type != *type_filter) {
            continue;
        }
        if (active_only && *active_only && !ctx.is_active) {
            continue;
        }
        
        result.push_back(ctx);
    }
    
    // Sort by priority (descending) then last activity (descending)
    std::sort(result.begin(), result.end(), [](const Context& a, const Context& b) {
        if (a.priority != b.priority) {
            return a.priority > b.priority;
        }
        return a.last_activity > b.last_activity;
    });
    
    return result;
}

// Archive context
bool ContextManager::archive_context(const std::string& context_id) {
    auto it = impl_->contexts_.find(context_id);
    if (it == impl_->contexts_.end()) {
        return false;
    }
    
    // Archive to memory
    impl_->archive_to_memory(it->second);
    
    // Mark as inactive
    it->second.is_active = false;
    
    Logger::instance().info("Archived context: " + context_id);
    return true;
}

// Restore context
bool ContextManager::restore_context(const std::string& context_id) {
    auto it = impl_->contexts_.find(context_id);
    if (it == impl_->contexts_.end()) {
        return false;
    }
    
    it->second.is_active = true;
    Logger::instance().info("Restored context: " + context_id);
    return true;
}

// Set priority
void ContextManager::set_priority(const std::string& context_id, int priority) {
    auto it = impl_->contexts_.find(context_id);
    if (it != impl_->contexts_.end()) {
        it->second.priority = priority;
        Logger::instance().debug("Set context " + context_id + " priority to " + std::to_string(priority));
    }
}

// Is active
bool ContextManager::is_active(const std::string& context_id) const {
    auto ctx = get_context(context_id);
    return ctx && ctx->is_active;
}

// Get context by channel
std::optional<std::string> ContextManager::get_context_by_channel(const std::string& channel_id) const {
    auto it = impl_->channel_to_context_.find(channel_id);
    if (it != impl_->channel_to_context_.end()) {
        return it->second;
    }
    
    // Try to find by platform_channel_id in contexts
    for (const auto& [id, ctx] : impl_->contexts_) {
        if (ctx.platform_channel_id && *ctx.platform_channel_id == channel_id) {
            return id;
        }
    }
    
    return std::nullopt;
}

// Maintenance
void ContextManager::maintenance() {
    if (!impl_->config_.auto_archive_inactive) {
        return;
    }
    
    auto now = std::chrono::system_clock::now();
    int archived_count = 0;
    
    for (auto& [id, ctx] : impl_->contexts_) {
        if (!ctx.is_active) {
            continue;
        }
        
        auto idle_time = std::chrono::duration_cast<std::chrono::seconds>(
            now - ctx.last_activity
        ).count();
        
        if (idle_time > impl_->config_.inactive_context_timeout) {
            archive_context(id);
            archived_count++;
        }
    }
    
    if (archived_count > 0) {
        Logger::instance().info("Maintenance: Archived " + std::to_string(archived_count) + " inactive contexts");
    }
}

// Get config
const ContextConfig& ContextManager::get_config() const {
    return impl_->config_;
}

// Update config
void ContextManager::update_config(const ContextConfig& config) {
    impl_->config_ = config;
    Logger::instance().info("Context manager config updated");
}

// Save to file
bool ContextManager::save_to_file(const std::string& filepath) const {
    // TODO: Implement full serialization
    Logger::instance().info("Contexts saved to " + filepath);
    return true;
}

// Load from file
bool ContextManager::load_from_file(const std::string& filepath) {
    // TODO: Implement full deserialization
    Logger::instance().info("Contexts loaded from " + filepath);
    return true;
}

} // namespace ash
