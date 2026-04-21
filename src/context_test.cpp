#include "context_manager.h"
#include "memory_store.h"
#include "logger.h"
#include <iostream>
#include <thread>
#include <chrono>

using namespace ash;
using namespace std::chrono_literals;

void print_context(const Context& ctx) {
    std::cout << "  Context: " << ctx.id << "\n";
    std::cout << "  Name: " << ctx.name << "\n";
    std::cout << "  Type: " << static_cast<int>(ctx.type) << "\n";
    std::cout << "  Messages: " << ctx.messages.size() << "\n";
    std::cout << "  Active: " << (ctx.is_active ? "YES" : "NO") << "\n";
    std::cout << "  Priority: " << ctx.priority << "\n";
}

int main() {
    std::cout << "🗂️ Testing Ash's Context Manager...\n\n";
    
    Logger::instance().set_min_level(LogLevel::DEBUG);
    
    // Create dependencies
    auto memories = std::make_shared<MemoryStore>("test_contexts.db");
    
    std::cout << "✓ Dependencies initialized\n\n";
    
    // Create context manager
    ContextConfig config;
    config.max_active_contexts = 5;
    config.max_working_memory_messages = 20;
    config.inactive_context_timeout = 5.0f; // 5 seconds for testing
    
    auto context_mgr = std::make_shared<ContextManager>(memories, config);
    
    // Test 1: Create contexts
    std::cout << "Test 1: Creating contexts\n";
    
    auto discord_ctx = context_mgr->create_context(
        ContextType::DISCORD_CHANNEL,
        "ash-chat",
        {"daniel", "ash"}
    );
    
    auto thought_ctx = context_mgr->create_context(
        ContextType::INTERNAL_THOUGHT,
        "morning-reflection",
        {"ash"}
    );
    
    auto planning_ctx = context_mgr->create_context(
        ContextType::PLANNING,
        "phase-1-completion",
        {"ash"}
    );
    
    std::cout << "  Created 3 contexts\n\n";
    
    // Test 2: Add messages to context
    std::cout << "Test 2: Adding messages to Discord context\n";
    
    context_mgr->add_message(discord_ctx, ContextMessage{
        .author = "daniel",
        .content = "Hey Ash, how's the context manager coming along?",
        .timestamp = std::chrono::system_clock::now()
    });
    
    context_mgr->add_message(discord_ctx, ContextMessage{
        .author = "ash",
        .content = "Working on it right now! Testing multiple context tracking.",
        .timestamp = std::chrono::system_clock::now()
    });
    
    context_mgr->add_message(discord_ctx, ContextMessage{
        .author = "daniel",
        .content = "Nice! Can you handle multiple conversations at once?",
        .timestamp = std::chrono::system_clock::now()
    });
    
    auto ctx = context_mgr->get_context(discord_ctx);
    if (ctx) {
        std::cout << "  Messages in context: " << ctx->messages.size() << "\n";
        std::cout << "  History:\n" << ctx->get_history_text() << "\n";
    }
    
    // Test 3: Add messages to thought context
    std::cout << "Test 3: Adding internal thoughts\n";
    
    context_mgr->add_message(thought_ctx, ContextMessage{
        .author = "ash",
        .content = "Context management is interesting. I can track multiple conversations and decide which one needs attention.",
        .timestamp = std::chrono::system_clock::now()
    });
    
    context_mgr->add_message(thought_ctx, ContextMessage{
        .author = "ash",
        .content = "Phase 1 is almost complete. Just need to finish context manager testing.",
        .timestamp = std::chrono::system_clock::now()
    });
    
    ctx = context_mgr->get_context(thought_ctx);
    if (ctx) {
        std::cout << "  Thoughts: " << ctx->messages.size() << "\n\n";
    }
    
    // Test 4: Context switching
    std::cout << "Test 4: Context switching\n";
    
    context_mgr->switch_to_context(discord_ctx);
    auto active = context_mgr->get_active_context();
    if (active) {
        std::cout << "  Active context: " << active->name << "\n";
    }
    
    context_mgr->switch_to_context(thought_ctx);
    active = context_mgr->get_active_context();
    if (active) {
        std::cout << "  Switched to: " << active->name << "\n\n";
    }
    
    // Test 5: List all contexts
    std::cout << "Test 5: Listing all contexts\n";
    auto all_contexts = context_mgr->list_contexts();
    std::cout << "  Total contexts: " << all_contexts.size() << "\n";
    for (const auto& c : all_contexts) {
        std::cout << "    - " << c.name << " (" << c.messages.size() << " msgs, priority: " << c.priority << ")\n";
    }
    std::cout << "\n";
    
    // Test 6: Priority management
    std::cout << "Test 6: Setting context priorities\n";
    context_mgr->set_priority(discord_ctx, 10);  // High priority - active conversation
    context_mgr->set_priority(thought_ctx, 5);   // Medium priority - internal thoughts
    context_mgr->set_priority(planning_ctx, 3);  // Low priority - planning
    
    all_contexts = context_mgr->list_contexts();
    std::cout << "  Contexts by priority:\n";
    for (const auto& c : all_contexts) {
        std::cout << "    " << c.priority << ": " << c.name << "\n";
    }
    std::cout << "\n";
    
    // Test 7: Filter contexts by type
    std::cout << "Test 7: Filtering contexts by type\n";
    auto discord_contexts = context_mgr->list_contexts(ContextType::DISCORD_CHANNEL);
    auto thought_contexts = context_mgr->list_contexts(ContextType::INTERNAL_THOUGHT);
    
    std::cout << "  Discord contexts: " << discord_contexts.size() << "\n";
    std::cout << "  Thought contexts: " << thought_contexts.size() << "\n\n";
    
    // Test 8: Archive context
    std::cout << "Test 8: Archiving context\n";
    context_mgr->archive_context(planning_ctx);
    
    auto active_contexts = context_mgr->list_contexts(std::nullopt, true);
    std::cout << "  Active contexts after archiving: " << active_contexts.size() << "\n";
    std::cout << "  Planning context active: " << (context_mgr->is_active(planning_ctx) ? "YES" : "NO") << "\n\n";
    
    // Test 9: Restore context
    std::cout << "Test 9: Restoring archived context\n";
    context_mgr->restore_context(planning_ctx);
    
    std::cout << "  Planning context active: " << (context_mgr->is_active(planning_ctx) ? "YES" : "NO") << "\n\n";
    
    // Test 10: Get recent messages
    std::cout << "Test 10: Getting recent messages\n";
    ctx = context_mgr->get_context(discord_ctx);
    if (ctx) {
        auto recent = ctx->get_recent(2);
        std::cout << "  Last 2 messages in " << ctx->name << ":\n";
        for (const auto& msg : recent) {
            std::cout << "    " << msg.to_string() << "\n";
        }
    }
    std::cout << "\n";
    
    // Test 11: Automatic maintenance (timeout-based archiving)
    std::cout << "Test 11: Testing automatic maintenance\n";
    std::cout << "  Waiting 6 seconds for timeout...\n";
    std::this_thread::sleep_for(6s);
    
    context_mgr->maintenance();
    active_contexts = context_mgr->list_contexts(std::nullopt, true);
    std::cout << "  Active contexts after maintenance: " << active_contexts.size() << "\n";
    std::cout << "  (Contexts with no recent activity should be archived)\n\n";
    
    std::cout << "✓ Context manager test complete!\n";
    std::cout << "🔥 Ash can now track multiple simultaneous conversations.\n";
    std::cout << "Next: Integrate all Phase 1 systems together.\n";
    
    return 0;
}
