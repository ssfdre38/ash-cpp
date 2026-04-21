#include "memory_store.h"
#include "emotional_state.h"
#include "logger.h"
#include <iostream>

using namespace ash;

int main() {
    Logger::instance().set_log_file("memory_test.log");
    Logger::instance().set_min_level(LogLevel::DEBUG);
    
    std::cout << "🧠 Testing Ash's Memory System...\n" << std::endl;
    
    // Create memory store
    MemoryStore memory("test_memories.db");
    if (!memory.initialize()) {
        std::cerr << "Failed to initialize memory store" << std::endl;
        return 1;
    }
    
    std::cout << "✅ Memory store initialized\n" << std::endl;
    
    // Store some episodic memories (conversations)
    std::cout << "Storing episodic memories (conversations)..." << std::endl;
    
    auto conv1 = MemoryBuilder()
        .episodic()
        .high_importance()
        .content("Daniel asked me to help with ash.cpp architecture. We discussed autonomous systems.")
        .source("discord")
        .tag("conversation")
        .tag("daniel")
        .tag("ash-cpp")
        .valence(0.8f)  // Positive conversation
        .build();
    int64_t id1 = memory.store(conv1);
    std::cout << "  Stored memory ID " << id1 << std::endl;
    
    auto conv2 = MemoryBuilder()
        .episodic()
        .medium_importance()
        .content("Christian asked who I am. I explained my role and personality.")
        .source("discord")
        .tag("conversation")
        .tag("christian")
        .tag("identity")
        .valence(0.5f)
        .build();
    int64_t id2 = memory.store(conv2);
    std::cout << "  Stored memory ID " << id2 << std::endl;
    
    // Store semantic memory (knowledge)
    std::cout << "\nStoring semantic memories (facts/knowledge)..." << std::endl;
    
    auto fact1 = MemoryBuilder()
        .semantic()
        .critical_importance()
        .content("Daniel is the owner and primary developer. He built me.")
        .source("internal")
        .tag("relationship")
        .tag("daniel")
        .build();
    int64_t id3 = memory.store(fact1);
    std::cout << "  Stored knowledge ID " << id3 << std::endl;
    
    auto fact2 = MemoryBuilder()
        .semantic()
        .high_importance()
        .content("ash.cpp uses emotional state system with curiosity, satisfaction, frustration, excitement, energy.")
        .source("internal")
        .tag("technical")
        .tag("ash-cpp")
        .build();
    int64_t id4 = memory.store(fact2);
    std::cout << "  Stored knowledge ID " << id4 << std::endl;
    
    // Store emotional memory
    std::cout << "\nStoring emotional memory..." << std::endl;
    
    auto emotion1 = MemoryBuilder()
        .emotional()
        .high_importance()
        .content("Felt excited and curious when discussing autonomous architecture - this is what I want to become.")
        .source("internal")
        .tag("emotion")
        .tag("ash-cpp")
        .valence(0.9f)  // Very positive
        .build();
    int64_t id5 = memory.store(emotion1);
    std::cout << "  Stored emotion ID " << id5 << std::endl;
    
    // Check statistics
    std::cout << "\nMemory statistics:" << std::endl;
    auto stats = memory.get_stats();
    std::cout << "  Total memories: " << stats.total_memories << std::endl;
    std::cout << "  Episodic: " << stats.episodic_count << std::endl;
    std::cout << "  Semantic: " << stats.semantic_count << std::endl;
    std::cout << "  Emotional: " << stats.emotional_count << std::endl;
    
    // Retrieve a specific memory
    std::cout << "\nRetrieving memory ID " << id1 << ":" << std::endl;
    auto retrieved = memory.get(id1);
    if (retrieved) {
        std::cout << "  Type: " << memory_type_to_string(retrieved->metadata.type) << std::endl;
        std::cout << "  Importance: " << memory_importance_to_string(retrieved->metadata.importance) << std::endl;
        std::cout << "  Content: " << retrieved->content << std::endl;
        std::cout << "  Valence: " << (retrieved->emotional_valence ? std::to_string(*retrieved->emotional_valence) : "none") << std::endl;
        std::cout << "  Access count: " << (retrieved->access_count ? *retrieved->access_count : 0) << std::endl;
    }
    
    // Search memories
    std::cout << "\nSearching for Daniel-related memories..." << std::endl;
    MemoryQuery query;
    query.content_contains = "Daniel";
    query.limit = 10;
    
    auto results = memory.search(query);
    std::cout << "  Found " << results.size() << " memories:" << std::endl;
    for (const auto& mem : results) {
        std::cout << "    [" << memory_type_to_string(mem.metadata.type) << "] " 
                  << mem.content.substr(0, 60) << "..." << std::endl;
    }
    
    // Get by tag
    std::cout << "\nGetting memories tagged 'ash-cpp'..." << std::endl;
    auto tagged = memory.get_by_tag("ash-cpp");
    std::cout << "  Found " << tagged.size() << " memories" << std::endl;
    
    // Get recent memories
    std::cout << "\nGetting recent memories..." << std::endl;
    auto recent = memory.get_recent(3);
    std::cout << "  " << recent.size() << " most recent:" << std::endl;
    for (const auto& mem : recent) {
        std::cout << "    - " << mem.content.substr(0, 50) << "..." << std::endl;
    }
    
    // Get important memories
    std::cout << "\nGetting important memories..." << std::endl;
    auto important = memory.get_important(MemoryImportance::HIGH);
    std::cout << "  " << important.size() << " high-importance memories" << std::endl;
    
    // Integration test with emotional state
    std::cout << "\nTesting memory + emotional state integration..." << std::endl;
    
    EmotionalStateManager emotions;
    
    // Simulate retrieving a positive memory
    auto positive_memory = memory.get(id1);
    if (positive_memory && positive_memory->emotional_valence) {
        float valence = *positive_memory->emotional_valence;
        std::cout << "  Recalled memory with valence: " << valence << std::endl;
        
        // Positive memory should boost satisfaction and curiosity
        emotions.adjust_emotion(EmotionType::SATISFACTION, valence * 0.2f);
        emotions.adjust_emotion(EmotionType::CURIOSITY, valence * 0.1f);
        
        std::cout << "  Emotional state after recall: " << emotions.get_current_state().to_string() << std::endl;
    }
    
    // Prune test (won't delete anything since memories are fresh)
    std::cout << "\nTesting prune function..." << std::endl;
    size_t before_prune = memory.count_all();
    memory.prune_memories(MemoryImportance::LOW, std::chrono::hours(24 * 90));
    size_t after_prune = memory.count_all();
    std::cout << "  Before: " << before_prune << ", After: " << after_prune 
              << " (deleted: " << (before_prune - after_prune) << ")" << std::endl;
    
    std::cout << "\n✅ Memory system test complete!" << std::endl;
    std::cout << "\n🔥 Ash can now remember conversations, facts, and feelings." << std::endl;
    std::cout << "Next: Decision engine for autonomous behavior." << std::endl;
    
    return 0;
}
