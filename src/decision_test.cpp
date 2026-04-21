#include "decision_engine.h"
#include "emotional_state.h"
#include "memory_store.h"
#include "logger.h"
#include <iostream>
#include <thread>
#include <chrono>

using namespace ash;
using namespace std::chrono_literals;

void print_decision(const Decision& decision) {
    const char* type_str = "";
    switch (decision.type) {
        case DecisionType::SPEAK: type_str = "SPEAK"; break;
        case DecisionType::INITIATE: type_str = "INITIATE"; break;
        case DecisionType::THINK: type_str = "THINK"; break;
        case DecisionType::IDLE: type_str = "IDLE"; break;
        case DecisionType::RECALL: type_str = "RECALL"; break;
    }
    
    std::cout << "  Decision: " << type_str << "\n";
    std::cout << "  Confidence: " << decision.confidence << "\n";
    std::cout << "  Reasoning: " << decision.reasoning << "\n";
    if (decision.memory_query) {
        std::cout << "  Memory query: " << *decision.memory_query << "\n";
    }
}

int main() {
    std::cout << "🧠 Testing Ash's Decision Engine...\n\n";
    
    Logger::instance().set_min_level(LogLevel::DEBUG);
    
    // Create dependencies
    auto emotions = std::make_shared<EmotionalStateManager>();
    auto memories = std::make_shared<MemoryStore>("test_decisions.db");
    
    std::cout << "✓ Dependencies initialized\n\n";
    
    // Create decision engine
    DecisionConfig config;
    config.min_confidence_to_speak = 0.4f;
    config.min_confidence_to_initiate = 0.7f;
    
    auto decision_engine = std::make_shared<DecisionEngine>(emotions, memories, config);
    
    // Test 1: MESSAGE_RECEIVED with neutral emotions
    std::cout << "Test 1: Message received (neutral emotions)\n";
    auto decision = decision_engine->decide(DecisionTrigger::MESSAGE_RECEIVED, "Hey Ash, how are you?");
    print_decision(decision);
    std::cout << "\n";
    
    // Test 2: High curiosity state
    std::cout << "Test 2: Message received (high curiosity)\n";
    emotions->apply_impact(EventImpact::INTERESTING_TOPIC);
    std::this_thread::sleep_for(100ms);
    
    decision = decision_engine->decide(DecisionTrigger::MESSAGE_RECEIVED, "Want to talk about ash.cpp architecture?");
    print_decision(decision);
    std::cout << "\n";
    
    // Test 3: Memory relevance trigger
    std::cout << "Test 3: Memory relevance trigger\n";
    decision = decision_engine->decide(DecisionTrigger::MEMORY_RELEVANCE, "Remember when Daniel asked about autonomous systems?");
    print_decision(decision);
    std::cout << "\n";
    
    // Test 4: Low energy state
    std::cout << "Test 4: Low energy state\n";
    emotions->apply_impact(EventImpact::TASK_FAILURE);
    emotions->apply_impact(EventImpact::TASK_FAILURE);
    std::this_thread::sleep_for(100ms);
    
    auto state = emotions->get_current_state();
    std::cout << "  Current energy: " << state.get(EmotionType::ENERGY) << "\n";
    
    decision = decision_engine->decide(DecisionTrigger::MESSAGE_RECEIVED, "Can you help me with something?");
    print_decision(decision);
    std::cout << "\n";
    
    // Test 5: should_speak() evaluation
    std::cout << "Test 5: should_speak() evaluation\n";
    emotions->reset_to_baseline();
    emotions->apply_impact(EventImpact::POSITIVE_CONVERSATION);
    
    auto now = std::chrono::system_clock::now();
    auto recent = now - std::chrono::seconds(10);
    auto old = now - std::chrono::seconds(3700);
    
    bool should_speak_recent = decision_engine->should_speak(recent, "test-channel");
    bool should_speak_old = decision_engine->should_speak(old, "test-channel");
    
    std::cout << "  Should speak (10s ago): " << (should_speak_recent ? "YES" : "NO") << "\n";
    std::cout << "  Should speak (1h ago): " << (should_speak_old ? "YES" : "NO") << "\n";
    std::cout << "\n";
    
    // Test 6: should_initiate_conversation()
    std::cout << "Test 6: should_initiate_conversation()\n";
    emotions->reset_to_baseline();
    emotions->apply_impact(EventImpact::LEARNING_SUCCESS);
    emotions->apply_impact(EventImpact::LEARNING_SUCCESS);
    
    state = emotions->get_current_state();
    std::cout << "  Current excitement: " << state.get(EmotionType::EXCITEMENT) << "\n";
    std::cout << "  Current curiosity: " << state.get(EmotionType::CURIOSITY) << "\n";
    
    bool should_initiate = decision_engine->should_initiate_conversation(old, "test-channel");
    std::cout << "  Should initiate (1h silence, high excitement): " << (should_initiate ? "YES" : "NO") << "\n";
    std::cout << "\n";
    
    // Test 7: Social context tracking
    std::cout << "Test 7: Social context tracking\n";
    decision_engine->record_message("active-channel", now - std::chrono::seconds(30));
    decision_engine->record_message("active-channel", now - std::chrono::seconds(60));
    decision_engine->record_message("active-channel", now - std::chrono::seconds(90));
    decision_engine->record_message("quiet-channel", now - std::chrono::seconds(400));
    
    bool active = decision_engine->is_conversation_active("active-channel");
    bool quiet = decision_engine->is_conversation_active("quiet-channel");
    
    std::cout << "  Active channel (3 msgs in 90s): " << (active ? "ACTIVE" : "QUIET") << "\n";
    std::cout << "  Quiet channel (1 msg 6m ago): " << (quiet ? "ACTIVE" : "QUIET") << "\n";
    std::cout << "\n";
    
    // Test 8: Decision confidence calculation
    std::cout << "Test 8: Decision confidence scores\n";
    emotions->reset_to_baseline();
    emotions->apply_impact(EventImpact::INTERESTING_TOPIC);
    
    float speak_conf = decision_engine->calculate_confidence(
        DecisionType::SPEAK, 
        DecisionTrigger::MESSAGE_RECEIVED, 
        "Ash, can you help?"
    );
    float initiate_conf = decision_engine->calculate_confidence(
        DecisionType::INITIATE,
        DecisionTrigger::TIME_ELAPSED,
        ""
    );
    float think_conf = decision_engine->calculate_confidence(
        DecisionType::THINK,
        DecisionTrigger::EMOTIONAL_SHIFT,
        ""
    );
    
    std::cout << "  SPEAK confidence: " << speak_conf << "\n";
    std::cout << "  INITIATE confidence: " << initiate_conf << "\n";
    std::cout << "  THINK confidence: " << think_conf << "\n";
    std::cout << "\n";
    
    std::cout << "✓ Decision engine test complete!\n";
    std::cout << "🔥 Ash can now autonomously decide when to speak and what to do.\n";
    std::cout << "Next: Context manager for handling multiple conversations.\n";
    
    return 0;
}
