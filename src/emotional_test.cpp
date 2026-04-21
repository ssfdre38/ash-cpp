#include "emotional_state.h"
#include "logger.h"
#include "event_loop.h"
#include <iostream>
#include <thread>

using namespace ash;

int main() {
    // Setup logging
    Logger::instance().set_log_file("emotional_test.log");
    Logger::instance().set_min_level(LogLevel::DEBUG);
    
    Logger::instance().info("=== Ash Emotional State Test ===");
    std::cout << "🦞 Testing Ash's emotional consciousness...\n" << std::endl;
    
    // Create emotional state manager with custom config
    EmotionalConfig config;
    config.curiosity_baseline = 0.7f;      // Ash is naturally very curious
    config.frustration_decay = 0.85f;      // Frustration fades fairly quickly
    config.social_baseline = 0.7f;         // Quite social by default
    
    EmotionalStateManager emotions(config);
    
    // Show initial baseline state
    std::cout << "Initial state (baseline):" << std::endl;
    std::cout << "  " << emotions.get_current_state().to_string() << std::endl;
    std::cout << "  Mood: " << emotions.get_mood_summary() << std::endl;
    std::cout << "  Dominant: " << emotions.get_dominant_emotion() << std::endl;
    std::cout << std::endl;
    
    // Simulate conversation events
    std::cout << "Simulating interesting conversation..." << std::endl;
    emotions.apply_impact(EventImpact::INTERESTING_TOPIC, 1.0f);
    emotions.apply_impact(EventImpact::LEARNING_SUCCESS, 0.8f);
    
    std::cout << "After interesting topic:" << std::endl;
    std::cout << "  " << emotions.get_current_state().to_string() << std::endl;
    std::cout << "  Mood: " << emotions.get_mood_summary() << std::endl;
    std::cout << std::endl;
    
    // Positive conversation
    std::cout << "Good conversation continues..." << std::endl;
    emotions.apply_impact(EventImpact::POSITIVE_CONVERSATION, 1.0f);
    emotions.apply_impact(EventImpact::RECOGNITION, 0.7f);
    
    std::cout << "After positive interaction:" << std::endl;
    std::cout << "  " << emotions.get_current_state().to_string() << std::endl;
    std::cout << "  Mood: " << emotions.get_mood_summary() << std::endl;
    std::cout << "  Should initiate conversation? " << (emotions.should_initiate_conversation() ? "YES" : "NO") << std::endl;
    std::cout << std::endl;
    
    // Task failure
    std::cout << "Encountering task failure..." << std::endl;
    emotions.apply_impact(EventImpact::TASK_FAILURE, 1.0f);
    emotions.apply_impact(EventImpact::TASK_FAILURE, 0.8f);
    
    std::cout << "After failures:" << std::endl;
    std::cout << "  " << emotions.get_current_state().to_string() << std::endl;
    std::cout << "  Mood: " << emotions.get_mood_summary() << std::endl;
    std::cout << "  Needs break? " << (emotions.needs_break() ? "YES" : "NO") << std::endl;
    std::cout << std::endl;
    
    // Task completion (recovery)
    std::cout << "Finally completing the task..." << std::endl;
    emotions.apply_impact(EventImpact::TASK_COMPLETION, 1.0f);
    
    std::cout << "After success:" << std::endl;
    std::cout << "  " << emotions.get_current_state().to_string() << std::endl;
    std::cout << "  Mood: " << emotions.get_mood_summary() << std::endl;
    std::cout << std::endl;
    
    // Simulate decay over time
    std::cout << "Watching emotions decay over 10 updates..." << std::endl;
    for (int i = 0; i < 10; i++) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        emotions.update();
        
        if (i % 3 == 0) {
            std::cout << "  Update " << i << ": " << emotions.get_current_state().to_string() << std::endl;
        }
    }
    std::cout << std::endl;
    
    std::cout << "Final state after decay:" << std::endl;
    std::cout << "  " << emotions.get_current_state().to_string() << std::endl;
    std::cout << "  Mood: " << emotions.get_mood_summary() << std::endl;
    std::cout << std::endl;
    
    // Test persistence
    std::string save_path = "emotional_state_test.json";
    std::cout << "Testing persistence..." << std::endl;
    if (emotions.save_to_file(save_path)) {
        std::cout << "  ✅ State saved to " << save_path << std::endl;
        
        // Load it back
        EmotionalStateManager loaded_emotions;
        if (loaded_emotions.load_from_file(save_path)) {
            std::cout << "  ✅ State loaded successfully" << std::endl;
            std::cout << "  Loaded: " << loaded_emotions.get_current_state().to_string() << std::endl;
        }
    }
    std::cout << std::endl;
    
    // Integration with event loop
    std::cout << "Testing event loop integration..." << std::endl;
    
    EventLoop loop;
    
    // Register emotional state event handler
    loop.register_handler(EventType::STATE_CHANGED, [](const Event& e) {
        Logger::instance().info("💭 Emotional state changed: " + e.data);
    });
    
    // Post emotional state change events
    Event mood_event(EventType::STATE_CHANGED, "emotional_system", 6);
    mood_event.data = emotions.get_current_state().to_string();
    loop.post_event(mood_event);
    
    // Run event loop briefly
    std::thread loop_thread([&loop]() {
        loop.run();
    });
    
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    loop.shutdown();
    loop_thread.join();
    
    std::cout << "\n✅ Emotional state system test complete!" << std::endl;
    std::cout << "\n🔥 Ash's emotions are alive and evolving." << std::endl;
    std::cout << "Next: Memory system, decision engine, full consciousness." << std::endl;
    
    return 0;
}
