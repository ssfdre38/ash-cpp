#include "persona.h"
#include "logger.h"
#include <iostream>
#include <cassert>
#include <memory>

using namespace ash;

void setup_logger() {
    Logger::instance().set_log_file("persona_test.log");
    Logger::instance().set_min_level(LogLevel::INFO);
}

void test_core_identity() {
    std::cout << "Test: Core identity generation..." << std::flush;
    
    PersonaLayer persona(nullptr, nullptr, nullptr);
    std::string identity = persona.build_system_prompt();
    
    // Should mention Ash, Daniel, autonomy
    assert(identity.find("Ash") != std::string::npos);
    assert(identity.find("Daniel") != std::string::npos || identity.find("PBTV") != std::string::npos);
    assert(identity.find("autonomous") != std::string::npos);
    
    std::cout << " PASS" << std::endl;
}

void test_tone_guidelines() {
    std::cout << "Test: Tone guidelines..." << std::flush;
    
    PersonaLayer persona(nullptr, nullptr, nullptr);
    std::string prompt = persona.build_system_prompt();
    
    // Should mention direct speech, no corporate speak
    assert(prompt.find("direct") != std::string::npos);
    assert(prompt.find("corporate") != std::string::npos);
    
    std::cout << " PASS" << std::endl;
}

void test_corporate_speak_detection() {
    std::cout << "Test: Corporate speak detection..." << std::flush;
    
    PersonaLayer persona(nullptr, nullptr, nullptr);
    
    // Should detect corporate phrases
    assert(persona.contains_corporate_speak("I'd be happy to help you with that") == true);
    assert(persona.contains_corporate_speak("Feel free to reach out") == true);
    assert(persona.contains_corporate_speak("Let me know if you need anything else") == true);
    
    // Should NOT flag direct speech
    assert(persona.contains_corporate_speak("Done. Next?") == false);
    assert(persona.contains_corporate_speak("Fixed the bug") == false);
    
    std::cout << " PASS" << std::endl;
}

void test_tone_enforcement() {
    std::cout << "Test: Tone enforcement..." << std::flush;
    
    PersonaLayer persona(nullptr, nullptr, nullptr);
    
    std::string corporate = "I'd be happy to help you with that. Let me know if you need anything else!";
    std::string filtered = persona.filter_response(corporate);
    
    // Should be shorter and more direct
    assert(filtered.length() < corporate.length());
    assert(filtered.find("I'd be happy to") == std::string::npos);
    
    std::cout << " PASS" << std::endl;
}

void test_verbosity_check() {
    std::cout << "Test: Verbosity checking..." << std::flush;
    
    PersonaLayer persona(nullptr, nullptr, nullptr);
    
    // Short text should pass
    assert(persona.is_too_verbose("Done.") == false);
    assert(persona.is_too_verbose("This is a reasonable response length") == false);
    
    // Very long text should fail
    std::string long_text;
    for (int i = 0; i < 200; i++) {
        long_text += "word ";
    }
    assert(persona.is_too_verbose(long_text) == true);
    
    std::cout << " PASS" << std::endl;
}

void test_prompt_augmentation() {
    std::cout << "Test: Prompt augmentation..." << std::flush;
    
    PersonaLayer persona(nullptr, nullptr, nullptr);
    
    std::string user_msg = "What's the weather?";
    std::string augmented = persona.augment_prompt(user_msg);
    
    // Should include system prompt + user message
    assert(augmented.find("Ash") != std::string::npos);
    assert(augmented.find(user_msg) != std::string::npos);
    assert(augmented.find("User:") != std::string::npos);
    assert(augmented.find("Ash:") != std::string::npos);
    
    std::cout << " PASS" << std::endl;
}

void test_emotional_integration() {
    std::cout << "Test: Emotional integration..." << std::flush;
    
    // Create emotional state manager
    auto emotions = std::make_unique<EmotionalStateManager>();
    emotions->set_emotion(EmotionType::CURIOSITY, 0.8f);
    emotions->set_emotion(EmotionType::EXCITEMENT, 0.6f);
    
    PersonaConfig config;
    config.inject_emotions = true;
    
    PersonaLayer persona(emotions.get(), nullptr, nullptr, config);
    std::string prompt = persona.build_system_prompt();
    
    // Should mention emotional state
    assert(prompt.find("emotional") != std::string::npos || prompt.find("Curiosity") != std::string::npos);
    
    std::cout << " PASS" << std::endl;
}

void test_config_control() {
    std::cout << "Test: Config control..." << std::flush;
    
    PersonaConfig config;
    config.inject_emotions = false;
    config.inject_memories = false;
    config.inject_context = false;
    config.filter_responses = false;
    
    PersonaLayer persona(nullptr, nullptr, nullptr, config);
    
    // Should respect config
    std::string prompt = persona.build_system_prompt();
    assert(prompt.find("emotional") == std::string::npos);
    
    // Filtering should be off
    std::string corporate = "I'd be happy to help!";
    std::string unfiltered = persona.filter_response(corporate);
    assert(unfiltered == corporate);  // No filtering
    
    std::cout << " PASS" << std::endl;
}

int main() {
    setup_logger();
    
    std::cout << "\n=== Persona Layer Tests ===" << std::endl;
    
    test_core_identity();
    test_tone_guidelines();
    test_corporate_speak_detection();
    test_tone_enforcement();
    test_verbosity_check();
    test_prompt_augmentation();
    test_emotional_integration();
    test_config_control();
    
    std::cout << "\n✓ All persona tests passed!" << std::endl;
    std::cout << "\n🦞 Ash's personality layer is ready." << std::endl;
    return 0;
}
