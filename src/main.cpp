#include "event_loop.h"
#include "logger.h"
#include <iostream>
#include <csignal>
#include <atomic>

using namespace ash;

// Global event loop for signal handling
static EventLoop* g_event_loop = nullptr;
static std::atomic<bool> g_shutdown_requested{false};

void signal_handler(int signal) {
    if (signal == SIGINT || signal == SIGTERM) {
        std::cout << "\n🛑 Shutdown signal received..." << std::endl;
        g_shutdown_requested = true;
        if (g_event_loop) {
            g_event_loop->shutdown();
        }
    }
}

int main(int argc, char* argv[]) {
    // Setup logger
    Logger::instance().set_log_file("ash_engine.log");
    Logger::instance().set_min_level(LogLevel::DEBUG);
    
    Logger::instance().info("=== Ash Engine v0.1.0 ===");
    Logger::instance().info("🦞 Built by lobsters, for autonomy");
    
    // Create event loop
    EventLoop loop;
    g_event_loop = &loop;
    
    // Setup signal handlers
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);
    
    // Register event handlers
    loop.register_handler(EventType::MESSAGE_RECEIVED, [](const Event& e) {
        Logger::instance().info("📨 Message: " + e.data);
    });
    
    loop.register_handler(EventType::TIMER_FIRED, [](const Event& e) {
        Logger::instance().info("⏰ Timer fired: " + e.data);
    });
    
    loop.register_handler(EventType::STATE_CHANGED, [](const Event& e) {
        Logger::instance().info("💭 State changed: " + e.data);
    });
    
    // Schedule a test timer
    Logger::instance().info("Scheduling test timers...");
    loop.schedule_timer(std::chrono::seconds(5), "heartbeat-5s");
    loop.schedule_timer(std::chrono::seconds(10), "check-in-10s");
    
    // Post a test event
    Event test_msg(EventType::MESSAGE_RECEIVED, "test_system", 5);
    test_msg.data = "Hello from Ash Engine!";
    loop.post_event(test_msg);
    
    // Post a state change event
    Event state_event(EventType::STATE_CHANGED, "emotional_system", 6);
    state_event.data = "curiosity: 0.8, excitement: 0.6";
    loop.post_event(state_event);
    
    // Schedule shutdown after 15 seconds (for testing)
    loop.schedule_timer(std::chrono::seconds(15), "auto-shutdown");
    loop.register_handler(EventType::TIMER_FIRED, [&loop](const Event& e) {
        if (e.data == "auto-shutdown") {
            Logger::instance().info("⏱️ Auto-shutdown timer expired");
            Event shutdown_event(EventType::SHUTDOWN, "main", 10);
            loop.post_event(shutdown_event);
        }
    });
    
    Logger::instance().info("Starting main event loop...");
    Logger::instance().info("Press Ctrl+C to shutdown");
    
    // Run the event loop (blocking)
    loop.run();
    
    Logger::instance().info("=== Ash Engine shutdown complete ===");
    return 0;
}
