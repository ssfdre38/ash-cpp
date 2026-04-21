#pragma once

#include <functional>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <memory>
#include <chrono>
#include <string>
#include <unordered_map>
#include <vector>
#include <atomic>

namespace ash {

// Event types that can occur in the system
enum class EventType {
    MESSAGE_RECEIVED,    // Discord message received
    TIMER_FIRED,         // Scheduled timer expired
    MEMORY_UPDATED,      // Memory database changed
    STATE_CHANGED,       // Emotional state updated
    SYSTEM_COMMAND,      // Internal system command
    SHUTDOWN             // Shutdown signal
};

// Base event structure
struct Event {
    EventType type;
    std::string source;  // Where did this event come from?
    std::chrono::system_clock::time_point timestamp;
    int priority;        // Higher = more urgent (0-10)
    
    // Event-specific data (will be std::any or variant in real impl)
    std::string data;
    
    Event(EventType t, const std::string& src, int prio = 5)
        : type(t), source(src), timestamp(std::chrono::system_clock::now()), priority(prio) {}
    
    // For priority queue (higher priority = process first)
    bool operator<(const Event& other) const {
        return priority < other.priority;
    }
};

// Main event loop - Ash's consciousness
class EventLoop {
public:
    EventLoop();
    ~EventLoop();
    
    // Start the event loop (blocking)
    void run();
    
    // Request shutdown
    void shutdown();
    
    // Post an event to the queue
    void post_event(const Event& event);
    
    // Register a handler for specific event types
    using EventHandler = std::function<void(const Event&)>;
    void register_handler(EventType type, EventHandler handler);
    
    // Schedule a timer event
    void schedule_timer(std::chrono::milliseconds delay, const std::string& timer_id);
    
private:
    void process_event(const Event& event);
    void timer_thread_func();
    
    // Event queue (priority-based)
    std::priority_queue<Event> event_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    
    // Event handlers
    std::unordered_map<EventType, std::vector<EventHandler>> handlers_;
    std::mutex handlers_mutex_;
    
    // Shutdown flag
    std::atomic<bool> running_{false};
    
    // Timer management (will be expanded)
    struct Timer {
        std::chrono::system_clock::time_point fire_time;
        std::string id;
    };
    std::vector<Timer> timers_;
    std::mutex timers_mutex_;
};

} // namespace ash
