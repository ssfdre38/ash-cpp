#include "event_loop.h"
#include "logger.h"
#include <thread>
#include <algorithm>

namespace ash {

EventLoop::EventLoop() : running_(false) {}

EventLoop::~EventLoop() {
    if (running_) {
        shutdown();
    }
}

void EventLoop::run() {
    Logger::instance().info("🔥 Ash Engine starting...");
    running_ = true;
    
    // Start timer thread
    std::thread timer_thread(&EventLoop::timer_thread_func, this);
    
    Logger::instance().info("🦞 Event loop running");
    
    while (running_) {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        
        // Wait for events or timeout
        queue_cv_.wait_for(lock, std::chrono::milliseconds(100), [this] {
            return !event_queue_.empty() || !running_;
        });
        
        if (!event_queue_.empty()) {
            // Get highest priority event
            Event event = event_queue_.top();
            event_queue_.pop();
            lock.unlock();
            
            // Process event outside the lock
            process_event(event);
        }
    }
    
    Logger::instance().info("Event loop shutting down...");
    timer_thread.join();
    Logger::instance().info("✅ Event loop stopped");
}

void EventLoop::shutdown() {
    Logger::instance().info("Shutdown requested");
    running_ = false;
    queue_cv_.notify_all();
}

void EventLoop::post_event(const Event& event) {
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        event_queue_.push(event);
    }
    queue_cv_.notify_one();
}

void EventLoop::register_handler(EventType type, EventHandler handler) {
    std::lock_guard<std::mutex> lock(handlers_mutex_);
    handlers_[type].push_back(handler);
}

void EventLoop::schedule_timer(std::chrono::milliseconds delay, const std::string& timer_id) {
    std::lock_guard<std::mutex> lock(timers_mutex_);
    Timer timer;
    timer.fire_time = std::chrono::system_clock::now() + delay;
    timer.id = timer_id;
    timers_.push_back(timer);
}

void EventLoop::process_event(const Event& event) {
    std::string type_name;
    switch (event.type) {
        case EventType::MESSAGE_RECEIVED: type_name = "MESSAGE_RECEIVED"; break;
        case EventType::TIMER_FIRED: type_name = "TIMER_FIRED"; break;
        case EventType::MEMORY_UPDATED: type_name = "MEMORY_UPDATED"; break;
        case EventType::STATE_CHANGED: type_name = "STATE_CHANGED"; break;
        case EventType::SYSTEM_COMMAND: type_name = "SYSTEM_COMMAND"; break;
        case EventType::SHUTDOWN: type_name = "SHUTDOWN"; break;
    }
    
    Logger::instance().debug("Processing event: " + type_name + " from " + event.source);
    
    // Handle shutdown specially
    if (event.type == EventType::SHUTDOWN) {
        shutdown();
        return;
    }
    
    // Call registered handlers
    std::lock_guard<std::mutex> lock(handlers_mutex_);
    auto it = handlers_.find(event.type);
    if (it != handlers_.end()) {
        for (const auto& handler : it->second) {
            try {
                handler(event);
            } catch (const std::exception& e) {
                Logger::instance().error("Handler exception: " + std::string(e.what()));
            }
        }
    }
}

void EventLoop::timer_thread_func() {
    Logger::instance().debug("Timer thread started");
    
    while (running_) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        
        auto now = std::chrono::system_clock::now();
        std::lock_guard<std::mutex> lock(timers_mutex_);
        
        // Check for expired timers
        auto it = timers_.begin();
        while (it != timers_.end()) {
            if (it->fire_time <= now) {
                // Timer fired - post event
                Event timer_event(EventType::TIMER_FIRED, "timer_thread", 7);
                timer_event.data = it->id;
                post_event(timer_event);
                
                it = timers_.erase(it);
            } else {
                ++it;
            }
        }
    }
    
    Logger::instance().debug("Timer thread stopped");
}

} // namespace ash
