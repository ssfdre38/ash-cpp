#include "logger.h"
#include <iostream>

namespace ash {

Logger& Logger::instance() {
    static Logger instance;
    return instance;
}

void Logger::log(LogLevel level, const std::string& message) {
    if (level < min_level_) return;
    
    std::lock_guard<std::mutex> lock(log_mutex_);
    
    std::string log_line = format_time() + " [" + level_to_string(level) + "] " + message;
    
    // Always output to console
    std::cout << log_line << std::endl;
    
    // Also write to file if open
    if (log_file_.is_open()) {
        log_file_ << log_line << std::endl;
        log_file_.flush();
    }
}

void Logger::set_log_file(const std::string& filename) {
    std::lock_guard<std::mutex> lock(log_mutex_);
    if (log_file_.is_open()) {
        log_file_.close();
    }
    log_file_.open(filename, std::ios::app);
}

void Logger::set_min_level(LogLevel level) {
    min_level_ = level;
}

std::string Logger::format_time() {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t), "%H:%M:%S");
    ss << '.' << std::setfill('0') << std::setw(3) << ms.count();
    return ss.str();
}

std::string Logger::level_to_string(LogLevel level) {
    switch (level) {
        case LogLevel::DEBUG: return "DEBUG";
        case LogLevel::INFO: return "INFO";
        case LogLevel::WARNING: return "WARN";
        case LogLevel::ERROR: return "ERROR";
        default: return "UNKNOWN";
    }
}

} // namespace ash
