#pragma once

#include <string>
#include <chrono>
#include <fstream>
#include <mutex>
#include <sstream>
#include <iomanip>

namespace ash {

enum class LogLevel {
    DEBUG,
    INFO,
    WARNING,
    ERROR
};

class Logger {
public:
    static Logger& instance();
    
    void log(LogLevel level, const std::string& message);
    void set_log_file(const std::string& filename);
    void set_min_level(LogLevel level);
    
    // Convenience methods
    void debug(const std::string& msg) { log(LogLevel::DEBUG, msg); }
    void info(const std::string& msg) { log(LogLevel::INFO, msg); }
    void warning(const std::string& msg) { log(LogLevel::WARNING, msg); }
    void error(const std::string& msg) { log(LogLevel::ERROR, msg); }
    
private:
    Logger() = default;
    std::string format_time();
    std::string level_to_string(LogLevel level);
    
    std::ofstream log_file_;
    LogLevel min_level_ = LogLevel::INFO;
    std::mutex log_mutex_;
};

} // namespace ash
