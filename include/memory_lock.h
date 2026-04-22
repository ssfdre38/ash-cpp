#pragma once

#include <cstddef>
#include <string>

namespace ash {

// Memory locking utilities for high-performance inference
// Prevents OS from swapping model weights to disk

class MemoryLocker {
public:
    // Lock a region of memory into physical RAM
    // Prevents the OS from paging it out to disk
    // Returns true on success, false on failure
    static bool lock_memory(void* addr, size_t length);
    
    // Unlock previously locked memory
    static bool unlock_memory(void* addr, size_t length);
    
    // Lock all current and future pages (process-wide)
    // WARNING: Requires elevated privileges
    static bool lock_all();
    
    // Unlock all locked pages
    static bool unlock_all();
    
    // Check if memory locking is supported
    static bool is_supported();
    
    // Get maximum lockable memory (in bytes)
    // Returns 0 if unlimited or unknown
    static size_t get_max_lockable();
    
    // Get current locked memory (in bytes)
    static size_t get_current_locked();
    
    // Error message from last operation
    static std::string last_error();

private:
    static std::string last_error_;
};

// RAII wrapper for memory locking
class ScopedMemoryLock {
public:
    ScopedMemoryLock(void* addr, size_t length);
    ~ScopedMemoryLock();
    
    // No copy
    ScopedMemoryLock(const ScopedMemoryLock&) = delete;
    ScopedMemoryLock& operator=(const ScopedMemoryLock&) = delete;
    
    // Move is OK
    ScopedMemoryLock(ScopedMemoryLock&& other) noexcept;
    ScopedMemoryLock& operator=(ScopedMemoryLock&& other) noexcept;
    
    // Check if lock was successful
    bool is_locked() const { return locked_; }
    
    // Get error message if locking failed
    std::string error() const { return error_; }

private:
    void* addr_;
    size_t length_;
    bool locked_;
    std::string error_;
};

// Platform-specific memory allocation hints
namespace memory_hints {
    // Allocate memory with preference for contiguous physical pages
    void* allocate_contiguous(size_t size);
    
    // Free memory allocated with allocate_contiguous
    void free_contiguous(void* addr, size_t size);
    
    // Prefault pages (touch them to force allocation)
    void prefault_pages(void* addr, size_t length);
    
    // Advise kernel about access patterns
    enum class AccessPattern {
        SEQUENTIAL,   // Will be accessed sequentially
        RANDOM,       // Will be accessed randomly
        WILLNEED,     // Will be needed soon
        DONTNEED      // Won't be needed soon
    };
    
    void advise_access_pattern(void* addr, size_t length, AccessPattern pattern);
}

} // namespace ash
