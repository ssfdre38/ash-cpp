#include "memory_lock.h"
#include "logger.h"
#include <sstream>
#include <cstring>

#ifdef _WIN32
    #include <windows.h>
#else
    #include <sys/mman.h>
    #include <sys/resource.h>
    #include <unistd.h>
    #include <errno.h>
#endif

namespace ash {

std::string MemoryLocker::last_error_;

bool MemoryLocker::lock_memory(void* addr, size_t length) {
    if (!addr || length == 0) {
        last_error_ = "Invalid address or length";
        return false;
    }
    
#ifdef _WIN32
    // Windows: VirtualLock
    if (!VirtualLock(addr, length)) {
        DWORD error = GetLastError();
        std::stringstream ss;
        ss << "VirtualLock failed with error " << error;
        if (error == ERROR_WORKING_SET_QUOTA) {
            ss << " (working set quota exceeded - increase process memory limit)";
        }
        last_error_ = ss.str();
        return false;
    }
#else
    // POSIX: mlock
    if (mlock(addr, length) != 0) {
        std::stringstream ss;
        ss << "mlock failed: " << strerror(errno);
        if (errno == ENOMEM) {
            ss << " (insufficient memory or limit exceeded - check ulimit -l)";
        } else if (errno == EPERM) {
            ss << " (permission denied - requires CAP_IPC_LOCK or elevated privileges)";
        }
        last_error_ = ss.str();
        return false;
    }
#endif
    
    Logger::instance().info("Locked " + std::to_string(length) + " bytes at " + 
                           std::to_string(reinterpret_cast<uintptr_t>(addr)));
    return true;
}

bool MemoryLocker::unlock_memory(void* addr, size_t length) {
    if (!addr || length == 0) {
        last_error_ = "Invalid address or length";
        return false;
    }
    
#ifdef _WIN32
    if (!VirtualUnlock(addr, length)) {
        DWORD error = GetLastError();
        std::stringstream ss;
        ss << "VirtualUnlock failed with error " << error;
        last_error_ = ss.str();
        return false;
    }
#else
    if (munlock(addr, length) != 0) {
        last_error_ = std::string("munlock failed: ") + strerror(errno);
        return false;
    }
#endif
    
    return true;
}

bool MemoryLocker::lock_all() {
#ifdef _WIN32
    // Windows doesn't have a direct equivalent - would need to lock each region
    last_error_ = "lock_all not supported on Windows";
    return false;
#else
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
        last_error_ = std::string("mlockall failed: ") + strerror(errno);
        return false;
    }
    Logger::instance().info("Locked all current and future pages");
    return true;
#endif
}

bool MemoryLocker::unlock_all() {
#ifdef _WIN32
    last_error_ = "unlock_all not supported on Windows";
    return false;
#else
    if (munlockall() != 0) {
        last_error_ = std::string("munlockall failed: ") + strerror(errno);
        return false;
    }
    return true;
#endif
}

bool MemoryLocker::is_supported() {
    // Both Windows and POSIX support memory locking
    return true;
}

size_t MemoryLocker::get_max_lockable() {
#ifdef _WIN32
    // On Windows, check working set size
    SIZE_T minSize, maxSize;
    if (GetProcessWorkingSetSize(GetCurrentProcess(), &minSize, &maxSize)) {
        return static_cast<size_t>(maxSize);
    }
    return 0;
#else
    // On POSIX, check RLIMIT_MEMLOCK
    struct rlimit limit;
    if (getrlimit(RLIMIT_MEMLOCK, &limit) == 0) {
        return limit.rlim_cur == RLIM_INFINITY ? 0 : limit.rlim_cur;
    }
    return 0;
#endif
}

size_t MemoryLocker::get_current_locked() {
    // This is difficult to query accurately on most systems
    // Would need to parse /proc/self/status on Linux or use GetProcessMemoryInfo on Windows
    return 0;  // TODO: Implement if needed
}

std::string MemoryLocker::last_error() {
    return last_error_;
}

// ScopedMemoryLock implementation
ScopedMemoryLock::ScopedMemoryLock(void* addr, size_t length)
    : addr_(addr), length_(length), locked_(false) {
    locked_ = MemoryLocker::lock_memory(addr, length);
    if (!locked_) {
        error_ = MemoryLocker::last_error();
    }
}

ScopedMemoryLock::~ScopedMemoryLock() {
    if (locked_ && addr_ && length_ > 0) {
        MemoryLocker::unlock_memory(addr_, length_);
    }
}

ScopedMemoryLock::ScopedMemoryLock(ScopedMemoryLock&& other) noexcept
    : addr_(other.addr_)
    , length_(other.length_)
    , locked_(other.locked_)
    , error_(std::move(other.error_)) {
    other.addr_ = nullptr;
    other.length_ = 0;
    other.locked_ = false;
}

ScopedMemoryLock& ScopedMemoryLock::operator=(ScopedMemoryLock&& other) noexcept {
    if (this != &other) {
        // Unlock current
        if (locked_ && addr_ && length_ > 0) {
            MemoryLocker::unlock_memory(addr_, length_);
        }
        
        // Move from other
        addr_ = other.addr_;
        length_ = other.length_;
        locked_ = other.locked_;
        error_ = std::move(other.error_);
        
        other.addr_ = nullptr;
        other.length_ = 0;
        other.locked_ = false;
    }
    return *this;
}

// Memory hints implementation
namespace memory_hints {

void* allocate_contiguous(size_t size) {
#ifdef _WIN32
    // Use VirtualAlloc with MEM_COMMIT | MEM_RESERVE for contiguous pages
    void* addr = VirtualAlloc(NULL, size, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    return addr;
#else
    // Use mmap with MAP_ANONYMOUS | MAP_PRIVATE
    void* addr = mmap(NULL, size, PROT_READ | PROT_WRITE, 
                      MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    if (addr == MAP_FAILED) {
        return nullptr;
    }
    return addr;
#endif
}

void free_contiguous(void* addr, size_t size) {
    if (!addr) return;
    
#ifdef _WIN32
    VirtualFree(addr, 0, MEM_RELEASE);
#else
    munmap(addr, size);
#endif
}

void prefault_pages(void* addr, size_t length) {
    if (!addr || length == 0) return;
    
    // Touch each page to force allocation
    size_t page_size = 4096;  // Assume 4KB pages
    volatile char* ptr = static_cast<volatile char*>(addr);
    
    for (size_t i = 0; i < length; i += page_size) {
        ptr[i] = ptr[i];  // Read and write to force page in
    }
}

void advise_access_pattern(void* addr, size_t length, AccessPattern pattern) {
#ifdef _WIN32
    // Windows: Limited support via PrefetchVirtualMemory
    // Skip for now
    (void)addr; (void)length; (void)pattern;
#else
    int advice = MADV_NORMAL;
    switch (pattern) {
        case AccessPattern::SEQUENTIAL:
            advice = MADV_SEQUENTIAL;
            break;
        case AccessPattern::RANDOM:
            advice = MADV_RANDOM;
            break;
        case AccessPattern::WILLNEED:
            advice = MADV_WILLNEED;
            break;
        case AccessPattern::DONTNEED:
            advice = MADV_DONTNEED;
            break;
    }
    
    madvise(addr, length, advice);
#endif
}

} // namespace memory_hints

} // namespace ash
