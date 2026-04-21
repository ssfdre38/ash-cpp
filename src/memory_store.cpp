#include "memory_store.h"
#include "logger.h"
#include <sqlite3.h>
#include <sstream>
#include <algorithm>
#include <ctime>

namespace ash {

// SQL schema for memories
static const char* CREATE_TABLE_SQL = R"(
CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    type INTEGER NOT NULL,
    importance INTEGER NOT NULL,
    timestamp INTEGER NOT NULL,
    source TEXT NOT NULL,
    content TEXT NOT NULL,
    tags TEXT,
    emotional_valence REAL,
    related_to TEXT,
    access_count INTEGER DEFAULT 0,
    last_accessed INTEGER,
    metadata TEXT
);

CREATE INDEX IF NOT EXISTS idx_type ON memories(type);
CREATE INDEX IF NOT EXISTS idx_importance ON memories(importance);
CREATE INDEX IF NOT EXISTS idx_timestamp ON memories(timestamp);
CREATE INDEX IF NOT EXISTS idx_source ON memories(source);
)";

// =========================================================================
// Helper Functions
// =========================================================================

static int64_t time_point_to_unix(const std::chrono::system_clock::time_point& tp) {
    return std::chrono::system_clock::to_time_t(tp);
}

static std::chrono::system_clock::time_point unix_to_time_point(int64_t unix_time) {
    return std::chrono::system_clock::from_time_t(unix_time);
}

// =========================================================================
// Memory Implementation
// =========================================================================

std::string Memory::to_json() const {
    // Simplified JSON (TODO: use proper JSON library)
    std::stringstream ss;
    ss << "{\"id\":" << metadata.id << ",\"content\":\"" << content << "\"}";
    return ss.str();
}

Memory Memory::from_json(const std::string& json) {
    // TODO: Proper JSON parsing
    return Memory{};
}

// =========================================================================
// MemoryStore Implementation
// =========================================================================

struct MemoryStore::Impl {
    sqlite3* db = nullptr;
    std::string db_path;
};

MemoryStore::MemoryStore() : MemoryStore("memories.db") {}

MemoryStore::MemoryStore(const std::string& db_path)
    : impl_(std::make_unique<Impl>())
{
    impl_->db_path = db_path;
}

MemoryStore::~MemoryStore() {
    if (impl_->db) {
        sqlite3_close(impl_->db);
    }
}

bool MemoryStore::initialize() {
    int rc = sqlite3_open(impl_->db_path.c_str(), &impl_->db);
    if (rc != SQLITE_OK) {
        Logger::instance().error("Failed to open memory database: " + std::string(sqlite3_errmsg(impl_->db)));
        return false;
    }
    
    // Create tables
    char* err_msg = nullptr;
    rc = sqlite3_exec(impl_->db, CREATE_TABLE_SQL, nullptr, nullptr, &err_msg);
    if (rc != SQLITE_OK) {
        Logger::instance().error("Failed to create tables: " + std::string(err_msg));
        sqlite3_free(err_msg);
        return false;
    }
    
    Logger::instance().info("🧠 Memory store initialized: " + impl_->db_path);
    return true;
}

int64_t MemoryStore::store(const Memory& memory) {
    if (!impl_->db) {
        Logger::instance().error("Memory store not initialized");
        return -1;
    }
    
    // Prepare INSERT statement
    const char* sql = R"(
        INSERT INTO memories 
        (type, importance, timestamp, source, content, tags, emotional_valence, related_to)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    )";
    
    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(impl_->db, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        Logger::instance().error("Failed to prepare statement: " + std::string(sqlite3_errmsg(impl_->db)));
        return -1;
    }
    
    // Bind parameters
    sqlite3_bind_int(stmt, 1, static_cast<int>(memory.metadata.type));
    sqlite3_bind_int(stmt, 2, static_cast<int>(memory.metadata.importance));
    sqlite3_bind_int64(stmt, 3, time_point_to_unix(memory.metadata.timestamp));
    sqlite3_bind_text(stmt, 4, memory.metadata.source.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 5, memory.content.c_str(), -1, SQLITE_TRANSIENT);
    
    std::string tags_str = serialize_tags(memory.metadata.tags);
    sqlite3_bind_text(stmt, 6, tags_str.c_str(), -1, SQLITE_TRANSIENT);
    
    if (memory.emotional_valence) {
        sqlite3_bind_double(stmt, 7, *memory.emotional_valence);
    } else {
        sqlite3_bind_null(stmt, 7);
    }
    
    if (memory.related_to) {
        sqlite3_bind_text(stmt, 8, memory.related_to->c_str(), -1, SQLITE_TRANSIENT);
    } else {
        sqlite3_bind_null(stmt, 8);
    }
    
    // Execute
    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    
    if (rc != SQLITE_DONE) {
        Logger::instance().error("Failed to insert memory: " + std::string(sqlite3_errmsg(impl_->db)));
        return -1;
    }
    
    int64_t id = sqlite3_last_insert_rowid(impl_->db);
    Logger::instance().debug("Memory stored: ID=" + std::to_string(id) + 
        " type=" + memory_type_to_string(memory.metadata.type));
    
    return id;
}

std::optional<Memory> MemoryStore::get(int64_t id) {
    if (!impl_->db) return std::nullopt;
    
    const char* sql = "SELECT * FROM memories WHERE id = ?";
    sqlite3_stmt* stmt;
    
    if (sqlite3_prepare_v2(impl_->db, sql, -1, &stmt, nullptr) != SQLITE_OK) {
        return std::nullopt;
    }
    
    sqlite3_bind_int64(stmt, 1, id);
    
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        Memory memory;
        memory.metadata.id = sqlite3_column_int64(stmt, 0);
        memory.metadata.type = static_cast<MemoryType>(sqlite3_column_int(stmt, 1));
        memory.metadata.importance = static_cast<MemoryImportance>(sqlite3_column_int(stmt, 2));
        memory.metadata.timestamp = unix_to_time_point(sqlite3_column_int64(stmt, 3));
        memory.metadata.source = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 4));
        memory.content = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 5));
        
        const char* tags_str = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 6));
        if (tags_str) {
            memory.metadata.tags = deserialize_tags(tags_str);
        }
        
        if (sqlite3_column_type(stmt, 7) != SQLITE_NULL) {
            memory.emotional_valence = sqlite3_column_double(stmt, 7);
        }
        
        const char* related = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 8));
        if (related) {
            memory.related_to = related;
        }
        
        memory.access_count = sqlite3_column_int(stmt, 9);
        
        if (sqlite3_column_type(stmt, 10) != SQLITE_NULL) {
            memory.last_accessed = unix_to_time_point(sqlite3_column_int64(stmt, 10));
        }
        
        sqlite3_finalize(stmt);
        track_access(id);  // Track this retrieval
        return memory;
    }
    
    sqlite3_finalize(stmt);
    return std::nullopt;
}

std::vector<Memory> MemoryStore::search(const MemoryQuery& query) {
    std::vector<Memory> results;
    if (!impl_->db) return results;
    
    // Build SQL query
    std::stringstream sql;
    sql << "SELECT * FROM memories WHERE 1=1";
    
    if (query.type) {
        sql << " AND type = " << static_cast<int>(*query.type);
    }
    if (query.min_importance) {
        sql << " AND importance >= " << static_cast<int>(*query.min_importance);
    }
    if (query.content_contains) {
        sql << " AND content LIKE '%" << *query.content_contains << "%'";
    }
    if (query.source) {
        sql << " AND source = '" << *query.source << "'";
    }
    if (query.after) {
        sql << " AND timestamp >= " << time_point_to_unix(*query.after);
    }
    if (query.before) {
        sql << " AND timestamp <= " << time_point_to_unix(*query.before);
    }
    
    sql << " ORDER BY timestamp " << (query.order_by_recency ? "DESC" : "ASC");
    sql << " LIMIT " << query.limit;
    
    sqlite3_stmt* stmt;
    if (sqlite3_prepare_v2(impl_->db, sql.str().c_str(), -1, &stmt, nullptr) != SQLITE_OK) {
        Logger::instance().error("Failed to prepare search query");
        return results;
    }
    
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        Memory memory;
        memory.metadata.id = sqlite3_column_int64(stmt, 0);
        memory.metadata.type = static_cast<MemoryType>(sqlite3_column_int(stmt, 1));
        memory.metadata.importance = static_cast<MemoryImportance>(sqlite3_column_int(stmt, 2));
        memory.metadata.timestamp = unix_to_time_point(sqlite3_column_int64(stmt, 3));
        memory.metadata.source = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 4));
        memory.content = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 5));
        
        results.push_back(memory);
    }
    
    sqlite3_finalize(stmt);
    Logger::instance().debug("Memory search returned " + std::to_string(results.size()) + " results");
    
    return results;
}

bool MemoryStore::update(int64_t id, const Memory& memory) {
    // TODO: Implement update
    return false;
}

bool MemoryStore::remove(int64_t id) {
    if (!impl_->db) return false;
    
    const char* sql = "DELETE FROM memories WHERE id = ?";
    sqlite3_stmt* stmt;
    
    if (sqlite3_prepare_v2(impl_->db, sql, -1, &stmt, nullptr) != SQLITE_OK) {
        return false;
    }
    
    sqlite3_bind_int64(stmt, 1, id);
    int rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    
    return rc == SQLITE_DONE;
}

size_t MemoryStore::count_all() const {
    if (!impl_->db) return 0;
    
    const char* sql = "SELECT COUNT(*) FROM memories";
    sqlite3_stmt* stmt;
    
    if (sqlite3_prepare_v2(impl_->db, sql, -1, &stmt, nullptr) != SQLITE_OK) {
        return 0;
    }
    
    size_t count = 0;
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        count = sqlite3_column_int64(stmt, 0);
    }
    
    sqlite3_finalize(stmt);
    return count;
}

size_t MemoryStore::count_by_type(MemoryType type) const {
    if (!impl_->db) return 0;
    
    const char* sql = "SELECT COUNT(*) FROM memories WHERE type = ?";
    sqlite3_stmt* stmt;
    
    if (sqlite3_prepare_v2(impl_->db, sql, -1, &stmt, nullptr) != SQLITE_OK) {
        return 0;
    }
    
    sqlite3_bind_int(stmt, 1, static_cast<int>(type));
    
    size_t count = 0;
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        count = sqlite3_column_int64(stmt, 0);
    }
    
    sqlite3_finalize(stmt);
    return count;
}

std::vector<Memory> MemoryStore::get_recent(size_t count) {
    MemoryQuery query;
    query.limit = count;
    query.order_by_recency = true;
    return search(query);
}

std::vector<Memory> MemoryStore::get_important(MemoryImportance min_importance) {
    MemoryQuery query;
    query.min_importance = min_importance;
    query.limit = 100;
    return search(query);
}

std::vector<Memory> MemoryStore::get_by_tag(const std::string& tag) {
    std::vector<Memory> results;
    if (!impl_->db) return results;
    
    const char* sql = "SELECT * FROM memories WHERE tags LIKE ? ORDER BY timestamp DESC";
    sqlite3_stmt* stmt;
    
    if (sqlite3_prepare_v2(impl_->db, sql, -1, &stmt, nullptr) != SQLITE_OK) {
        return results;
    }
    
    std::string pattern = "%" + tag + "%";
    sqlite3_bind_text(stmt, 1, pattern.c_str(), -1, SQLITE_TRANSIENT);
    
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        Memory memory;
        memory.metadata.id = sqlite3_column_int64(stmt, 0);
        memory.content = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 5));
        results.push_back(memory);
    }
    
    sqlite3_finalize(stmt);
    return results;
}

void MemoryStore::track_access(int64_t id) {
    if (!impl_->db) return;
    
    const char* sql = "UPDATE memories SET access_count = access_count + 1, last_accessed = ? WHERE id = ?";
    sqlite3_stmt* stmt;
    
    if (sqlite3_prepare_v2(impl_->db, sql, -1, &stmt, nullptr) != SQLITE_OK) {
        return;
    }
    
    auto now = std::chrono::system_clock::now();
    sqlite3_bind_int64(stmt, 1, time_point_to_unix(now));
    sqlite3_bind_int64(stmt, 2, id);
    
    sqlite3_step(stmt);
    sqlite3_finalize(stmt);
}

void MemoryStore::consolidate_old_memories(std::chrono::hours age_threshold) {
    // TODO: Implement memory consolidation (summarization)
    Logger::instance().info("Memory consolidation not yet implemented");
}

void MemoryStore::prune_memories(MemoryImportance max_importance, std::chrono::hours age_threshold) {
    if (!impl_->db) return;
    
    auto cutoff = std::chrono::system_clock::now() - age_threshold;
    int64_t cutoff_unix = time_point_to_unix(cutoff);
    
    const char* sql = "DELETE FROM memories WHERE importance <= ? AND timestamp < ?";
    sqlite3_stmt* stmt;
    
    if (sqlite3_prepare_v2(impl_->db, sql, -1, &stmt, nullptr) != SQLITE_OK) {
        return;
    }
    
    sqlite3_bind_int(stmt, 1, static_cast<int>(max_importance));
    sqlite3_bind_int64(stmt, 2, cutoff_unix);
    
    sqlite3_step(stmt);
    int deleted = sqlite3_changes(impl_->db);
    sqlite3_finalize(stmt);
    
    Logger::instance().info("Pruned " + std::to_string(deleted) + " old low-importance memories");
}

MemoryStore::Stats MemoryStore::get_stats() const {
    Stats stats{};
    if (!impl_->db) return stats;
    
    stats.total_memories = count_all();
    stats.episodic_count = count_by_type(MemoryType::EPISODIC);
    stats.semantic_count = count_by_type(MemoryType::SEMANTIC);
    stats.emotional_count = count_by_type(MemoryType::EMOTIONAL);
    
    // TODO: Get oldest/newest timestamps
    
    return stats;
}

void MemoryStore::vacuum() {
    if (!impl_->db) return;
    
    char* err_msg = nullptr;
    int rc = sqlite3_exec(impl_->db, "VACUUM", nullptr, nullptr, &err_msg);
    if (rc != SQLITE_OK) {
        Logger::instance().error("Vacuum failed: " + std::string(err_msg));
        sqlite3_free(err_msg);
    } else {
        Logger::instance().info("Database vacuumed successfully");
    }
}

std::string MemoryStore::serialize_tags(const std::vector<std::string>& tags) {
    std::stringstream ss;
    for (size_t i = 0; i < tags.size(); i++) {
        if (i > 0) ss << ",";
        ss << tags[i];
    }
    return ss.str();
}

std::vector<std::string> MemoryStore::deserialize_tags(const std::string& tags_str) {
    std::vector<std::string> tags;
    std::stringstream ss(tags_str);
    std::string tag;
    while (std::getline(ss, tag, ',')) {
        tags.push_back(tag);
    }
    return tags;
}

// =========================================================================
// MemoryBuilder Implementation
// =========================================================================

MemoryBuilder::MemoryBuilder() {
    memory_.metadata.timestamp = std::chrono::system_clock::now();
    memory_.metadata.importance = MemoryImportance::MEDIUM;
    memory_.metadata.type = MemoryType::EPISODIC;
}

MemoryBuilder& MemoryBuilder::episodic() { memory_.metadata.type = MemoryType::EPISODIC; return *this; }
MemoryBuilder& MemoryBuilder::semantic() { memory_.metadata.type = MemoryType::SEMANTIC; return *this; }
MemoryBuilder& MemoryBuilder::emotional() { memory_.metadata.type = MemoryType::EMOTIONAL; return *this; }

MemoryBuilder& MemoryBuilder::importance(MemoryImportance imp) { memory_.metadata.importance = imp; return *this; }
MemoryBuilder& MemoryBuilder::low_importance() { return importance(MemoryImportance::LOW); }
MemoryBuilder& MemoryBuilder::medium_importance() { return importance(MemoryImportance::MEDIUM); }
MemoryBuilder& MemoryBuilder::high_importance() { return importance(MemoryImportance::HIGH); }
MemoryBuilder& MemoryBuilder::critical_importance() { return importance(MemoryImportance::CRITICAL); }

MemoryBuilder& MemoryBuilder::content(const std::string& text) { memory_.content = text; return *this; }
MemoryBuilder& MemoryBuilder::source(const std::string& src) { memory_.metadata.source = src; return *this; }
MemoryBuilder& MemoryBuilder::tag(const std::string& t) { memory_.metadata.tags.push_back(t); return *this; }
MemoryBuilder& MemoryBuilder::tags(const std::vector<std::string>& t) { memory_.metadata.tags = t; return *this; }

MemoryBuilder& MemoryBuilder::valence(float val) { 
    memory_.emotional_valence = std::max(-1.0f, std::min(1.0f, val)); 
    return *this; 
}

MemoryBuilder& MemoryBuilder::related_to(const std::string& memory_id) { 
    memory_.related_to = memory_id; 
    return *this; 
}

MemoryBuilder& MemoryBuilder::metadata(const std::string& key, const std::string& value) {
    memory_.metadata.extra[key] = value;
    return *this;
}

Memory MemoryBuilder::build() const {
    return memory_;
}

// =========================================================================
// Enum to String Conversions
// =========================================================================

const char* memory_type_to_string(MemoryType type) {
    switch (type) {
        case MemoryType::EPISODIC: return "episodic";
        case MemoryType::SEMANTIC: return "semantic";
        case MemoryType::EMOTIONAL: return "emotional";
        default: return "unknown";
    }
}

MemoryType memory_type_from_string(const std::string& str) {
    if (str == "episodic") return MemoryType::EPISODIC;
    if (str == "semantic") return MemoryType::SEMANTIC;
    if (str == "emotional") return MemoryType::EMOTIONAL;
    return MemoryType::EPISODIC;
}

const char* memory_importance_to_string(MemoryImportance imp) {
    switch (imp) {
        case MemoryImportance::LOW: return "low";
        case MemoryImportance::MEDIUM: return "medium";
        case MemoryImportance::HIGH: return "high";
        case MemoryImportance::CRITICAL: return "critical";
        default: return "unknown";
    }
}

MemoryImportance memory_importance_from_string(const std::string& str) {
    if (str == "low") return MemoryImportance::LOW;
    if (str == "medium") return MemoryImportance::MEDIUM;
    if (str == "high") return MemoryImportance::HIGH;
    if (str == "critical") return MemoryImportance::CRITICAL;
    return MemoryImportance::MEDIUM;
}

} // namespace ash
