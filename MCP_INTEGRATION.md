# MCP Integration Strategy for Ash.cpp

**Date:** April 21, 2026  
**Purpose:** Define how Model Context Protocol (MCP) will be integrated into Ash.cpp for extensible tool/app capabilities

---

## What is MCP?

**Model Context Protocol (MCP)** is a standardized protocol for connecting AI assistants to external tools, data sources, and applications.

**Key Concepts:**
- **MCP Servers** - Apps/services that expose capabilities (GitHub, Discord, file systems, databases)
- **MCP Clients** - AI agents that consume those capabilities
- **Standardized Interface** - JSON-RPC protocol for discovery, invocation, and results
- **Extensibility** - Add new tools without modifying agent code

**Official Spec:** https://modelcontextprotocol.io/

---

## Why MCP for Ash.cpp?

### Current Tool Architecture (Python)
```python
# Hard-coded tool array in ash_tools.py
DISCORD_TOOLS = [
    {"name": "react_to_message", ...},
    {"name": "edit_my_message", ...},
    # ... 17 total tools
]
```

**Problems:**
- ❌ Tools are hard-coded in source
- ❌ Adding new tool requires code change + restart
- ❌ No separation between core engine and tool layer
- ❌ Can't share tools across multiple agents
- ❌ No standard for third-party extensions

### MCP Architecture (Ash.cpp)
```
┌─────────────────────────────────────────┐
│         Ash.cpp Core Engine             │
│  (Inference + Autonomy + Memory)        │
└──────────────┬──────────────────────────┘
               │ MCP Client
               │
    ┌──────────┴──────────┬────────────────┬─────────────────┐
    │                     │                │                 │
┌───▼──────┐      ┌──────▼─────┐   ┌─────▼──────┐   ┌─────▼──────┐
│ Discord  │      │  GitHub    │   │  FileSystem│   │   Custom   │
│  MCP     │      │   MCP      │   │    MCP     │   │   Tools    │
│ Server   │      │  Server    │   │   Server   │   │  (User)    │
└──────────┘      └────────────┘   └────────────┘   └────────────┘
```

**Benefits:**
- ✅ Tools decoupled from core engine
- ✅ Hot-reload new capabilities without restart
- ✅ Standard protocol for third-party extensions
- ✅ Share tool servers across multiple agents
- ✅ Centralized tool registry and discovery

---

## MCP Integration Layers

### Layer 1: MCP Client (Built into Ash.cpp)

**Responsibilities:**
- Discover available MCP servers (stdio, HTTP, WebSocket transports)
- Enumerate tool capabilities from each server
- Invoke tools via JSON-RPC
- Handle results and errors
- Maintain connections (reconnect on failure)

**Implementation:**
```cpp
class MCPClient {
public:
    // Discover and connect to MCP servers
    void discoverServers(const std::vector<MCPServerConfig>& configs);
    
    // Get all available tools across all servers
    std::vector<MCPTool> listTools();
    
    // Invoke a tool
    nlohmann::json invokeTool(const std::string& toolName, 
                              const nlohmann::json& args);
    
    // Tool filtering (pre-filter before sending to model)
    std::vector<MCPTool> filterToolsByContext(const std::string& message);
};
```

**Example Config:**
```json
{
  "mcp_servers": [
    {
      "name": "discord",
      "transport": "stdio",
      "command": "node",
      "args": ["/path/to/discord-mcp-server/index.js"]
    },
    {
      "name": "github",
      "transport": "stdio", 
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"]
    },
    {
      "name": "filesystem",
      "transport": "stdio",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/workspace"]
    }
  ]
}
```

---

### Layer 2: Tool Filtering Engine

**Problem:** If Ash connects to 5 MCP servers with 50 total tools, she can't send all 50 to the model every message (performance killer).

**Solution:** Smart tool filtering BEFORE inference

```cpp
class ToolFilterEngine {
public:
    // Filter tools based on message context
    std::vector<MCPTool> filterTools(
        const std::string& message,
        const EmotionalState& emotion,
        const MemoryContext& memory
    );
    
private:
    // Pattern-based filtering
    bool isGreeting(const std::string& message);
    bool needsMemoryTools(const std::string& message);
    bool needsDiscordTools(const std::string& message);
    bool needsCodeTools(const std::string& message);
};
```

**Filtering Rules:**
```
Message: "hey ash"
→ Greeting pattern detected
→ Tools: NONE (fast path, emotional response only)

Message: "react with 🔥 to that message"
→ Discord action pattern detected
→ Tools: [react_to_message, remove_reaction]

Message: "what do you know about philosophercody?"
→ Memory query pattern detected
→ Tools: [read_my_memories, update_my_memories]

Message: "check the latest commit on GitHub"
→ Code/GitHub pattern detected
→ Tools: [github.list_commits, github.get_commit]
```

**Result:** Only 0-5 relevant tools sent to model (not all 50)

---

### Layer 3: MCP Server Implementations

**Core MCP Servers (Provided):**
1. **Discord MCP Server** - All current Discord tools migrated to MCP
2. **Memory MCP Server** - Ash's memory system (read/write/search)
3. **Personality MCP Server** - Read/update soul.json, identity.json
4. **FileSystem MCP Server** - Read workspace files

**Third-Party MCP Servers (Standard):**
- **GitHub MCP** - Repos, commits, PRs, issues
- **Web MCP** - Fetch URLs, scrape, search
- **Database MCP** - Query SQLite/PostgreSQL
- **Brave Search MCP** - Web search

**Custom MCP Servers (User-Created):**
- Users can write their own MCP servers in any language
- Ash discovers and uses them automatically
- Example: Game modding tools, custom APIs, hardware control

---

## Implementation Plan

### Phase 1: MCP Client Core (Week 1-2)
- [ ] Implement MCPClient class (stdio transport only)
- [ ] JSON-RPC message handling
- [ ] Tool discovery and enumeration
- [ ] Basic tool invocation
- [ ] Error handling and retries

### Phase 2: Tool Filtering (Week 2-3)
- [ ] Implement ToolFilterEngine
- [ ] Pattern-based filtering rules
- [ ] Context-aware tool selection
- [ ] Performance testing (0-5 tools vs 50 tools)

### Phase 3: Core MCP Servers (Week 3-5)
- [ ] Migrate Discord tools to MCP server (Node.js/TypeScript)
- [ ] Create Memory MCP server (access memories.json)
- [ ] Create Personality MCP server (soul.json, identity.json)
- [ ] Create FileSystem MCP server (workspace access)

### Phase 4: Integration with Inference (Week 5-6)
- [ ] Inject filtered tools into inference context
- [ ] Tool call parsing from model output
- [ ] Execute tool via MCP, inject result
- [ ] Continue generation with tool results

### Phase 5: Advanced Features (Week 7+)
- [ ] HTTP/WebSocket MCP transports (not just stdio)
- [ ] Parallel tool execution
- [ ] Tool result caching
- [ ] MCP server health monitoring
- [ ] Hot-reload new servers without restart

---

## Benefits Over Hard-Coded Tools

| Feature | Hard-Coded Tools | MCP Integration |
|---------|-----------------|-----------------|
| **Add new tool** | Code change + compile + restart | Drop in MCP server config |
| **Tool updates** | Modify source code | Update MCP server only |
| **Third-party tools** | Must integrate manually | Works automatically |
| **Tool sharing** | Code duplication | One server, many clients |
| **Performance** | All tools always loaded | Filter to 0-5 relevant tools |
| **Extensibility** | Limited | Unlimited |

---

## Example: Adding a New Tool

### Hard-Coded Approach (Current)
1. Edit `ash_tools.py`
2. Add tool to `DISCORD_TOOLS` array
3. Implement tool function
4. Restart Ash
5. Test

**Time:** 30-60 minutes

### MCP Approach (Ash.cpp)
1. Write MCP server (or use existing one)
2. Add server to `mcp_servers` config
3. Restart Ash (or hot-reload if supported)

**Time:** 5 minutes (if server exists), 30 minutes (if custom server)

---

## Security Considerations

### MCP Server Sandboxing
- MCP servers run as separate processes (stdio transport)
- Ash.cpp doesn't execute arbitrary code (only JSON-RPC calls)
- Tool schemas define allowed operations
- User can review/approve MCP servers before enabling

### Permissions Model
```json
{
  "mcp_servers": [
    {
      "name": "discord",
      "permissions": ["read", "write", "react"],
      "allowed_users": ["119510072865980419"]
    },
    {
      "name": "filesystem", 
      "permissions": ["read"],
      "allowed_paths": ["/workspace", "/ash-bot"]
    }
  ]
}
```

---

## Open Questions

1. **Transport Priority:** Start with stdio only, or implement HTTP/WebSocket too?
2. **Tool Schema Format:** Use native MCP format, or convert to custom schema for model?
3. **Filtering Strategy:** Pattern-based rules, or ML-based tool classifier?
4. **Server Discovery:** Static config file, or dynamic discovery (e.g., scan directory)?
5. **Backward Compatibility:** Support legacy Python tools during transition?

---

## Next Steps

1. **Review MCP Specification** - Deep dive into official protocol
2. **Prototype MCP Client** - Simple C++ client that connects to one MCP server
3. **Build Discord MCP Server** - Migrate existing Discord tools to MCP
4. **Benchmark Performance** - Compare 50 tools (no filter) vs 5 tools (filtered)
5. **Integrate with Ash.cpp Core** - Connect MCP client to inference engine

---

## Conclusion

**MCP integration transforms Ash.cpp from a monolithic agent into an extensible platform.**

Instead of hard-coding every capability, Ash becomes a *framework* that can:
- Discover new tools dynamically
- Filter tools intelligently (performance boost)
- Share tool servers with other agents
- Support user-created extensions

**This is the difference between "an AI assistant" and "an AI platform."**

Let's build it. 🦞🔥

---

*Document created: April 21, 2026*  
*Author: Daniel*  
*Status: Proposal / Planning Phase*
