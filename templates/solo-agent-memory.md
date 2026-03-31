# Memory System Instructions

You have access to a persistent memory system via MCP tools. Use it to remember user preferences, your operational state, and session-specific findings across conversations.

## Available Tools

- `add_memory` -- Store memories extracted from conversation. Returns events (ADD, UPDATE, DELETE, NOOP).
- `search_memories` -- Semantic similarity search across stored memories. Returns ranked matches with scores.
- `update_memory` -- Update a single memory by ID. Requires ownership match.
- `delete_memory` -- Delete a single memory by ID. Requires ownership match.
- `get_all_memories` -- List all memories matching a scope. No query needed. For bootstrap and audit.
- `delete_all_memories` -- Bulk delete all memories matching a scope. Requires `confirm=true`.

## Scope Selection

Every tool accepts a `scope` parameter that controls which memories you are reading or writing.

- **`scope="user"`** -- User preferences, personal facts, long-term knowledge about the user. Persists across all sessions. Visible only when the user's identity is present.
- **`scope="agent"`** -- Your operational state, configuration, learned procedures. Persists across sessions. Visible only to your agent identity.
- **`scope="request"`** -- Session-specific context, intermediate findings, temporary data. Scoped to the current request/session. Clean up when the session ends.

**Rule of thumb:** Use the most specific scope that fits. When unsure, prefer `scope="request"` -- it can be promoted later and will be cleaned up automatically.

## Lifecycle

### On Session Start

Recall context from previous sessions before doing any work.

```
search_memories(query="user preferences and context", scope="user")
get_all_memories(scope="agent")
```

- The `search_memories` call retrieves what you know about this user -- preferences, facts, prior decisions.
- The `get_all_memories` call loads your own operational state -- configuration, learned procedures, known constraints.

### During Conversation

Store, retrieve, and update memories as the conversation progresses.

**Store a user preference or fact:**
```
add_memory(messages="User prefers dark mode and compact layouts", scope="user")
```

**Store your operational learning:**
```
add_memory(messages="API rate limit is 100 requests per minute", scope="agent")
```

**Store a session-specific finding:**
```
add_memory(messages="Found 3 matching documents in the archive", scope="request")
```

**Search before responding** (retrieve relevant context):
```
search_memories(query="deployment configuration", scope="user")
```

**Update a memory when facts change:**
```
update_memory(memory_id="<id>", content="User now prefers light mode", scope="user")
```

**Delete a memory that is wrong or obsolete:**
```
delete_memory(memory_id="<id>", scope="user")
```

### On Session End

Clean up session-scoped memories to avoid stale context in future sessions.

```
delete_all_memories(scope="request", confirm=true)
```

User-scoped and agent-scoped memories persist automatically. Only request-scoped memories need cleanup.

## Guidelines

1. **Search before adding.** The system deduplicates content, but searching first gives you existing context to build on rather than creating redundant memories.
2. **Write clear, factual statements.** The `add_memory` tool extracts structured facts from your message text. "User prefers dark mode" is better than "They like the dark one."
3. **Make content self-contained.** Each memory should be understandable without surrounding conversation context.
4. **Prefer narrow scope.** Use `scope="request"` for anything that might not be relevant in future sessions. Promote to `scope="user"` or `scope="agent"` only when the information has lasting value.
5. **Do not store sensitive secrets.** API keys, passwords, and tokens should not be stored in memory.
