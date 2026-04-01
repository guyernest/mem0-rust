//! MCP server core for mem0 memory operations.
//!
//! Exposes six memory tools and one prompt via the pmcp `#[mcp_server]` macros:
//!
//! **Tools** (`#[mcp_tool]`):
//! - `add_memory` — add memories from a conversation message
//! - `search_memories` — search memories by semantic similarity
//! - `update_memory` — update the content of an existing memory
//! - `delete_memory` — delete a memory by its ID
//! - `get_all_memories` — list all memories matching a scope
//! - `delete_all_memories` — delete all memories matching a scope (requires confirm=true)
//!
//! **Prompts** (`#[mcp_prompt]`):
//! - `dream` — load agent memories and return a structured consolidation template

// ============================================================================
// IMPORTS
// ============================================================================

use chrono::{Duration, Utc};
use mem0_rust::{AddOptions, DeleteOptions, GetAllOptions, Memory, MemoryType, ResetOptions, SearchOptions, UpdateOptions};
use pmcp::mcp_server;
use pmcp::types::{Content, GetPromptResult, PromptMessage, ServerCapabilities, ToolCapabilities};
use pmcp::{Error, RequestHandlerExtra, Result, Server};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

// ============================================================================
// TOOL INPUT TYPES
// ============================================================================

/// Scoping tier for memory operations.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Scope {
    /// Use the caller's user identity (from X-Pmcp-User-Id header)
    User,
    /// Use the caller's agent identity (from X-Pmcp-Agent-Id header)
    Agent,
    /// Use the request_id provided in args (per-session, not per-caller)
    Request,
}

/// Input for the add_memory tool.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
#[schemars(deny_unknown_fields)]
pub struct AddMemoryInput {
    #[schemars(description = "Conversation message or text to extract memories from")]
    pub messages: String,

    #[schemars(description = "Scoping tier: 'user' (from caller identity), 'agent' (from agent identity), or 'request' (from request_id). Overrides the matching explicit ID field.")]
    pub scope: Option<Scope>,

    #[schemars(description = "User ID scope for the memory (at least one of user_id, agent_id, request_id required)")]
    pub user_id: Option<String>,

    #[schemars(description = "Agent ID scope for the memory (at least one of user_id, agent_id, request_id required)")]
    pub agent_id: Option<String>,

    #[schemars(description = "Request ID scope for the memory session/thread (at least one of user_id, agent_id, request_id required)")]
    pub request_id: Option<String>,

    #[schemars(description = "Memory type: semantic_memory, episodic_memory, or procedural_memory")]
    pub memory_type: Option<String>,
}

/// Input for the search_memories tool.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
#[schemars(deny_unknown_fields)]
pub struct SearchMemoriesInput {
    #[schemars(description = "Query text to search memories by semantic similarity")]
    pub query: String,

    #[schemars(description = "Scoping tier: 'user' (from caller identity), 'agent' (from agent identity), or 'request' (from request_id). Overrides the matching explicit ID field.")]
    pub scope: Option<Scope>,

    #[schemars(description = "Filter by user ID scope")]
    pub user_id: Option<String>,

    #[schemars(description = "Filter by agent ID scope")]
    pub agent_id: Option<String>,

    #[schemars(description = "Filter by request ID scope (session/thread)")]
    pub request_id: Option<String>,

    #[schemars(description = "Filter by memory type: semantic_memory, episodic_memory, or procedural_memory")]
    pub memory_type: Option<String>,

    #[schemars(description = "Maximum number of results to return (default: 10)")]
    pub limit: Option<usize>,
}

/// Input for the update_memory tool.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
#[schemars(deny_unknown_fields)]
pub struct UpdateMemoryInput {
    #[schemars(description = "The ID of the memory to update")]
    pub memory_id: String,

    #[schemars(description = "The new content for the memory")]
    pub content: String,

    #[schemars(description = "Scoping tier: 'user' (from caller identity), 'agent' (from agent identity), or 'request' (from request_id). Overrides the matching explicit ID field.")]
    pub scope: Option<Scope>,

    #[schemars(description = "Caller's user ID for ownership validation")]
    pub user_id: Option<String>,

    #[schemars(description = "Caller's agent ID")]
    pub agent_id: Option<String>,

    #[schemars(description = "Caller's request ID")]
    pub request_id: Option<String>,
}

/// Input for the delete_memory tool.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
#[schemars(deny_unknown_fields)]
pub struct DeleteMemoryInput {
    #[schemars(description = "The ID of the memory to delete")]
    pub memory_id: String,

    #[schemars(description = "Scoping tier: 'user' (from caller identity), 'agent' (from agent identity), or 'request' (from request_id). Overrides the matching explicit ID field.")]
    pub scope: Option<Scope>,

    #[schemars(description = "Caller's user ID for ownership validation")]
    pub user_id: Option<String>,

    #[schemars(description = "Caller's agent ID")]
    pub agent_id: Option<String>,

    #[schemars(description = "Caller's request ID")]
    pub request_id: Option<String>,
}

/// Input for the get_all_memories tool.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
#[schemars(deny_unknown_fields)]
pub struct GetAllMemoriesInput {
    #[schemars(description = "Scoping tier: 'user' (from caller identity), 'agent' (from agent identity), or 'request' (from request_id). Overrides the matching explicit ID field.")]
    pub scope: Option<Scope>,

    #[schemars(description = "Filter by user ID scope")]
    pub user_id: Option<String>,

    #[schemars(description = "Filter by agent ID scope")]
    pub agent_id: Option<String>,

    #[schemars(description = "Filter by request ID scope (session/thread)")]
    pub request_id: Option<String>,

    #[schemars(description = "Filter by memory type: semantic_memory, episodic_memory, or procedural_memory")]
    pub memory_type: Option<String>,

    #[schemars(description = "Maximum number of memories to return (default: 100, max: 1000)")]
    pub limit: Option<usize>,
}

/// Input for the delete_all_memories tool.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
#[schemars(deny_unknown_fields)]
pub struct DeleteAllMemoriesInput {
    #[schemars(description = "Scoping tier: 'user' (from caller identity), 'agent' (from agent identity), or 'request' (from request_id). Overrides the matching explicit ID field.")]
    pub scope: Option<Scope>,

    #[schemars(description = "Filter by user ID scope")]
    pub user_id: Option<String>,

    #[schemars(description = "Filter by agent ID scope")]
    pub agent_id: Option<String>,

    #[schemars(description = "Filter by request ID scope (session/thread)")]
    pub request_id: Option<String>,

    #[schemars(description = "Safety flag. Must be true to confirm deletion. Prevents accidental mass deletion.")]
    pub confirm: bool,
}

// ============================================================================
// PROMPT INPUT TYPES
// ============================================================================

/// Input for the dream consolidation prompt.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct DreamInput {
    /// Scoping tier: 'agent' (default), 'user', or 'request'. Controls which memories are loaded for consolidation.
    #[schemars(description = "Scoping tier: 'agent' (default), 'user', or 'request'. Controls which memories are loaded for consolidation.")]
    pub scope: Option<String>,

    /// Agent ID for memory loading (used when scope is not set).
    #[schemars(description = "Agent ID for memory loading (used when scope is not set)")]
    pub agent_id: Option<String>,

    /// User ID for memory loading (used when scope is not set).
    #[schemars(description = "User ID for memory loading (used when scope is not set)")]
    pub user_id: Option<String>,

    /// Request ID for memory loading (used when scope is not set).
    #[schemars(description = "Request ID for memory loading (used when scope is not set)")]
    pub request_id: Option<String>,
}

// ============================================================================
// TOOL OUTPUT TYPES
// ============================================================================

/// Result of an add_memory operation for a single extracted memory event.
#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct AddMemoryResult {
    /// Memory ID (UUID)
    pub id: String,
    /// Memory content
    pub content: String,
    /// Event type: ADD, UPDATE, DELETE, or NOOP
    pub event: String,
}

/// A single result from a search_memories operation.
#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct SearchMemoryResult {
    /// Memory ID (UUID)
    pub id: String,
    /// Memory content
    pub content: String,
    /// Similarity score
    pub score: f32,
    /// User ID scope (if any)
    pub user_id: Option<String>,
    /// Memory type (if any)
    pub memory_type: Option<String>,
}

/// Result of an update_memory or delete_memory operation.
#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct MemoryOpResult {
    /// Whether the operation succeeded
    pub success: bool,
    /// The affected memory ID
    pub id: String,
}

/// A single result from get_all_memories.
#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct GetAllMemoryResult {
    /// Memory ID (UUID)
    pub id: String,
    /// Memory content
    pub content: String,
    /// User ID scope (if any)
    pub user_id: Option<String>,
    /// Agent ID scope (if any)
    pub agent_id: Option<String>,
    /// Request ID scope (if any)
    pub request_id: Option<String>,
    /// Memory type (if any)
    pub memory_type: Option<String>,
    /// Creation timestamp (RFC 3339)
    pub created_at: String,
}

// ============================================================================
// ERROR MAPPING
// ============================================================================

fn map_memory_error(e: mem0_rust::MemoryError) -> Error {
    match e {
        mem0_rust::MemoryError::NotFound(id) => Error::not_found(id),
        mem0_rust::MemoryError::InvalidInput(msg) => Error::invalid_params(msg),
        mem0_rust::MemoryError::Unauthorized { memory_id, reason } => {
            Error::invalid_params(format!(
                "ownership validation failed for memory {}: {}",
                memory_id, reason
            ))
        }
        other => Error::internal(other.to_string()),
    }
}

// ============================================================================
// MEMORY TYPE PARSING HELPER
// ============================================================================

fn parse_memory_type(s: Option<&str>) -> Result<Option<MemoryType>> {
    s.map(|val| match val {
        "semantic_memory" => Ok(MemoryType::Semantic),
        "episodic_memory" => Ok(MemoryType::Episodic),
        "procedural_memory" => Ok(MemoryType::Procedural),
        other => Err(Error::invalid_params(format!(
            "Invalid memory_type '{}': expected semantic_memory, episodic_memory, or procedural_memory",
            other
        ))),
    })
    .transpose()
}

// ============================================================================
// SCOPE PARSING (for prompt args -- string-only per MCP protocol)
// ============================================================================

/// Parse a scope string into the Scope enum.
///
/// MCP prompt arguments are transmitted as strings, so we need manual parsing
/// instead of serde deserialization (which works for tool JSON args).
fn parse_scope(s: &str) -> Result<Scope> {
    match s {
        "user" => Ok(Scope::User),
        "agent" => Ok(Scope::Agent),
        "request" => Ok(Scope::Request),
        other => Err(Error::invalid_params(format!(
            "Invalid scope '{}': expected 'user', 'agent', or 'request'",
            other
        ))),
    }
}

// ============================================================================
// SCOPE RESOLUTION (per D-01, D-02, D-03, D-04)
// ============================================================================

/// Caller identity extracted from platform-injected request headers.
struct CallerContext {
    /// From X-Pmcp-User-Id (AuthContext.subject)
    user_id: Option<String>,
    /// From X-Pmcp-Agent-Id (AuthContext.claims["agent_id"])
    agent_id: Option<String>,
}

/// Resolved scoping IDs after applying scope parameter.
struct ResolvedIds {
    user_id: Option<String>,
    agent_id: Option<String>,
    request_id: Option<String>,
}

/// Extract caller context from platform-injected headers via pmcp AuthContext.
fn extract_caller_context(extra: &RequestHandlerExtra) -> CallerContext {
    let auth = extra.auth_context();
    CallerContext {
        user_id: auth.map(|a| a.subject.clone()),
        agent_id: auth
            .and_then(|a| a.claims.get("agent_id"))
            .and_then(|v| v.as_str())
            .map(|s| s.to_string()),
    }
}

/// Resolve scope parameter to concrete IDs (per D-01, D-02, D-03, D-04).
///
/// - scope="user" -> user_id from caller context (X-Pmcp-User-Id header)
/// - scope="agent" -> agent_id from caller context (X-Pmcp-Agent-Id header)
/// - scope="request" -> request_id from explicit args (per-session, not per-caller)
/// - None -> use explicit IDs as-is (backward compatible)
///
/// Per D-03: scope overrides ONLY the matching tier. Other explicit IDs pass through.
fn resolve_scope(
    scope: Option<&Scope>,
    caller: &CallerContext,
    explicit_user_id: Option<String>,
    explicit_agent_id: Option<String>,
    explicit_request_id: Option<String>,
) -> Result<ResolvedIds> {
    match scope {
        Some(Scope::User) => {
            let uid = caller.user_id.clone().ok_or_else(|| {
                Error::invalid_params(
                    "scope='user' requires X-Pmcp-User-Id header (not present)",
                )
            })?;
            Ok(ResolvedIds {
                user_id: Some(uid),
                agent_id: explicit_agent_id,
                request_id: explicit_request_id,
            })
        }
        Some(Scope::Agent) => {
            let aid = caller.agent_id.clone().ok_or_else(|| {
                Error::invalid_params(
                    "scope='agent' requires X-Pmcp-Agent-Id header (not present)",
                )
            })?;
            Ok(ResolvedIds {
                user_id: explicit_user_id,
                agent_id: Some(aid),
                request_id: explicit_request_id,
            })
        }
        Some(Scope::Request) => {
            let rid = explicit_request_id.ok_or_else(|| {
                Error::invalid_params(
                    "scope='request' requires request_id to be provided in args",
                )
            })?;
            Ok(ResolvedIds {
                user_id: explicit_user_id,
                agent_id: explicit_agent_id,
                request_id: Some(rid),
            })
        }
        None => Ok(ResolvedIds {
            user_id: explicit_user_id,
            agent_id: explicit_agent_id,
            request_id: explicit_request_id,
        }),
    }
}

// ============================================================================
// DREAM CONSOLIDATION PROMPT
// ============================================================================

/// Consolidation prompt template for the dream workflow.
///
/// This prompt is returned to the calling agent's LLM, which then reasons about
/// which memories to merge, supersede, or rewrite using existing MCP tools.
const DREAM_CONSOLIDATION_PROMPT: &str = r#"You are reviewing your operational memories for consolidation. Your goal is to clean up redundant, superseded, or unclear memories so your memory stays sharp and useful.

## Actions You Can Take

For each issue you find, use one of these three actions:

### MERGE
Two memories say the same thing differently. Keep the better-worded one and delete the other.
- Use `update_memory` to improve the kept memory's wording if needed
- Use `delete_memory` to remove the redundant one

### SUPERSEDED
A newer memory contradicts or replaces an older one. The old one is no longer accurate.
- Use `delete_memory` to remove the outdated memory

### REWRITE
A memory is unclear, ambiguous, or poorly worded. Improve it without changing its meaning.
- Use `update_memory` to replace it with clearer wording

## Safeguards

Follow these rules strictly:

1. Do NOT merge memories about different topics even if they sound similar
2. Do NOT infer new facts -- only consolidate what already exists
3. When in doubt, KEEP BOTH memories (slight redundancy > lost nuance)
4. Only DELETE when a newer memory clearly contradicts an older one

## Output Format

Before executing any tool calls, first list your planned actions:

1. State which memories you are examining (by ID)
2. Explain what action you will take and why
3. Then execute the tool calls

If no consolidation is needed, say so and do not call any tools.

Review the memories below and consolidate where appropriate.
"#;

/// Format a list of memory records into a numbered text block for the dream prompt.
fn format_dream_memories(records: &[mem0_rust::MemoryRecord]) -> String {
    let mut output = format!("## Memories to Review ({} total)\n\n", records.len());
    for (i, record) in records.iter().enumerate() {
        output.push_str(&format!(
            "{}. [ID: {}] (created: {})\n   {}\n\n",
            i + 1,
            record.id,
            record.created_at.format("%Y-%m-%d %H:%M UTC"),
            record.content,
        ));
    }
    output
}

// ============================================================================
// SERVER: Tools defined with #[mcp_server] + #[mcp_tool] macros
// ============================================================================

/// MCP server that wraps mem0-rust's Memory API as six MCP tools.
pub struct MemoryServer {
    pub memory: Arc<Memory>,
}

#[mcp_server]
impl MemoryServer {
    /// Add memories from a conversation message.
    #[mcp_tool(description = "Purpose: Store memories extracted from a conversation message.\nScope: Set scope='user' for personal memories tied to the caller's identity, scope='agent' for agent-specific memories, scope='request' for session-scoped memories. Or provide explicit user_id/agent_id/request_id.\nUse when: After a conversation where the user or agent shared information worth remembering.\nReturns: List of memory events (ADD, UPDATE, DELETE, NOOP) with memory IDs and content.")]
    pub async fn add_memory(&self, args: AddMemoryInput, extra: RequestHandlerExtra) -> Result<Vec<AddMemoryResult>> {
        let caller = extract_caller_context(&extra);
        let resolved = resolve_scope(
            args.scope.as_ref(),
            &caller,
            args.user_id,
            args.agent_id,
            args.request_id,
        )?;

        // Validate that at least one scoping ID is provided
        if resolved.user_id.is_none() && resolved.agent_id.is_none() && resolved.request_id.is_none() {
            return Err(Error::invalid_params(
                "At least one of user_id, agent_id, request_id, or scope is required",
            ));
        }

        let memory_type = parse_memory_type(args.memory_type.as_deref())?;

        let options = AddOptions {
            user_id: resolved.user_id,
            agent_id: resolved.agent_id,
            request_id: resolved.request_id,
            memory_type,
            infer: true,
            ..Default::default()
        };

        let add_result = self
            .memory
            .add(args.messages, options)
            .await
            .map_err(map_memory_error)?;

        let results = add_result
            .results
            .into_iter()
            .map(|event| {
                let event_str = match event.event {
                    mem0_rust::EventType::Add => "ADD",
                    mem0_rust::EventType::Update => "UPDATE",
                    mem0_rust::EventType::Delete => "DELETE",
                    mem0_rust::EventType::Noop => "NOOP",
                };

                AddMemoryResult {
                    id: event.id.to_string(),
                    content: event.memory,
                    event: event_str.to_owned(),
                }
            })
            .collect();

        Ok(results)
    }

    /// Search memories by semantic similarity.
    #[mcp_tool(description = "Purpose: Search memories by semantic similarity to a query.\nScope: Set scope='user' to search only the caller's memories, scope='agent' for agent-specific memories, scope='request' for current session memories. Or provide explicit user_id/agent_id/request_id filters.\nUse when: Before responding to a user, to recall relevant context or preferences.\nReturns: Ranked list of matching memories with similarity scores.")]
    pub async fn search_memories(&self, args: SearchMemoriesInput, extra: RequestHandlerExtra) -> Result<Vec<SearchMemoryResult>> {
        let caller = extract_caller_context(&extra);
        let resolved = resolve_scope(
            args.scope.as_ref(),
            &caller,
            args.user_id,
            args.agent_id,
            args.request_id,
        )?;

        let memory_type = parse_memory_type(args.memory_type.as_deref())?;

        let options = SearchOptions {
            user_id: resolved.user_id,
            agent_id: resolved.agent_id,
            request_id: resolved.request_id,
            memory_type,
            limit: args.limit,
            ..Default::default()
        };

        let search_result = self
            .memory
            .search(&args.query, options)
            .await
            .map_err(map_memory_error)?;

        let results = search_result
            .results
            .into_iter()
            .map(|scored| SearchMemoryResult {
                id: scored.record.id.to_string(),
                content: scored.record.content,
                score: scored.score,
                user_id: scored.record.user_id,
                memory_type: scored.record.memory_type.map(|mt| mt.to_string()),
            })
            .collect();

        Ok(results)
    }

    /// Update the content of an existing memory.
    #[mcp_tool(description = "Purpose: Update the content of an existing memory by ID.\nScope: Set scope='user'/'agent'/'request' to identify yourself for ownership validation. The caller must match the memory's original scope to update it.\nUse when: A previously stored fact has changed or needs correction.\nReturns: Success status and the updated memory ID.")]
    pub async fn update_memory(&self, args: UpdateMemoryInput, extra: RequestHandlerExtra) -> Result<MemoryOpResult> {
        let caller = extract_caller_context(&extra);
        let resolved = resolve_scope(
            args.scope.as_ref(),
            &caller,
            args.user_id,
            args.agent_id,
            args.request_id,
        )?;

        let options = UpdateOptions {
            user_id: resolved.user_id,
            agent_id: resolved.agent_id,
            request_id: resolved.request_id,
        };
        self.memory
            .update(&args.memory_id, &args.content, options)
            .await
            .map_err(map_memory_error)?;

        Ok(MemoryOpResult {
            success: true,
            id: args.memory_id,
        })
    }

    /// Delete a memory by its ID.
    #[mcp_tool(description = "Purpose: Delete a single memory by its ID.\nScope: Set scope='user'/'agent'/'request' to identify yourself for ownership validation. The caller must match the memory's original scope to delete it.\nUse when: A specific memory is no longer relevant or was stored in error.\nReturns: Success status and the deleted memory ID.")]
    pub async fn delete_memory(&self, args: DeleteMemoryInput, extra: RequestHandlerExtra) -> Result<MemoryOpResult> {
        let caller = extract_caller_context(&extra);
        let resolved = resolve_scope(
            args.scope.as_ref(),
            &caller,
            args.user_id,
            args.agent_id,
            args.request_id,
        )?;

        let options = DeleteOptions {
            user_id: resolved.user_id,
            agent_id: resolved.agent_id,
            request_id: resolved.request_id,
        };
        self.memory
            .delete(&args.memory_id, options)
            .await
            .map_err(map_memory_error)?;

        Ok(MemoryOpResult {
            success: true,
            id: args.memory_id,
        })
    }

    /// List all memories matching a scope.
    #[mcp_tool(description = "Purpose: List all memories matching a scope (no query needed).\nScope: Set scope='user' to list the caller's memories, scope='agent' for agent-specific memories, scope='request' for current session memories. Or provide explicit user_id/agent_id/request_id filters.\nUse when: Bootstrapping context at the start of a session (cold-start recall), or auditing what memories exist for a given scope.\nReturns: List of memory objects with id, content, scoping fields, type, and created_at.")]
    pub async fn get_all_memories(
        &self,
        args: GetAllMemoriesInput,
        extra: RequestHandlerExtra,
    ) -> Result<Vec<GetAllMemoryResult>> {
        let caller = extract_caller_context(&extra);
        let resolved = resolve_scope(
            args.scope.as_ref(),
            &caller,
            args.user_id,
            args.agent_id,
            args.request_id,
        )?;

        let options = GetAllOptions {
            user_id: resolved.user_id,
            agent_id: resolved.agent_id,
            request_id: resolved.request_id,
            memory_type: parse_memory_type(args.memory_type.as_deref())?,
            limit: Some(args.limit.unwrap_or(100).min(1000)),
        };

        let records = self
            .memory
            .get_all(options)
            .await
            .map_err(map_memory_error)?;

        Ok(records
            .into_iter()
            .map(|r| GetAllMemoryResult {
                id: r.id.to_string(),
                content: r.content,
                user_id: r.user_id,
                agent_id: r.agent_id,
                request_id: r.request_id,
                memory_type: r.memory_type.map(|mt| mt.to_string()),
                created_at: r.created_at.to_rfc3339(),
            })
            .collect())
    }

    /// Delete all memories matching a scope (requires confirm=true).
    #[mcp_tool(description = "Purpose: Delete all memories matching a scope (bulk cleanup).\nScope: Set scope='user' to delete the caller's memories, scope='agent' for agent-specific memories, scope='request' for current session memories. Or provide explicit user_id/agent_id/request_id filters. If no scope or IDs are provided, deletes ALL memories (full reset).\nUse when: Post-task cleanup of session memories, or resetting a scope.\nReturns: Success status. IMPORTANT: confirm=true is required to prevent accidental deletion.")]
    pub async fn delete_all_memories(
        &self,
        args: DeleteAllMemoriesInput,
        extra: RequestHandlerExtra,
    ) -> Result<MemoryOpResult> {
        // D-13: confirm gate — must be true to proceed
        if !args.confirm {
            return Err(Error::invalid_params(
                "confirm=true required to delete memories",
            ));
        }

        let caller = extract_caller_context(&extra);
        let resolved = resolve_scope(
            args.scope.as_ref(),
            &caller,
            args.user_id,
            args.agent_id,
            args.request_id,
        )?;

        let options = ResetOptions {
            user_id: resolved.user_id,
            agent_id: resolved.agent_id,
            request_id: resolved.request_id,
        };

        self.memory
            .reset(options)
            .await
            .map_err(map_memory_error)?;

        Ok(MemoryOpResult {
            success: true,
            id: "all".to_string(),
        })
    }

    // ========================================================================
    // PROMPT: dream — memory consolidation template
    // ========================================================================

    /// Load agent memories and return a structured consolidation template.
    #[mcp_prompt(description = "Load agent memories and return a structured consolidation template. The agent's LLM reviews memories and uses update_memory/delete_memory to clean up redundant, superseded, or unclear memories.")]
    pub async fn dream(&self, args: DreamInput, extra: RequestHandlerExtra) -> Result<GetPromptResult> {
        // 1. Resolve scope (default: agent per D-04)
        //    When scope is explicitly provided, parse and resolve from caller context.
        //    When scope is None, use explicit IDs (backward compat, same as tools).
        //    When neither scope nor explicit IDs given, default to "agent" from context.
        let caller = extract_caller_context(&extra);
        let has_explicit_ids = args.user_id.is_some()
            || args.agent_id.is_some()
            || args.request_id.is_some();
        let parsed_scope = match args.scope.as_deref() {
            Some(s) => Some(parse_scope(s)?),
            None if !has_explicit_ids => Some(parse_scope("agent")?),
            None => None,
        };
        let resolved = resolve_scope(
            parsed_scope.as_ref(),
            &caller,
            args.user_id,
            args.agent_id,
            args.request_id,
        )?;

        // 2. Load memories (cap at 200 per Pitfall 4 — context window overflow)
        let options = GetAllOptions {
            user_id: resolved.user_id,
            agent_id: resolved.agent_id,
            request_id: resolved.request_id,
            memory_type: None,
            limit: Some(200),
        };
        let all_records = self.memory.get_all(options).await.map_err(map_memory_error)?;

        // 3. Filter to recent 30 days (per D-05)
        let cutoff = Utc::now() - Duration::days(30);
        let recent: Vec<_> = all_records
            .into_iter()
            .filter(|r| r.created_at >= cutoff)
            .collect();

        // 4. Handle empty case (per Pitfall 3)
        if recent.is_empty() {
            return Ok(GetPromptResult::new(
                vec![PromptMessage::user(Content::text(
                    "No recent memories found for consolidation. Nothing to do.",
                ))],
                Some("Dream: no memories to consolidate".to_string()),
            ));
        }

        // 5. Format memory list and return with consolidation instructions
        let memory_list = format_dream_memories(&recent);
        Ok(GetPromptResult::new(
            vec![
                PromptMessage::user(Content::text(DREAM_CONSOLIDATION_PROMPT)),
                PromptMessage::user(Content::text(memory_list)),
            ],
            Some(format!(
                "Dream: {} memories loaded for consolidation",
                recent.len()
            )),
        ))
    }
}

// ============================================================================
// SERVER BUILDER
// ============================================================================

/// Build the mem0 memory MCP server with a pre-initialized Memory instance.
///
/// The caller is responsible for constructing the `Memory` via `Memory::new(config).await`
/// before calling this function. This keeps configuration and initialization concerns
/// in the binary entry point (Lambda wrapper in Phase 5).
pub async fn build_memory_server(memory: Memory) -> Result<Server> {
    Server::builder()
        .name("mem0-memory")
        .version("1.0.0")
        .capabilities({
            let mut caps = ServerCapabilities::default();
            caps.tools = Some(ToolCapabilities {
                list_changed: Some(true),
            });
            caps
        })
        .mcp_server(MemoryServer {
            memory: Arc::new(memory),
        })
        .build()
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use mem0_rust::MemoryConfig;

    #[tokio::test]
    async fn test_server_creation() {
        let config = MemoryConfig::default();
        let memory = Memory::new(config).await.unwrap();
        let server = build_memory_server(memory).await;
        assert!(server.is_ok());
    }
}
