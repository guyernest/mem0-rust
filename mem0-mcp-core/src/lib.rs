//! MCP server core for mem0 memory operations.
//!
//! Exposes four memory tools via the pmcp `#[mcp_server]` / `#[mcp_tool]` macros:
//! - `add_memory` — add memories from a conversation message
//! - `search_memories` — search memories by semantic similarity
//! - `update_memory` — update the content of an existing memory
//! - `delete_memory` — delete a memory by its ID

// ============================================================================
// IMPORTS
// ============================================================================

use mem0_rust::{AddOptions, Memory, MemoryType, SearchOptions};
use pmcp::mcp_server;
use pmcp::types::{ServerCapabilities, ToolCapabilities};
use pmcp::{Error, Result, Server};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

// ============================================================================
// TOOL INPUT TYPES
// ============================================================================

/// Input for the add_memory tool.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
#[schemars(deny_unknown_fields)]
pub struct AddMemoryInput {
    #[schemars(description = "Conversation message or text to extract memories from")]
    pub messages: String,

    #[schemars(description = "User ID scope for the memory (at least one of user_id, agent_id, run_id required)")]
    pub user_id: Option<String>,

    #[schemars(description = "Agent ID scope for the memory (at least one of user_id, agent_id, run_id required)")]
    pub agent_id: Option<String>,

    #[schemars(description = "Run ID scope for the memory (at least one of user_id, agent_id, run_id required)")]
    pub run_id: Option<String>,

    #[schemars(description = "Memory type: semantic_memory, episodic_memory, or procedural_memory")]
    pub memory_type: Option<String>,
}

/// Input for the search_memories tool.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
#[schemars(deny_unknown_fields)]
pub struct SearchMemoriesInput {
    #[schemars(description = "Query text to search memories by semantic similarity")]
    pub query: String,

    #[schemars(description = "Filter by user ID scope")]
    pub user_id: Option<String>,

    #[schemars(description = "Filter by agent ID scope")]
    pub agent_id: Option<String>,

    #[schemars(description = "Filter by run ID scope")]
    pub run_id: Option<String>,

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
}

/// Input for the delete_memory tool.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
#[schemars(deny_unknown_fields)]
pub struct DeleteMemoryInput {
    #[schemars(description = "The ID of the memory to delete")]
    pub memory_id: String,
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

// ============================================================================
// ERROR MAPPING
// ============================================================================

fn map_memory_error(e: mem0_rust::MemoryError) -> Error {
    match e {
        mem0_rust::MemoryError::NotFound(id) => Error::not_found(id),
        mem0_rust::MemoryError::InvalidInput(msg) => Error::invalid_params(msg),
        other => Error::internal(other.to_string()),
    }
}

// ============================================================================
// MEMORY TYPE PARSING HELPER
// ============================================================================

fn parse_memory_type(s: Option<&str>) -> Result<Option<MemoryType>> {
    s.map(|val| {
        serde_json::from_value::<MemoryType>(serde_json::Value::String(val.to_string()))
            .map_err(|e| {
                Error::invalid_params(format!(
                    "Invalid memory_type '{}': expected semantic_memory, episodic_memory, or procedural_memory. Error: {}",
                    val, e
                ))
            })
    })
    .transpose()
}

// ============================================================================
// SERVER: Tools defined with #[mcp_server] + #[mcp_tool] macros
// ============================================================================

/// MCP server that wraps mem0-rust's Memory API as four MCP tools.
pub struct MemoryServer {
    pub memory: Arc<Memory>,
}

#[mcp_server]
impl MemoryServer {
    /// Add memories from a conversation message.
    #[mcp_tool(description = "Add memories from a conversation message")]
    pub async fn add_memory(&self, args: AddMemoryInput) -> Result<Vec<AddMemoryResult>> {
        // Validate that at least one scoping ID is provided
        if args.user_id.is_none() && args.agent_id.is_none() && args.run_id.is_none() {
            return Err(Error::invalid_params(
                "At least one of user_id, agent_id, or run_id is required",
            ));
        }

        let memory_type = parse_memory_type(args.memory_type.as_deref())?;

        let options = AddOptions {
            user_id: args.user_id,
            agent_id: args.agent_id,
            run_id: args.run_id,
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
                // Serialize the EventType to its serde string representation (ADD/UPDATE/DELETE/NOOP)
                let event_str = serde_json::to_value(&event.event)
                    .ok()
                    .and_then(|v| v.as_str().map(|s| s.to_string()))
                    .unwrap_or_else(|| format!("{:?}", event.event).to_uppercase());

                AddMemoryResult {
                    id: event.id.to_string(),
                    content: event.memory,
                    event: event_str,
                }
            })
            .collect();

        Ok(results)
    }

    /// Search memories by semantic similarity.
    #[mcp_tool(description = "Search memories by semantic similarity")]
    pub async fn search_memories(&self, args: SearchMemoriesInput) -> Result<Vec<SearchMemoryResult>> {
        let memory_type = parse_memory_type(args.memory_type.as_deref())?;

        let options = SearchOptions {
            user_id: args.user_id,
            agent_id: args.agent_id,
            run_id: args.run_id,
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
    #[mcp_tool(description = "Update the content of an existing memory")]
    pub async fn update_memory(&self, args: UpdateMemoryInput) -> Result<MemoryOpResult> {
        self.memory
            .update(&args.memory_id, &args.content)
            .await
            .map_err(map_memory_error)?;

        Ok(MemoryOpResult {
            success: true,
            id: args.memory_id,
        })
    }

    /// Delete a memory by its ID.
    #[mcp_tool(description = "Delete a memory by its ID")]
    pub async fn delete_memory(&self, args: DeleteMemoryInput) -> Result<MemoryOpResult> {
        self.memory
            .delete(&args.memory_id)
            .await
            .map_err(map_memory_error)?;

        Ok(MemoryOpResult {
            success: true,
            id: args.memory_id,
        })
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
