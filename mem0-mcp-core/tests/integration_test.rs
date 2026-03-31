//! Integration tests for mem0-mcp-core.
//!
//! Tests exercise all four MCP tool methods directly on MemoryServer
//! using a real Memory instance with the InMemory vector store and
//! Mock embedder (no external services required).
//!
//! Strategy:
//! - MemoryConfig::default() gives InMemory store + Mock embedder + no LLM
//! - With no LLM configured, memory.add() falls back to add_raw (infer ignored)
//! - So add_memory tool works end-to-end without an external LLM
//! - Seed helpers call memory.add() with infer=false directly for determinism
//! - Tests call tool methods directly (macro exposes them as real async methods)

use mem0_mcp_core::{
    AddMemoryInput, DeleteMemoryInput, MemoryServer, SearchMemoriesInput, UpdateMemoryInput,
};
use mem0_rust::{AddOptions, Memory, MemoryConfig};
use std::sync::Arc;

// ============================================================================
// TEST HELPERS
// ============================================================================

/// Create a MemoryServer backed by InMemory store + Mock embedder (no LLM).
async fn create_test_server() -> (MemoryServer, Arc<Memory>) {
    let config = MemoryConfig::default();
    let memory = Memory::new(config).await.expect("Memory::new failed");
    let memory = Arc::new(memory);
    let server = MemoryServer {
        memory: memory.clone(),
    };
    (server, memory)
}

/// Seed a memory directly via Memory::add with infer=false, bypassing the MCP
/// tool's add path. Returns the assigned memory ID as a String.
async fn seed_memory(memory: &Memory, content: &str, user_id: &str) -> String {
    let result = memory
        .add(
            content,
            AddOptions {
                user_id: Some(user_id.to_string()),
                infer: false,
                ..Default::default()
            },
        )
        .await
        .expect("seed_memory: add failed");

    assert!(
        !result.results.is_empty(),
        "seed_memory: expected at least one result"
    );
    result.results[0].id.to_string()
}

// ============================================================================
// TEST 1: search_memories finds seeded content
// ============================================================================

/// TST-02, MCP-03, MCP-06 — search_memories returns seeded memories.
#[tokio::test]
async fn test_search_memories_finds_seeded_content() {
    let (server, memory) = create_test_server().await;

    seed_memory(&memory, "I love Rust programming", "test-user").await;

    let results = server
        .search_memories(SearchMemoriesInput {
            query: "Rust programming".to_string(),
            user_id: Some("test-user".to_string()),
            agent_id: None,
            request_id: None,
            memory_type: None,
            limit: Some(10),
        })
        .await
        .expect("search_memories failed");

    assert!(!results.is_empty(), "Expected at least one search result");
    let first = &results[0];
    assert!(
        first.content.contains("Rust"),
        "Expected content to contain 'Rust', got: {}",
        first.content
    );
    assert!(first.score >= 0.0, "Expected score >= 0.0");
    // ID should be a parseable UUID
    assert!(
        !first.id.is_empty(),
        "Expected non-empty ID"
    );
    // Scoped user_id returned
    assert_eq!(
        first.user_id.as_deref(),
        Some("test-user"),
        "Expected user_id to be 'test-user'"
    );
}

// ============================================================================
// TEST 2: update_memory changes content
// ============================================================================

/// TST-02, MCP-04, MCP-06 — update_memory returns success and alters the record.
#[tokio::test]
async fn test_update_memory_changes_content() {
    let (server, memory) = create_test_server().await;

    let id = seed_memory(&memory, "Original content", "update-user").await;

    let op_result = server
        .update_memory(UpdateMemoryInput {
            memory_id: id.clone(),
            content: "Updated content".to_string(),
            user_id: Some("update-user".to_string()),
            agent_id: None,
            request_id: None,
        })
        .await
        .expect("update_memory failed");

    assert!(op_result.success, "Expected success=true");
    assert_eq!(op_result.id, id, "Expected ID to match original");

    // Verify update took effect by searching
    let search_results = server
        .search_memories(SearchMemoriesInput {
            query: "Updated content".to_string(),
            user_id: Some("update-user".to_string()),
            agent_id: None,
            request_id: None,
            memory_type: None,
            limit: Some(10),
        })
        .await
        .expect("search after update failed");

    // At least one result should contain the updated text
    let found_updated = search_results
        .iter()
        .any(|r| r.content.contains("Updated"));
    assert!(found_updated, "Expected to find 'Updated' in search results after update");
}

// ============================================================================
// TEST 3: delete_memory removes the record
// ============================================================================

/// TST-02, MCP-05, MCP-06 — delete_memory returns success and removes the record.
#[tokio::test]
async fn test_delete_memory_removes_record() {
    let (server, memory) = create_test_server().await;

    let id = seed_memory(&memory, "Content to delete", "delete-user").await;

    let op_result = server
        .delete_memory(DeleteMemoryInput {
            memory_id: id.clone(),
            user_id: Some("delete-user".to_string()),
            agent_id: None,
            request_id: None,
        })
        .await
        .expect("delete_memory failed");

    assert!(op_result.success, "Expected success=true");
    assert_eq!(op_result.id, id, "Expected ID to match original");

    // Verify deletion: re-search should return no results containing that text
    let after_delete = server
        .search_memories(SearchMemoriesInput {
            query: "Content to delete".to_string(),
            user_id: Some("delete-user".to_string()),
            agent_id: None,
            request_id: None,
            memory_type: None,
            limit: Some(10),
        })
        .await
        .expect("search after delete failed");

    let found = after_delete
        .iter()
        .any(|r| r.id == id);
    assert!(!found, "Expected deleted memory to no longer appear in search results");
}

// ============================================================================
// TEST 4: add_memory tool succeeds (no LLM → falls back to add_raw)
// ============================================================================

/// TST-02, MCP-02 — add_memory tool succeeds with default config (no LLM).
/// With MemoryConfig::default(), infer=true but llm=None → add_raw path is used.
#[tokio::test]
async fn test_add_memory_tool_succeeds_without_llm() {
    let (server, _memory) = create_test_server().await;

    let results = server
        .add_memory(AddMemoryInput {
            messages: "I enjoy hiking on weekends".to_string(),
            user_id: Some("add-user".to_string()),
            agent_id: None,
            request_id: None,
            memory_type: None,
        })
        .await
        .expect("add_memory should succeed with default config");

    assert!(
        !results.is_empty(),
        "Expected at least one AddMemoryResult"
    );
    assert!(
        !results[0].id.is_empty(),
        "Expected non-empty memory ID"
    );
    assert!(
        !results[0].content.is_empty(),
        "Expected non-empty memory content"
    );
    assert!(
        !results[0].event.is_empty(),
        "Expected non-empty event string"
    );
}

// ============================================================================
// TEST 5: add_memory returns error when no scope provided
// ============================================================================

/// MCP-02 — add_memory validates that at least one scope ID is provided.
#[tokio::test]
async fn test_add_memory_requires_scope_id() {
    let (server, _memory) = create_test_server().await;

    let result = server
        .add_memory(AddMemoryInput {
            messages: "test message".to_string(),
            user_id: None,
            agent_id: None,
            request_id: None,
            memory_type: None,
        })
        .await;

    assert!(
        result.is_err(),
        "Expected error when no scope ID is provided"
    );
}

// ============================================================================
// TEST 6: search_memories with no results returns empty vec
// ============================================================================

/// MCP-03, MCP-06 — search on empty store returns Ok with empty results.
#[tokio::test]
async fn test_search_memories_with_no_results() {
    let (server, _memory) = create_test_server().await;

    let results = server
        .search_memories(SearchMemoriesInput {
            query: "something that does not exist".to_string(),
            user_id: Some("empty-user".to_string()),
            agent_id: None,
            request_id: None,
            memory_type: None,
            limit: Some(10),
        })
        .await
        .expect("search_memories on empty store should succeed");

    assert!(
        results.is_empty(),
        "Expected empty results on empty store, got: {:?}",
        results
    );
}

// ============================================================================
// TEST 7: update_memory with nonexistent ID returns error
// ============================================================================

/// MCP-04, MCP-06 — update_memory with a fake UUID returns Err(not found).
#[tokio::test]
async fn test_update_nonexistent_memory_returns_error() {
    let (server, _memory) = create_test_server().await;

    let fake_id = "00000000-0000-0000-0000-000000000001".to_string();
    let result = server
        .update_memory(UpdateMemoryInput {
            memory_id: fake_id,
            content: "should not matter".to_string(),
            user_id: None,
            agent_id: None,
            request_id: None,
        })
        .await;

    assert!(
        result.is_err(),
        "Expected error when updating nonexistent memory"
    );
}

// ============================================================================
// TEST 8: delete_memory with nonexistent ID returns error
// ============================================================================

/// MCP-05, MCP-06 — delete_memory with a fake UUID returns Err(not found).
#[tokio::test]
async fn test_delete_nonexistent_memory_returns_error() {
    let (server, _memory) = create_test_server().await;

    let fake_id = "00000000-0000-0000-0000-000000000002".to_string();
    let result = server
        .delete_memory(DeleteMemoryInput {
            memory_id: fake_id,
            user_id: None,
            agent_id: None,
            request_id: None,
        })
        .await;

    assert!(
        result.is_err(),
        "Expected error when deleting nonexistent memory"
    );
}

// ============================================================================
// TEST 9: full CRUD flow end-to-end
// ============================================================================

/// TST-02 comprehensive — seed, search, update, search again, delete, search again.
#[tokio::test]
async fn test_full_crud_flow() {
    let (server, memory) = create_test_server().await;

    // 1. Seed a memory
    let id = seed_memory(&memory, "Favorite color is blue", "crud-user").await;

    // 2. Search and verify it exists
    let search1 = server
        .search_memories(SearchMemoriesInput {
            query: "favorite color".to_string(),
            user_id: Some("crud-user".to_string()),
            agent_id: None,
            request_id: None,
            memory_type: None,
            limit: Some(10),
        })
        .await
        .expect("search 1 failed");
    assert!(!search1.is_empty(), "Expected to find memory after seed");
    assert!(
        search1.iter().any(|r| r.id == id),
        "Expected seeded memory ID to appear in search results"
    );

    // 3. Update the memory
    let update_result = server
        .update_memory(UpdateMemoryInput {
            memory_id: id.clone(),
            content: "Favorite color is green".to_string(),
            user_id: Some("crud-user".to_string()),
            agent_id: None,
            request_id: None,
        })
        .await
        .expect("update failed");
    assert!(update_result.success, "Update should succeed");

    // 4. Search again — should find updated content
    let search2 = server
        .search_memories(SearchMemoriesInput {
            query: "favorite color green".to_string(),
            user_id: Some("crud-user".to_string()),
            agent_id: None,
            request_id: None,
            memory_type: None,
            limit: Some(10),
        })
        .await
        .expect("search 2 failed");
    let found_green = search2.iter().any(|r| r.content.contains("green"));
    assert!(found_green, "Expected updated content 'green' to appear in search");

    // 5. Delete the memory
    let delete_result = server
        .delete_memory(DeleteMemoryInput {
            memory_id: id.clone(),
            user_id: Some("crud-user".to_string()),
            agent_id: None,
            request_id: None,
        })
        .await
        .expect("delete failed");
    assert!(delete_result.success, "Delete should succeed");

    // 6. Search after delete — ID should not appear
    let search3 = server
        .search_memories(SearchMemoriesInput {
            query: "favorite color green".to_string(),
            user_id: Some("crud-user".to_string()),
            agent_id: None,
            request_id: None,
            memory_type: None,
            limit: Some(10),
        })
        .await
        .expect("search 3 failed");
    let still_present = search3.iter().any(|r| r.id == id);
    assert!(!still_present, "Deleted memory should not appear in search results");
}

// ============================================================================
// TEST 10: build_memory_server builds a valid Server
// ============================================================================

/// MCP-01 — build_memory_server() returns Ok(Server) with default config.
#[tokio::test]
async fn test_build_memory_server_succeeds() {
    use mem0_mcp_core::build_memory_server;

    let config = MemoryConfig::default();
    let memory = Memory::new(config).await.expect("Memory::new failed");
    let server = build_memory_server(memory).await;
    assert!(server.is_ok(), "build_memory_server should succeed");
}
