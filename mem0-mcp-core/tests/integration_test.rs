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
    AddMemoryInput, DeleteAllMemoriesInput, DeleteMemoryInput, GetAllMemoriesInput,
    MemoryServer, SearchMemoriesInput, UpdateMemoryInput,
};
use mem0_rust::{AddOptions, Memory, MemoryConfig};
use pmcp::RequestHandlerExtra;
use tokio_util::sync::CancellationToken;
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

/// Create a default RequestHandlerExtra for testing (no auth context).
fn test_extra() -> RequestHandlerExtra {
    RequestHandlerExtra::new("test-req".to_string(), CancellationToken::new())
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
            scope: None,
            user_id: Some("test-user".to_string()),
            agent_id: None,
            request_id: None,
            memory_type: None,
            limit: Some(10),
        }, test_extra())
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
            scope: None,
            user_id: Some("update-user".to_string()),
            agent_id: None,
            request_id: None,
        }, test_extra())
        .await
        .expect("update_memory failed");

    assert!(op_result.success, "Expected success=true");
    assert_eq!(op_result.id, id, "Expected ID to match original");

    // Verify update took effect by searching
    let search_results = server
        .search_memories(SearchMemoriesInput {
            query: "Updated content".to_string(),
            scope: None,
            user_id: Some("update-user".to_string()),
            agent_id: None,
            request_id: None,
            memory_type: None,
            limit: Some(10),
        }, test_extra())
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
            scope: None,
            user_id: Some("delete-user".to_string()),
            agent_id: None,
            request_id: None,
        }, test_extra())
        .await
        .expect("delete_memory failed");

    assert!(op_result.success, "Expected success=true");
    assert_eq!(op_result.id, id, "Expected ID to match original");

    // Verify deletion: re-search should return no results containing that text
    let after_delete = server
        .search_memories(SearchMemoriesInput {
            query: "Content to delete".to_string(),
            scope: None,
            user_id: Some("delete-user".to_string()),
            agent_id: None,
            request_id: None,
            memory_type: None,
            limit: Some(10),
        }, test_extra())
        .await
        .expect("search after delete failed");

    let found = after_delete
        .iter()
        .any(|r| r.id == id);
    assert!(!found, "Expected deleted memory to no longer appear in search results");
}

// ============================================================================
// TEST 4: add_memory tool succeeds (no LLM -> falls back to add_raw)
// ============================================================================

/// TST-02, MCP-02 — add_memory tool succeeds with default config (no LLM).
/// With MemoryConfig::default(), infer=true but llm=None -> add_raw path is used.
#[tokio::test]
async fn test_add_memory_tool_succeeds_without_llm() {
    let (server, _memory) = create_test_server().await;

    let results = server
        .add_memory(AddMemoryInput {
            messages: "I enjoy hiking on weekends".to_string(),
            scope: None,
            user_id: Some("add-user".to_string()),
            agent_id: None,
            request_id: None,
            memory_type: None,
        }, test_extra())
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
            scope: None,
            user_id: None,
            agent_id: None,
            request_id: None,
            memory_type: None,
        }, test_extra())
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
            scope: None,
            user_id: Some("empty-user".to_string()),
            agent_id: None,
            request_id: None,
            memory_type: None,
            limit: Some(10),
        }, test_extra())
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
            scope: None,
            user_id: None,
            agent_id: None,
            request_id: None,
        }, test_extra())
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
            scope: None,
            user_id: None,
            agent_id: None,
            request_id: None,
        }, test_extra())
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
            scope: None,
            user_id: Some("crud-user".to_string()),
            agent_id: None,
            request_id: None,
            memory_type: None,
            limit: Some(10),
        }, test_extra())
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
            scope: None,
            user_id: Some("crud-user".to_string()),
            agent_id: None,
            request_id: None,
        }, test_extra())
        .await
        .expect("update failed");
    assert!(update_result.success, "Update should succeed");

    // 4. Search again — should find updated content
    let search2 = server
        .search_memories(SearchMemoriesInput {
            query: "favorite color green".to_string(),
            scope: None,
            user_id: Some("crud-user".to_string()),
            agent_id: None,
            request_id: None,
            memory_type: None,
            limit: Some(10),
        }, test_extra())
        .await
        .expect("search 2 failed");
    let found_green = search2.iter().any(|r| r.content.contains("green"));
    assert!(found_green, "Expected updated content 'green' to appear in search");

    // 5. Delete the memory
    let delete_result = server
        .delete_memory(DeleteMemoryInput {
            memory_id: id.clone(),
            scope: None,
            user_id: Some("crud-user".to_string()),
            agent_id: None,
            request_id: None,
        }, test_extra())
        .await
        .expect("delete failed");
    assert!(delete_result.success, "Delete should succeed");

    // 6. Search after delete — ID should not appear
    let search3 = server
        .search_memories(SearchMemoriesInput {
            query: "favorite color green".to_string(),
            scope: None,
            user_id: Some("crud-user".to_string()),
            agent_id: None,
            request_id: None,
            memory_type: None,
            limit: Some(10),
        }, test_extra())
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

// ============================================================================
// TEST 11: get_all_memories returns seeded memories
// ============================================================================

/// TOOL-03 — get_all_memories lists memories matching scope.
#[tokio::test]
async fn test_get_all_memories_returns_seeded_content() {
    let (server, memory) = create_test_server().await;

    seed_memory(&memory, "Memory one for getall", "getall-user").await;
    seed_memory(&memory, "Memory two for getall", "getall-user").await;

    let results = server
        .get_all_memories(
            GetAllMemoriesInput {
                scope: None,
                user_id: Some("getall-user".to_string()),
                agent_id: None,
                request_id: None,
                memory_type: None,
                limit: None,
            },
            test_extra(),
        )
        .await
        .expect("get_all_memories failed");

    assert!(
        results.len() >= 2,
        "Expected at least 2 memories, got {}",
        results.len()
    );
    // Each result should have non-empty fields
    for r in &results {
        assert!(!r.id.is_empty(), "Expected non-empty ID");
        assert!(!r.content.is_empty(), "Expected non-empty content");
        assert!(!r.created_at.is_empty(), "Expected non-empty created_at");
    }
}

// ============================================================================
// TEST 12: get_all_memories respects limit
// ============================================================================

/// TOOL-03 — get_all_memories limit parameter works.
#[tokio::test]
async fn test_get_all_memories_respects_limit() {
    let (server, memory) = create_test_server().await;

    seed_memory(&memory, "Limit test one", "limit-user").await;
    seed_memory(&memory, "Limit test two", "limit-user").await;
    seed_memory(&memory, "Limit test three", "limit-user").await;

    let results = server
        .get_all_memories(
            GetAllMemoriesInput {
                scope: None,
                user_id: Some("limit-user".to_string()),
                agent_id: None,
                request_id: None,
                memory_type: None,
                limit: Some(2),
            },
            test_extra(),
        )
        .await
        .expect("get_all_memories with limit failed");

    assert_eq!(
        results.len(),
        2,
        "Expected exactly 2 memories with limit=2, got {}",
        results.len()
    );
}

// ============================================================================
// TEST 13: delete_all_memories with confirm=true succeeds
// ============================================================================

/// TOOL-04 — delete_all_memories clears matching memories when confirmed.
#[tokio::test]
async fn test_delete_all_memories_with_confirm_succeeds() {
    let (server, memory) = create_test_server().await;

    seed_memory(&memory, "To be bulk deleted", "bulk-del-user").await;
    seed_memory(&memory, "Also bulk deleted", "bulk-del-user").await;

    let result = server
        .delete_all_memories(
            DeleteAllMemoriesInput {
                scope: None,
                user_id: Some("bulk-del-user".to_string()),
                agent_id: None,
                request_id: None,
                confirm: Some(true),
            },
            test_extra(),
        )
        .await
        .expect("delete_all_memories failed");

    assert!(result.success, "Expected success=true");
    assert_eq!(result.id, "all", "Expected id='all'");

    // Verify deletion: get_all should return empty
    let after = server
        .get_all_memories(
            GetAllMemoriesInput {
                scope: None,
                user_id: Some("bulk-del-user".to_string()),
                agent_id: None,
                request_id: None,
                memory_type: None,
                limit: None,
            },
            test_extra(),
        )
        .await
        .expect("get_all after delete_all failed");

    assert!(
        after.is_empty(),
        "Expected no memories after delete_all, got {}",
        after.len()
    );
}

// ============================================================================
// TEST 14: delete_all_memories without confirm returns error (D-13)
// ============================================================================

/// TOOL-04, D-13 — delete_all_memories rejects when confirm is not true.
#[tokio::test]
async fn test_delete_all_memories_requires_confirm() {
    let (server, _memory) = create_test_server().await;

    // confirm=None
    let result = server
        .delete_all_memories(
            DeleteAllMemoriesInput {
                scope: None,
                user_id: Some("confirm-user".to_string()),
                agent_id: None,
                request_id: None,
                confirm: None,
            },
            test_extra(),
        )
        .await;

    assert!(
        result.is_err(),
        "Expected error when confirm is not provided"
    );

    // confirm=false
    let result2 = server
        .delete_all_memories(
            DeleteAllMemoriesInput {
                scope: None,
                user_id: Some("confirm-user".to_string()),
                agent_id: None,
                request_id: None,
                confirm: Some(false),
            },
            test_extra(),
        )
        .await;

    assert!(
        result2.is_err(),
        "Expected error when confirm is false"
    );
}

// ============================================================================
// TEST 15: scope="request" requires request_id in args (D-02)
// ============================================================================

/// TOOL-02 — scope="request" fails when request_id is not provided.
#[tokio::test]
async fn test_scope_request_requires_request_id() {
    let (server, _memory) = create_test_server().await;

    let result = server
        .search_memories(
            SearchMemoriesInput {
                query: "test".to_string(),
                scope: Some("request".to_string()),
                user_id: None,
                agent_id: None,
                request_id: None, // Missing — should fail
                memory_type: None,
                limit: None,
            },
            test_extra(),
        )
        .await;

    assert!(
        result.is_err(),
        "Expected error when scope='request' but request_id is not provided"
    );
}

// ============================================================================
// TEST 16: invalid scope value returns error
// ============================================================================

/// TOOL-02 — invalid scope value is rejected.
#[tokio::test]
async fn test_invalid_scope_returns_error() {
    let (server, _memory) = create_test_server().await;

    let result = server
        .add_memory(
            AddMemoryInput {
                messages: "test message".to_string(),
                scope: Some("invalid_scope".to_string()),
                user_id: None,
                agent_id: None,
                request_id: None,
                memory_type: None,
            },
            test_extra(),
        )
        .await;

    assert!(
        result.is_err(),
        "Expected error for invalid scope value"
    );
}

// ============================================================================
// TEST 17: scope=None with explicit IDs works (backward compat, D-03)
// ============================================================================

/// TOOL-02, D-03 — explicit IDs still work without scope.
#[tokio::test]
async fn test_explicit_ids_without_scope_still_work() {
    let (server, memory) = create_test_server().await;

    seed_memory(&memory, "Backward compat test", "compat-user").await;

    // Search with explicit user_id, no scope
    let results = server
        .search_memories(
            SearchMemoriesInput {
                query: "Backward compat".to_string(),
                scope: None,
                user_id: Some("compat-user".to_string()),
                agent_id: None,
                request_id: None,
                memory_type: None,
                limit: Some(10),
            },
            test_extra(),
        )
        .await
        .expect("search with explicit IDs should succeed");

    assert!(
        !results.is_empty(),
        "Expected results with explicit user_id and no scope"
    );
}
