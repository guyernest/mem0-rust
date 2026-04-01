//! Integration tests for DsqlHistoryStore against a real Aurora DSQL cluster.
//!
//! Run with:
//!   cargo test --features dsql,integration-tests --test integration_dsql
//!
//! Requires:
//! - AWS credentials with dsql:DbConnectAdmin on the test cluster
//! - DSQL_TEST_ENDPOINT env var set to the cluster endpoint

#![cfg(all(feature = "dsql", feature = "integration-tests"))]

use chrono::Utc;
use mem0_rust::history::{create_dsql_pool, DsqlHistoryStore, HistoryStore};
use mem0_rust::models::EventType;
use uuid::Uuid;

async fn setup_store() -> DsqlHistoryStore {
    let endpoint = std::env::var("DSQL_TEST_ENDPOINT")
        .expect("DSQL_TEST_ENDPOINT must be set for integration tests");
    let region = std::env::var("AWS_REGION").ok();
    let pool = create_dsql_pool(&endpoint, region.as_deref())
        .await
        .expect("Failed to create DSQL pool");
    let store = DsqlHistoryStore::new(pool)
        .await
        .expect("Failed to initialize DsqlHistoryStore");
    // Clean slate for each test
    store.reset().await.expect("Failed to reset store");
    store
}

#[tokio::test]
async fn test_dsql_add_and_get_history() {
    let store = setup_store().await;
    let memory_id = Uuid::new_v4();

    store
        .add_history(
            memory_id,
            None,
            "user likes Rust".to_string(),
            EventType::Add,
            Utc::now(),
            Some("u1".to_string()),
            Some("a1".to_string()),
            None,
        )
        .await
        .unwrap();

    let entries = store.get_history(memory_id).await.unwrap();
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].memory_id, memory_id);
    assert_eq!(entries[0].new_content, "user likes Rust");
    assert_eq!(entries[0].event, EventType::Add);
    assert!(entries[0].previous_content.is_none());
}

#[tokio::test]
async fn test_dsql_update_history_ordering() {
    let store = setup_store().await;
    let memory_id = Uuid::new_v4();
    let t1 = Utc::now();

    store
        .add_history(
            memory_id,
            None,
            "user likes Rust".to_string(),
            EventType::Add,
            t1,
            Some("u1".to_string()),
            None,
            None,
        )
        .await
        .unwrap();

    let t2 = t1 + chrono::Duration::seconds(10);
    store
        .add_history(
            memory_id,
            Some("user likes Rust".to_string()),
            "user loves Rust".to_string(),
            EventType::Update,
            t2,
            Some("u1".to_string()),
            None,
            None,
        )
        .await
        .unwrap();

    let entries = store.get_history(memory_id).await.unwrap();
    assert_eq!(entries.len(), 2);
    // Newest first (ORDER BY timestamp DESC)
    assert_eq!(entries[0].event, EventType::Update);
    assert_eq!(
        entries[0].previous_content,
        Some("user likes Rust".to_string())
    );
    assert_eq!(entries[0].new_content, "user loves Rust");
    assert_eq!(entries[1].event, EventType::Add);
}

#[tokio::test]
async fn test_dsql_reset_clears_history() {
    let store = setup_store().await;
    let memory_id = Uuid::new_v4();

    store
        .add_history(
            memory_id,
            None,
            "some content".to_string(),
            EventType::Add,
            Utc::now(),
            None,
            None,
            None,
        )
        .await
        .unwrap();

    assert_eq!(store.get_history(memory_id).await.unwrap().len(), 1);

    store.reset().await.unwrap();
    assert!(store.get_history(memory_id).await.unwrap().is_empty());
}

#[tokio::test]
async fn test_dsql_get_history_nonexistent_returns_empty() {
    let store = setup_store().await;
    let entries = store.get_history(Uuid::new_v4()).await.unwrap();
    assert!(entries.is_empty());
}
