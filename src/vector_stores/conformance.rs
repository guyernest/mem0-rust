//! Generic VectorStore trait conformance tests.
//!
//! Any VectorStore implementation should pass `conformance_suite` to prove
//! behavioral consistency with other backends (per D-09).
//!
//! Usage in a backend's test module:
//! ```rust
//! #[tokio::test]
//! async fn test_conformance() {
//!     let store = /* create your store with test/mocked client */;
//!     crate::vector_stores::conformance::conformance_suite(&store).await;
//! }
//! ```

use std::collections::HashMap;
use crate::models::{FilterCondition, FilterLogic, FilterOperator, Filters, Payload};
use crate::vector_stores::traits::VectorStore;
use chrono::Utc;

/// Helper to build a minimal test Payload.
pub fn test_payload(data: &str, user_id: Option<&str>) -> Payload {
    Payload {
        data: data.to_string(),
        hash: format!("hash-{}", data),
        created_at: Utc::now(),
        user_id: user_id.map(String::from),
        agent_id: None,
        request_id: None,
        memory_type: None,
        metadata: HashMap::new(),
    }
}

/// Helper to build a simple equality filter.
pub fn eq_filter(field: &str, value: &str) -> Filters {
    Filters {
        conditions: vec![FilterCondition {
            field: field.to_string(),
            operator: FilterOperator::Eq,
            value: serde_json::Value::String(value.to_string()),
        }],
        logic: FilterLogic::And,
    }
}

/// Run the full conformance suite against the provided store.
///
/// Assumes the store's collection already exists (call `create_collection` before if needed,
/// or pass a store whose constructor auto-creates it).
pub async fn conformance_suite(store: &dyn VectorStore) {
    let embedding = vec![0.1_f32; 4]; // 4-dim test vectors

    // --- insert ---
    let payload_a = test_payload("memory A", Some("user-alice"));
    store
        .insert("id-a", embedding.clone(), payload_a)
        .await
        .expect("conformance: insert id-a should succeed");

    let payload_b = test_payload("memory B", Some("user-bob"));
    store
        .insert("id-b", embedding.clone(), payload_b)
        .await
        .expect("conformance: insert id-b should succeed");

    // --- get: found ---
    let result = store
        .get("id-a")
        .await
        .expect("conformance: get id-a should succeed");
    assert!(result.is_some(), "conformance: get id-a should return Some");
    assert_eq!(
        result.unwrap().payload.data,
        "memory A",
        "conformance: get id-a should have correct data"
    );

    // --- get: not found ---
    let missing = store
        .get("id-missing")
        .await
        .expect("conformance: get missing should succeed (not error)");
    assert!(
        missing.is_none(),
        "conformance: get missing should return None"
    );

    // --- list: no filter ---
    let all = store
        .list(None, 100)
        .await
        .expect("conformance: list with no filter should succeed");
    assert!(
        all.len() >= 2,
        "conformance: list should return at least 2 items"
    );

    // --- list: with filter ---
    let filter = eq_filter("user_id", "user-alice");
    let filtered = store
        .list(Some(&filter), 100)
        .await
        .expect("conformance: list with user_id filter should succeed");
    assert!(
        filtered
            .iter()
            .all(|r| r.payload.user_id.as_deref() == Some("user-alice")),
        "conformance: list with user_id=user-alice should only return alice's memories"
    );

    // --- update: change payload ---
    let updated_payload = test_payload("memory A updated", Some("user-alice"));
    store
        .update("id-a", None, updated_payload)
        .await
        .expect("conformance: update id-a should succeed");

    let after_update = store
        .get("id-a")
        .await
        .expect("conformance: get after update should succeed")
        .expect("conformance: id-a should still exist after update");
    assert_eq!(
        after_update.payload.data,
        "memory A updated",
        "conformance: payload.data should reflect update"
    );

    // --- search ---
    let results = store
        .search(&embedding, 5, None)
        .await
        .expect("conformance: search should succeed");
    assert!(
        !results.is_empty(),
        "conformance: search should return at least one result"
    );

    // --- delete ---
    store
        .delete("id-b")
        .await
        .expect("conformance: delete id-b should succeed");
    let after_delete = store
        .get("id-b")
        .await
        .expect("conformance: get after delete should succeed");
    assert!(
        after_delete.is_none(),
        "conformance: id-b should not exist after delete"
    );

    // --- delete_all with filter ---
    let alice_filter = eq_filter("user_id", "user-alice");
    let deleted_count = store
        .delete_all(Some(&alice_filter))
        .await
        .expect("conformance: delete_all with filter should succeed");
    // Note: some backends return 0 for delete_all even if items were deleted (e.g. index-reset path)
    // so we just assert it doesn't error, not that count is exact.
    let _ = deleted_count;

    // Verify alice's memory is gone
    let after_delete_all = store
        .get("id-a")
        .await
        .expect("conformance: get after delete_all should succeed");
    assert!(
        after_delete_all.is_none(),
        "conformance: id-a should not exist after delete_all(user-alice filter)"
    );
}
