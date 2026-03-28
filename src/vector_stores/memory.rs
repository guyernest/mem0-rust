//! In-memory vector store for testing and development.

use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::RwLock;

use super::traits::{VectorSearchResult, VectorStore};
use crate::errors::VectorStoreError;
use crate::models::{Filters, Payload};

/// In-memory vector store entry
struct Entry {
    embedding: Vec<f32>,
    payload: Payload,
}

/// In-memory vector store
pub struct InMemoryStore {
    entries: RwLock<HashMap<String, Entry>>,
}

impl InMemoryStore {
    /// Create a new in-memory store
    pub fn new() -> Self {
        Self {
            entries: RwLock::new(HashMap::new()),
        }
    }

    /// Compute cosine similarity between two vectors
    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() || a.is_empty() {
            return 0.0;
        }

        let mut dot = 0.0f32;
        let mut norm_a = 0.0f32;
        let mut norm_b = 0.0f32;

        for (va, vb) in a.iter().zip(b.iter()) {
            dot += va * vb;
            norm_a += va * va;
            norm_b += vb * vb;
        }

        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }

        dot / (norm_a.sqrt() * norm_b.sqrt())
    }

}

impl Default for InMemoryStore {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl VectorStore for InMemoryStore {
    async fn insert(
        &self,
        id: &str,
        embedding: Vec<f32>,
        payload: Payload,
    ) -> Result<(), VectorStoreError> {
        let mut entries = self
            .entries
            .write()
            .map_err(|e| VectorStoreError::Insert(e.to_string()))?;

        entries.insert(id.to_string(), Entry { embedding, payload });
        Ok(())
    }

    async fn search(
        &self,
        embedding: &[f32],
        limit: usize,
        filters: Option<&Filters>,
    ) -> Result<Vec<VectorSearchResult>, VectorStoreError> {
        let entries = self
            .entries
            .read()
            .map_err(|e| VectorStoreError::Search(e.to_string()))?;

        let mut results: Vec<VectorSearchResult> = entries
            .iter()
            .filter(|(_, entry)| super::filter_eval::matches_filters(&entry.payload, filters))
            .map(|(id, entry)| VectorSearchResult {
                id: id.clone(),
                score: Self::cosine_similarity(embedding, &entry.embedding),
                payload: entry.payload.clone(),
            })
            .collect();

        // Sort by score descending
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(limit);

        Ok(results)
    }

    async fn get(&self, id: &str) -> Result<Option<VectorSearchResult>, VectorStoreError> {
        let entries = self
            .entries
            .read()
            .map_err(|e| VectorStoreError::Search(e.to_string()))?;

        Ok(entries.get(id).map(|entry| VectorSearchResult {
            id: id.to_string(),
            score: 1.0,
            payload: entry.payload.clone(),
        }))
    }

    async fn delete(&self, id: &str) -> Result<(), VectorStoreError> {
        let mut entries = self
            .entries
            .write()
            .map_err(|e| VectorStoreError::Delete(e.to_string()))?;

        entries
            .remove(id)
            .ok_or_else(|| VectorStoreError::NotFound(id.to_string()))?;

        Ok(())
    }

    async fn update(
        &self,
        id: &str,
        embedding: Option<Vec<f32>>,
        payload: Payload,
    ) -> Result<(), VectorStoreError> {
        let mut entries = self
            .entries
            .write()
            .map_err(|e| VectorStoreError::Update(e.to_string()))?;

        let entry = entries
            .get_mut(id)
            .ok_or_else(|| VectorStoreError::NotFound(id.to_string()))?;

        if let Some(emb) = embedding {
            entry.embedding = emb;
        }
        entry.payload = payload;

        Ok(())
    }

    async fn list(
        &self,
        filters: Option<&Filters>,
        limit: usize,
    ) -> Result<Vec<VectorSearchResult>, VectorStoreError> {
        let entries = self
            .entries
            .read()
            .map_err(|e| VectorStoreError::Search(e.to_string()))?;

        let results: Vec<VectorSearchResult> = entries
            .iter()
            .filter(|(_, entry)| super::filter_eval::matches_filters(&entry.payload, filters))
            .take(limit)
            .map(|(id, entry)| VectorSearchResult {
                id: id.clone(),
                score: 1.0,
                payload: entry.payload.clone(),
            })
            .collect();

        Ok(results)
    }

    async fn delete_all(&self, filters: Option<&Filters>) -> Result<usize, VectorStoreError> {
        let mut entries = self
            .entries
            .write()
            .map_err(|e| VectorStoreError::Delete(e.to_string()))?;

        let to_delete: Vec<String> = entries
            .iter()
            .filter(|(_, entry)| super::filter_eval::matches_filters(&entry.payload, filters))
            .map(|(id, _)| id.clone())
            .collect();

        let count = to_delete.len();
        for id in to_delete {
            entries.remove(&id);
        }

        Ok(count)
    }

    async fn collection_exists(&self) -> Result<bool, VectorStoreError> {
        Ok(true) // In-memory store always "exists"
    }

    async fn create_collection(&self) -> Result<(), VectorStoreError> {
        Ok(()) // No-op for in-memory store
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use std::collections::HashMap;

    /// Run the shared VectorStore conformance suite against InMemoryStore.
    ///
    /// InMemoryStore is a real in-process implementation — no mocking needed.
    /// Uses 4-dim vectors to match the conformance suite's embedding size.
    #[tokio::test]
    async fn test_conformance_suite() {
        let store = InMemoryStore::new();
        // InMemoryStore::create_collection is a no-op, but call it to satisfy
        // the conformance contract that the collection exists before use.
        store
            .create_collection()
            .await
            .expect("create_collection should succeed");
        crate::vector_stores::conformance::conformance_suite(&store).await;
    }

    fn create_test_payload(data: &str) -> Payload {
        Payload {
            data: data.to_string(),
            hash: "test_hash".to_string(),
            created_at: Utc::now(),
            user_id: None,
            agent_id: None,
            run_id: None,
            memory_type: None,
            metadata: HashMap::new(),
        }
    }

    #[tokio::test]
    async fn test_insert_and_get() {
        let store = InMemoryStore::new();
        let payload = create_test_payload("test content");
        let embedding = vec![0.1, 0.2, 0.3];

        store.insert("test-id", embedding, payload).await.unwrap();

        let result = store.get("test-id").await.unwrap();
        assert!(result.is_some());
        assert_eq!(result.unwrap().payload.data, "test content");
    }

    #[tokio::test]
    async fn test_search() {
        let store = InMemoryStore::new();

        store
            .insert("id1", vec![1.0, 0.0, 0.0], create_test_payload("doc1"))
            .await
            .unwrap();
        store
            .insert("id2", vec![0.0, 1.0, 0.0], create_test_payload("doc2"))
            .await
            .unwrap();

        let results = store.search(&[1.0, 0.0, 0.0], 10, None).await.unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, "id1"); // Most similar
    }

    #[tokio::test]
    async fn test_delete() {
        let store = InMemoryStore::new();
        store
            .insert("id1", vec![1.0], create_test_payload("doc1"))
            .await
            .unwrap();

        store.delete("id1").await.unwrap();
        assert!(store.get("id1").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn test_memory_type_filter() {
        use crate::models::{FilterCondition, FilterLogic, FilterOperator, MemoryType};

        let store = InMemoryStore::new();

        // Insert semantic memory for user-1
        let mut p1 = create_test_payload("semantic fact");
        p1.user_id = Some("user-1".to_string());
        p1.memory_type = Some(MemoryType::Semantic);
        store.insert("id-s1", vec![1.0, 0.0, 0.0], p1).await.unwrap();

        // Insert episodic memory for user-1 with run_id (session-scoped)
        let mut p2 = create_test_payload("episode event");
        p2.user_id = Some("user-1".to_string());
        p2.run_id = Some("run-abc".to_string());
        p2.memory_type = Some(MemoryType::Episodic);
        store.insert("id-e1", vec![0.0, 1.0, 0.0], p2).await.unwrap();

        // Insert memory with no type for user-1
        let mut p3 = create_test_payload("untyped memory");
        p3.user_id = Some("user-1".to_string());
        store.insert("id-u1", vec![0.0, 0.0, 1.0], p3).await.unwrap();

        // Filter: memory_type = semantic_memory
        let semantic_filter = crate::vector_stores::conformance::eq_filter("memory_type", "semantic_memory");
        let results = store.list(Some(&semantic_filter), 100).await.unwrap();
        assert_eq!(results.len(), 1, "only semantic memories");
        assert_eq!(results[0].id, "id-s1");

        // Filter: memory_type = episodic_memory
        let episodic_filter = crate::vector_stores::conformance::eq_filter("memory_type", "episodic_memory");
        let results = store.list(Some(&episodic_filter), 100).await.unwrap();
        assert_eq!(results.len(), 1, "only episodic memories");
        assert_eq!(results[0].id, "id-e1");

        // Filter: user_id + memory_type combined (scoping)
        let combined = Filters {
            conditions: vec![
                FilterCondition {
                    field: "user_id".to_string(),
                    operator: FilterOperator::Eq,
                    value: serde_json::Value::String("user-1".to_string()),
                },
                FilterCondition {
                    field: "memory_type".to_string(),
                    operator: FilterOperator::Eq,
                    value: serde_json::Value::String("episodic_memory".to_string()),
                },
            ],
            logic: FilterLogic::And,
        };
        let results = store.list(Some(&combined), 100).await.unwrap();
        assert_eq!(results.len(), 1, "user-1 episodic only");
        assert_eq!(results[0].id, "id-e1");

        // Search with memory_type filter
        let results = store.search(&[1.0, 0.0, 0.0], 10, Some(&semantic_filter)).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "id-s1");
    }
}
