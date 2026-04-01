use async_trait::async_trait;
use crate::errors::MemoryError;
use crate::history::traits::HistoryStore;
use crate::models::{EventType, HistoryEntry};
use chrono::{DateTime, Utc};
use rusqlite::{params, Connection};
use std::path::Path;
use std::sync::{Arc, Mutex};
use uuid::Uuid;

pub struct HistoryManager {
    conn: Arc<Mutex<Connection>>,
}

impl HistoryManager {
    pub fn new(path: impl AsRef<Path>) -> Result<Self, MemoryError> {
        // Ensure directory exists
        if let Some(parent) = path.as_ref().parent() {
            std::fs::create_dir_all(parent).map_err(|e| MemoryError::History(e.to_string()))?;
        }

        let conn = Connection::open(path).map_err(|e| MemoryError::History(e.to_string()))?;

        // Create table
        conn.execute(
            "CREATE TABLE IF NOT EXISTS history (
                id TEXT PRIMARY KEY,
                memory_id TEXT NOT NULL,
                previous_content TEXT,
                new_content TEXT NOT NULL,
                event TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                user_id TEXT,
                agent_id TEXT,
                request_id TEXT
            )",
            [],
        ).map_err(|e| MemoryError::History(e.to_string()))?;

        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
        })
    }
}

#[async_trait]
impl HistoryStore for HistoryManager {
    #[allow(clippy::too_many_arguments)]
    async fn add_history(
        &self,
        memory_id: Uuid,
        previous_content: Option<String>,
        new_content: String,
        event: EventType,
        timestamp: DateTime<Utc>,
        user_id: Option<String>,
        agent_id: Option<String>,
        request_id: Option<String>,
    ) -> Result<(), MemoryError> {
        let conn = self.conn.lock().unwrap();
        let id = Uuid::new_v4().to_string();

        // Serialize event enum
        let event_str = serde_json::to_string(&event)
            .map_err(|e| MemoryError::History(format!("Failed to serialize event: {}", e)))?;
        let event_str = event_str.trim_matches('"');

        conn.execute(
            "INSERT INTO history (id, memory_id, previous_content, new_content, event, timestamp, user_id, agent_id, request_id)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            params![
                id,
                memory_id.to_string(),
                previous_content,
                new_content,
                event_str,
                timestamp.to_rfc3339(),
                user_id,
                agent_id,
                request_id,
            ],
        ).map_err(|e| MemoryError::History(e.to_string()))?;

        Ok(())
    }

    async fn get_history(&self, memory_id: Uuid) -> Result<Vec<HistoryEntry>, MemoryError> {
        let conn = self.conn.lock().unwrap();

        let mut stmt = conn.prepare(
            "SELECT id, memory_id, previous_content, new_content, event, timestamp
             FROM history WHERE memory_id = ?1 ORDER BY timestamp DESC"
        ).map_err(|e| MemoryError::History(e.to_string()))?;

        let rows = stmt.query_map(params![memory_id.to_string()], |row| {
            let event_str: String = row.get(4)?;
            let event = match event_str.as_str() {
                "ADD" => EventType::Add,
                "UPDATE" => EventType::Update,
                "DELETE" => EventType::Delete,
                _ => EventType::Noop,
            };

            let timestamp_str: String = row.get(5)?;
            let timestamp = DateTime::parse_from_rfc3339(&timestamp_str)
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or(Utc::now());

            Ok(HistoryEntry {
                id: Uuid::parse_str(&row.get::<_, String>(0)?).unwrap_or_default(),
                memory_id: Uuid::parse_str(&row.get::<_, String>(1)?).unwrap_or_default(),
                previous_content: row.get(2)?,
                new_content: row.get(3)?,
                event,
                timestamp,
            })
        }).map_err(|e| MemoryError::History(e.to_string()))?;

        let mut history = Vec::new();
        for row in rows {
            history.push(row.map_err(|e| MemoryError::History(e.to_string()))?);
        }

        Ok(history)
    }

    async fn reset(&self) -> Result<(), MemoryError> {
        let conn = self.conn.lock().unwrap();
        conn.execute("DELETE FROM history", []).map_err(|e| MemoryError::History(e.to_string()))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::EventType;
    use chrono::Utc;
    use uuid::Uuid;
    use tempfile::tempdir;

    /// Helper: create a HistoryManager backed by a temp SQLite file.
    fn create_test_store() -> HistoryManager {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_history.db");
        // Leak the tempdir so it lives long enough (test-only).
        let path_owned = path.to_path_buf();
        std::mem::forget(dir);
        HistoryManager::new(path_owned).unwrap()
    }

    #[tokio::test]
    async fn test_sqlite_add_and_get_history() {
        let store = create_test_store();
        let memory_id = Uuid::new_v4();
        let now = Utc::now();

        store.add_history(
            memory_id,
            None,
            "user likes Rust".to_string(),
            EventType::Add,
            now,
            Some("user-1".to_string()),
            Some("agent-1".to_string()),
            None,
        ).await.unwrap();

        let entries = store.get_history(memory_id).await.unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].memory_id, memory_id);
        assert_eq!(entries[0].new_content, "user likes Rust");
        assert_eq!(entries[0].event, EventType::Add);
        assert!(entries[0].previous_content.is_none());
    }

    #[tokio::test]
    async fn test_sqlite_update_history_ordering() {
        let store = create_test_store();
        let memory_id = Uuid::new_v4();
        let t1 = Utc::now();

        store.add_history(
            memory_id,
            None,
            "user likes Rust".to_string(),
            EventType::Add,
            t1,
            Some("user-1".to_string()),
            None,
            None,
        ).await.unwrap();

        let t2 = t1 + chrono::Duration::seconds(10);
        store.add_history(
            memory_id,
            Some("user likes Rust".to_string()),
            "user loves Rust".to_string(),
            EventType::Update,
            t2,
            Some("user-1".to_string()),
            None,
            None,
        ).await.unwrap();

        let entries = store.get_history(memory_id).await.unwrap();
        assert_eq!(entries.len(), 2);
        // Newest first (ORDER BY timestamp DESC)
        assert_eq!(entries[0].event, EventType::Update);
        assert_eq!(entries[0].previous_content, Some("user likes Rust".to_string()));
        assert_eq!(entries[0].new_content, "user loves Rust");
        assert_eq!(entries[1].event, EventType::Add);
    }

    #[tokio::test]
    async fn test_sqlite_reset_clears_history() {
        let store = create_test_store();
        let memory_id = Uuid::new_v4();

        store.add_history(
            memory_id,
            None,
            "some content".to_string(),
            EventType::Add,
            Utc::now(),
            None,
            None,
            None,
        ).await.unwrap();

        assert_eq!(store.get_history(memory_id).await.unwrap().len(), 1);

        store.reset().await.unwrap();
        assert!(store.get_history(memory_id).await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_sqlite_get_history_nonexistent_returns_empty() {
        let store = create_test_store();
        let entries = store.get_history(Uuid::new_v4()).await.unwrap();
        assert!(entries.is_empty());
    }
}
