//! HistoryStore trait definition.

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use uuid::Uuid;
use crate::errors::MemoryError;
use crate::models::{EventType, HistoryEntry};

/// Trait for pluggable history storage backends.
///
/// Implementations track memory change history (add, update, delete events).
/// The trait is async to support both local (SQLite) and remote (DSQL) backends.
#[async_trait]
pub trait HistoryStore: Send + Sync {
    /// Record a history event for a memory operation.
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
    ) -> Result<(), MemoryError>;

    /// Retrieve history entries for a specific memory, ordered by timestamp descending.
    async fn get_history(&self, memory_id: Uuid) -> Result<Vec<HistoryEntry>, MemoryError>;

    /// Delete all history entries.
    async fn reset(&self) -> Result<(), MemoryError>;
}
