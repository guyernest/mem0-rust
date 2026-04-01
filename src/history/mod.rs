mod sqlite;
mod traits;

pub use sqlite::HistoryManager;
pub use traits::HistoryStore;

use crate::config::HistoryStoreConfig;
use crate::errors::MemoryError;
use std::sync::Arc;

/// Create a history store from configuration.
///
/// Returns `None` for `HistoryStoreConfig::None` (no history tracking).
/// Returns `Some(Arc<dyn HistoryStore>)` for configured backends.
pub fn create_history_store(
    config: &HistoryStoreConfig,
) -> Result<Option<Arc<dyn HistoryStore>>, MemoryError> {
    match config {
        HistoryStoreConfig::SQLite { path } => {
            let manager = HistoryManager::new(path)?;
            Ok(Some(Arc::new(manager)))
        }
        HistoryStoreConfig::None => Ok(None),
    }
}
