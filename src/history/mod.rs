mod sqlite;
mod traits;
#[cfg(feature = "dsql")]
mod dsql;

pub use sqlite::HistoryManager;
pub use traits::HistoryStore;
#[cfg(feature = "dsql")]
pub use dsql::DsqlHistoryStore;
#[cfg(feature = "dsql")]
pub use dsql::create_dsql_pool;

use crate::config::HistoryStoreConfig;
use crate::errors::MemoryError;
use std::sync::Arc;

/// Create a history store from configuration.
///
/// Returns `None` for `HistoryStoreConfig::None` (no history tracking).
/// Returns `Some(Arc<dyn HistoryStore>)` for configured backends.
pub async fn create_history_store(
    config: &HistoryStoreConfig,
) -> Result<Option<Arc<dyn HistoryStore>>, MemoryError> {
    match config {
        HistoryStoreConfig::SQLite { path } => {
            let manager = HistoryManager::new(path)?;
            Ok(Some(Arc::new(manager)))
        }
        #[cfg(feature = "dsql")]
        HistoryStoreConfig::Dsql { endpoint, region } => {
            let pool = dsql::create_dsql_pool(endpoint, region.as_deref()).await?;
            let store = DsqlHistoryStore::new(pool).await?;
            Ok(Some(Arc::new(store)))
        }
        HistoryStoreConfig::None => Ok(None),
    }
}
