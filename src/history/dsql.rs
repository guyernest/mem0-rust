//! Aurora DSQL history storage backend.
//!
//! Implements the `HistoryStore` trait using Amazon Aurora DSQL
//! (serverless PostgreSQL-compatible) as the storage backend.
//! Designed for Lambda production deployments where scale-to-zero
//! SQL is preferable to file-based SQLite.

use async_trait::async_trait;
use aws_config::Region;
use chrono::{DateTime, Utc};
use sqlx::{postgres::PgPoolOptions, PgPool, Row};
use std::time::Duration;
use uuid::Uuid;

use crate::errors::MemoryError;
use crate::history::traits::HistoryStore;
use crate::models::{EventType, HistoryEntry};

/// History store backed by Amazon Aurora DSQL.
pub struct DsqlHistoryStore {
    pool: PgPool,
}

/// Create an Aurora DSQL connection pool.
///
/// Uses the `aurora-dsql-sqlx-connector` crate to handle IAM-based
/// token authentication automatically.
///
/// # Arguments
/// * `endpoint` — DSQL cluster endpoint (e.g. "abc123.dsql.us-east-1.on.aws")
/// * `region` — AWS region string; if `None`, falls back to the `AWS_REGION`
///   environment variable (handled by the connector).
pub async fn create_dsql_pool(
    endpoint: &str,
    region: Option<&str>,
) -> Result<PgPool, sqlx::Error> {
    use aurora_dsql_sqlx_connector::DsqlConnectOptionsBuilder;
    use sqlx::postgres::PgConnectOptions;

    let base_opts = PgConnectOptions::new()
        .host(endpoint)
        .username("admin")
        .database("postgres");

    let mut builder = DsqlConnectOptionsBuilder::new(base_opts);
    if let Some(r) = region {
        builder = builder.region(Region::new(r.to_string()));
    }
    let opts = builder.build().await?;

    PgPoolOptions::new()
        .max_connections(5)
        .acquire_timeout(Duration::from_secs(30))
        .idle_timeout(Duration::from_secs(300))
        .connect_with(opts)
        .await
}

impl DsqlHistoryStore {
    /// Create a new `DsqlHistoryStore` and ensure the history table exists.
    pub async fn new(pool: PgPool) -> Result<Self, MemoryError> {
        // Create history table (idempotent)
        sqlx::query(
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
        )
        .execute(&pool)
        .await
        .map_err(|e| MemoryError::History(e.to_string()))?;

        // Create index on memory_id for fast lookup
        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_history_memory_id ON history(memory_id)",
        )
        .execute(&pool)
        .await
        .map_err(|e| MemoryError::History(e.to_string()))?;

        Ok(Self { pool })
    }
}

#[async_trait]
impl HistoryStore for DsqlHistoryStore {
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
        let id = Uuid::new_v4().to_string();

        sqlx::query(
            "INSERT INTO history (id, memory_id, previous_content, new_content, event, timestamp, user_id, agent_id, request_id)
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)",
        )
        .bind(id)
        .bind(memory_id.to_string())
        .bind(previous_content)
        .bind(new_content)
        .bind(event.to_string())
        .bind(timestamp.to_rfc3339())
        .bind(user_id)
        .bind(agent_id)
        .bind(request_id)
        .execute(&self.pool)
        .await
        .map_err(|e| MemoryError::History(e.to_string()))?;

        Ok(())
    }

    async fn get_history(&self, memory_id: Uuid) -> Result<Vec<HistoryEntry>, MemoryError> {
        let rows = sqlx::query(
            "SELECT id, memory_id, previous_content, new_content, event, timestamp
             FROM history WHERE memory_id = $1 ORDER BY timestamp DESC",
        )
        .bind(memory_id.to_string())
        .fetch_all(&self.pool)
        .await
        .map_err(|e| MemoryError::History(e.to_string()))?;

        let entries = rows
            .iter()
            .map(|row| {
                let event = match row.get::<String, _>("event").as_str() {
                    "ADD" => EventType::Add,
                    "UPDATE" => EventType::Update,
                    "DELETE" => EventType::Delete,
                    _ => EventType::Noop,
                };

                let timestamp = DateTime::parse_from_rfc3339(
                    &row.get::<String, _>("timestamp"),
                )
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(|_| Utc::now());

                HistoryEntry {
                    id: Uuid::parse_str(&row.get::<String, _>("id")).unwrap_or_default(),
                    memory_id: Uuid::parse_str(&row.get::<String, _>("memory_id"))
                        .unwrap_or_default(),
                    previous_content: row.get::<Option<String>, _>("previous_content"),
                    new_content: row.get::<String, _>("new_content"),
                    event,
                    timestamp,
                }
            })
            .collect();

        Ok(entries)
    }

    async fn reset(&self) -> Result<(), MemoryError> {
        // Per D-10: DSQL does not support TRUNCATE; use DELETE instead
        sqlx::query("DELETE FROM history")
            .execute(&self.pool)
            .await
            .map_err(|e| MemoryError::History(e.to_string()))?;

        Ok(())
    }
}
