//! Core Memory manager.

use std::collections::HashMap;
use std::sync::Arc;
use tracing::{debug, info, warn};
use uuid::Uuid;
use chrono::Utc;

use crate::config::MemoryConfig;
use crate::embeddings::{create_embedder, Embedder};
use crate::errors::{LLMError, MemoryError};
use crate::history::HistoryManager;
use crate::llms::{create_llm, generate_json, GenerateOptions, LLM};
use crate::models::{
    AddOptions, AddResult, DeleteOptions, EventType, FilterCondition, FilterLogic, FilterOperator,
    Filters, GetAllOptions, HistoryEntry, MemoryEvent, MemoryRecord, MemoryType, Message, Messages,
    Payload, ResetOptions, Role, ScoredMemory, SearchOptions, SearchResult, UpdateOptions,
    REQUEST_ID_FIELD,
};
use crate::vector_stores::filter_eval::matches_filters;
use crate::vector_stores::{create_vector_store, VectorStore};
use crate::rerankers::{create_reranker, Reranker};

use super::prompts::{
    format_fact_extraction_input, format_memory_update_input, should_use_agent_extraction,
    AGENT_FACT_EXTRACTION_PROMPT, FACT_EXTRACTION_PROMPT, MEMORY_UPDATE_PROMPT,
};

/// Main Memory interface
pub struct Memory {
    embedder: Arc<dyn Embedder>,
    vector_store: Arc<dyn VectorStore>,
    llm: Option<Arc<dyn LLM>>,
    history: Option<Arc<HistoryManager>>,
    reranker: Option<Arc<dyn Reranker>>,
    #[allow(dead_code)]
    config: MemoryConfig,
}

impl Memory {
    /// Create a new Memory instance
    pub async fn new(config: MemoryConfig) -> Result<Self, MemoryError> {
        let embedder = create_embedder(&config.embedder)?;
        let dimensions = embedder.dimensions();

        let vector_store =
            create_vector_store(&config.vector_store, &config.collection_name, dimensions).await?;

        let llm = if let Some(llm_config) = &config.llm {
            Some(create_llm(llm_config)?)
        } else {
            None
        };

        let history = if let Some(path) = &config.history_db_path {
            Some(Arc::new(HistoryManager::new(path)?))
        } else {
            None
        };

        let reranker = if let Some(reranker_config) = &config.reranker {
            Some(create_reranker(reranker_config)?)
        } else {
            None
        };

        info!(
            "Initialized Memory with {} embedder, {} dimensions",
            embedder.model_name(),
            dimensions
        );

        Ok(Self {
            embedder,
            vector_store,
            llm,
            history,
            reranker,
            config,
        })
    }

    /// Add memories from messages
    pub async fn add(
        &self,
        messages: impl Into<Messages>,
        options: AddOptions,
    ) -> Result<AddResult, MemoryError> {
        let messages = messages.into().into_messages();
        // Validate scoping
        if options.user_id.is_none() && options.agent_id.is_none() && options.request_id.is_none() {
            return Err(MemoryError::InvalidInput(
                "At least one of user_id, agent_id, or request_id is required".to_string(),
            ));
        }

        let results = if options.infer && self.llm.is_some() {
            // Use LLM for fact extraction
            self.add_with_inference(&messages, &options).await?
        } else {
            // Add messages directly without inference
            self.add_raw(&messages, &options).await?
        };

        Ok(AddResult { results })
    }

    /// Add messages directly without LLM inference
    async fn add_raw(
        &self,
        messages: &[Message],
        options: &AddOptions,
    ) -> Result<Vec<MemoryEvent>, MemoryError> {
        let mut results = Vec::new();

        for msg in messages {
            if msg.role == Role::System {
                continue;
            }

            let mut record = MemoryRecord::with_scoping(
                msg.content.clone(),
                options.metadata_value(),
                options.user_id.clone(),
                options.agent_id.clone(),
                options.request_id.clone(),
            );
            if let Some(mt) = options.memory_type {
                record.memory_type = Some(mt);
            }

            let embedding = self.embedder.embed(&record.content).await?;
            let payload = Payload::from(&record);

            self.vector_store
                .insert(&record.id.to_string(), embedding, payload)
                .await?;

            if let Some(history) = &self.history {
                let _ = history.add_history(
                    record.id,
                    None,
                    record.content.clone(),
                    EventType::Add,
                    record.created_at,
                    record.user_id.clone(),
                    record.agent_id.clone(),
                    record.request_id.clone(),
                );
            }

            results.push(MemoryEvent {
                id: record.id,
                memory: record.content,
                event: EventType::Add,
                previous_memory: None,
            });
        }

        Ok(results)
    }

    /// Add messages with LLM inference
    async fn add_with_inference(
        &self,
        messages: &[Message],
        options: &AddOptions,
    ) -> Result<Vec<MemoryEvent>, MemoryError> {
        let llm = self.llm.as_ref().ok_or(LLMError::NotConfigured)?;

        // Embedding cache: reuse embeddings for duplicate texts within this add() call (OPS-06)
        let mut embedding_cache: HashMap<String, Vec<f32>> = HashMap::new();

        // Format messages for extraction
        let messages_text = messages
            .iter()
            .map(|m| format!("{:?}: {}", m.role, m.content))
            .collect::<Vec<_>>()
            .join("\n");

        // Select prompt based on agent_id presence and assistant messages (OPS-04)
        let extraction_prompt = if should_use_agent_extraction(messages, options.agent_id.as_deref()) {
            AGENT_FACT_EXTRACTION_PROMPT
        } else {
            FACT_EXTRACTION_PROMPT
        };

        // Extract facts
        let extraction_messages = vec![
            Message::system(extraction_prompt),
            Message::user(format_fact_extraction_input(&messages_text)),
        ];

        #[derive(serde::Deserialize)]
        struct FactsResponse {
            facts: Vec<String>,
        }

        let facts: FactsResponse = generate_json(
            llm.as_ref(),
            &extraction_messages,
            GenerateOptions::default(),
        )
        .await?;

        if facts.facts.is_empty() {
            debug!("No facts extracted from messages");
            return Ok(Vec::new());
        }

        info!("Extracted {} facts", facts.facts.len());

        // Search for existing related memories
        let mut existing_memories: Vec<(String, String)> = Vec::new(); // (Index, Content)
        let mut memory_map: HashMap<String, String> = HashMap::new(); // Index -> RealID

        let search_filters = build_scope_filters(
            options.user_id.as_deref(),
            options.agent_id.as_deref(),
            options.request_id.as_deref(),
            options.memory_type,
        );

        for fact in &facts.facts {
            let embedding = cached_embed(self.embedder.as_ref(), &mut embedding_cache, fact).await?;

            let similar = self
                .vector_store
                .search(&embedding, 5, search_filters.as_ref())
                .await?;

            for result in similar {
                // Check if we already have this memory in our list (dedupe by real ID)
                let real_id = result.id.clone();
                if !memory_map.values().any(|rid| rid == &real_id) {
                     let index = memory_map.len().to_string();
                     memory_map.insert(index.clone(), real_id);
                     existing_memories.push((index, result.payload.data));
                }
            }
        }

        // Determine memory actions
        let update_messages = vec![
            Message::system(MEMORY_UPDATE_PROMPT),
            Message::user(format_memory_update_input(&existing_memories, &facts.facts)),
        ];

        #[derive(serde::Deserialize)]
        struct MemoryAction {
            event: String,
            text: Option<String>,
            id: Option<String>,
        }

        #[derive(serde::Deserialize)]
        struct MemoryActionsResponse {
            memory: Vec<MemoryAction>,
        }

        let actions: MemoryActionsResponse = generate_json(
            llm.as_ref(),
            &update_messages,
            GenerateOptions::default(),
        )
        .await?;

        let mut results = Vec::new();

        for action in actions.memory {
            match action.event.to_uppercase().as_str() {
                "ADD" => {
                    if let Some(text) = action.text {
                        let mut record = MemoryRecord::with_scoping(
                            &text,
                            options
                                .metadata
                                .as_ref()
                                .map(|m| serde_json::to_value(m).unwrap_or_default())
                                .unwrap_or(serde_json::Value::Object(serde_json::Map::new())),
                            options.user_id.clone(),
                            options.agent_id.clone(),
                            options.request_id.clone(),
                        );
                        if let Some(mt) = options.memory_type {
                            record.memory_type = Some(mt);
                        }

                        let embedding = cached_embed(self.embedder.as_ref(), &mut embedding_cache, &text).await?;
                        let payload = Payload::from(&record);

                        self.vector_store
                            .insert(&record.id.to_string(), embedding, payload)
                            .await?;

                        if let Some(history) = &self.history {
                            let _ = history.add_history(
                                record.id,
                                None,
                                record.content.clone(),
                                EventType::Add,
                                record.created_at,
                                record.user_id.clone(),
                                record.agent_id.clone(),
                                record.request_id.clone(),
                            );
                        }

                        results.push(MemoryEvent {
                            id: record.id,
                            memory: text,
                            event: EventType::Add,
                            previous_memory: None,
                        });
                    }
                }
                "UPDATE" => {
                    if let (Some(index_id), Some(text)) = (action.id, action.text) {
                        if let Some(real_id) = memory_map.get(&index_id).cloned() {
                            debug!("Updating memory {} (index {}) with: {}", real_id, index_id, text);

                            // Capture old content for previous_memory (OPS-02)
                            let old_content = existing_memories
                                .iter()
                                .find(|(idx, _)| idx == &index_id)
                                .map(|(_, content)| content.clone());

                            // Perform update via self.update() which handles history
                            let update_options = UpdateOptions {
                                user_id: options.user_id.clone(),
                                agent_id: options.agent_id.clone(),
                                request_id: options.request_id.clone(),
                            };
                            match self.update(&real_id, &text, update_options).await {
                                Ok(record) => {
                                    results.push(MemoryEvent {
                                        id: record.id,
                                        memory: text.clone(),
                                        event: EventType::Update,
                                        previous_memory: old_content,
                                    });
                                    // Note: self.update() computes its own embedding internally;
                                    // we cannot retrieve it to warm the cache here.
                                },
                                Err(e) => {
                                    warn!("Failed to update memory {}: {}", real_id, e);
                                }
                            }
                        } else {
                            warn!("LLM tried to update unknown memory index: {}", index_id);
                        }
                    }
                }
                "DELETE" => {
                    if let Some(index_id) = action.id {
                        if let Some(real_id) = memory_map.get(&index_id) {
                            debug!("Deleting memory {} (index {})", real_id, index_id);
                            
                            // Perform delete
                            let delete_options = DeleteOptions {
                                user_id: options.user_id.clone(),
                                agent_id: options.agent_id.clone(),
                                request_id: options.request_id.clone(),
                            };
                            match self.delete(real_id, delete_options).await {
                                Ok(_) => {
                                     // ID is needed for event, but delete returns void.
                                     // We can use Uuid::parse_str(real_id)
                                     if let Ok(uuid) = Uuid::parse_str(real_id) {
                                         results.push(MemoryEvent {
                                            id: uuid,
                                            memory: String::new(), // Deleted
                                            event: EventType::Delete,
                                            previous_memory: None,
                                        });
                                     }
                                },
                                Err(e) => {
                                    warn!("Failed to delete memory {}: {}", real_id, e);
                                }
                            }
                        } else {
                            warn!("LLM tried to delete unknown memory index: {}", index_id);
                        }
                    }
                }
                "NOOP" => {
                    // Update session IDs (agent_id, request_id) on the matching memory without
                    // re-embedding. This propagates session context on repeated encounters (OPS-01).
                    if let Some(index_id) = action.id {
                        if let Some(real_id) = memory_map.get(&index_id) {
                            if options.agent_id.is_some() || options.request_id.is_some() {
                                match self.vector_store.get(real_id).await {
                                    Ok(Some(existing)) => {
                                        // PRIV-04: Check user_id ownership before mutating session IDs (per D-07)
                                        if !is_owner(existing.payload.user_id.as_deref(), options.user_id.as_deref()) {
                                            warn!(
                                                "Skipping NOOP session ID update for memory {} \
                                                 (owned by user_id={:?}, caller has user_id={:?})",
                                                real_id, existing.payload.user_id, options.user_id
                                            );
                                            continue;
                                        }
                                        let mut payload = existing.payload;
                                        let mut changed = false;
                                        if let Some(ref aid) = options.agent_id {
                                            if payload.agent_id.as_ref() != Some(aid) {
                                                payload.agent_id = Some(aid.clone());
                                                changed = true;
                                            }
                                        }
                                        if let Some(ref rid) = options.request_id {
                                            if payload.request_id.as_ref() != Some(rid) {
                                                payload.request_id = Some(rid.clone());
                                                changed = true;
                                            }
                                        }
                                        if changed {
                                            match self.vector_store.update(real_id, None, payload).await {
                                                Ok(_) => debug!("Updated session IDs for memory {}", real_id),
                                                Err(e) => warn!("Failed to update session IDs for memory {}: {}", real_id, e),
                                            }
                                        }
                                    }
                                    Ok(None) => warn!("Memory {} not found for session ID update, skipping", real_id),
                                    Err(e) => warn!("Failed to fetch memory {} for session ID update: {}", real_id, e),
                                }
                            } else {
                                debug!("No action needed for memory index {}", index_id);
                            }
                            // No history record for NONE metadata-only updates (D-02)
                        }
                    } else {
                        debug!("No action needed");
                    }
                }
                _ => {
                    warn!("Unknown memory action: {}", action.event);
                }
            }
        }

        Ok(results)
    }

    /// Search for memories
    pub async fn search(
        &self,
        query: &str,
        options: SearchOptions,
    ) -> Result<SearchResult, MemoryError> {
        let embedding = self.embedder.embed(query).await?;
        let limit = options.limit.unwrap_or(10);
        let threshold = options.threshold.unwrap_or(0.0);

        // Fetch more candidates if reranking is enabled
        let search_limit = if options.rerank { limit * 10 } else { limit * 2 };

        let scope_filters = build_scope_filters(
            options.user_id.as_deref(),
            options.agent_id.as_deref(),
            options.request_id.as_deref(),
            options.memory_type,
        );

        // Merge: if user provided custom filters, combine with scope filters
        let effective_filters = match (scope_filters, options.filters.as_ref()) {
            (Some(scope), Some(custom)) => {
                let mut combined = scope.conditions;
                combined.extend(custom.conditions.iter().cloned());
                Some(Filters {
                    conditions: combined,
                    logic: FilterLogic::And,
                })
            }
            (Some(scope), None) => Some(scope),
            (None, Some(custom)) => Some(custom.clone()),
            (None, None) => None,
        };

        let results = self
            .vector_store
            .search(&embedding, search_limit, effective_filters.as_ref())
            .await?;

        // PRIV-01 + PRIV-05: Fused post-filter for privacy and Contains/IContains.
        // Privacy filter only matters when caller has no user_id (agent-only queries);
        // when user_id is provided, build_scope_filters already scoped server-side.
        let caller_uid = options.user_id.as_deref();
        let needs_client_filter = needs_client_side_filter(effective_filters.as_ref());
        let mut scored: Vec<ScoredMemory> = results
            .into_iter()
            .filter(|r| {
                // Privacy: if memory has user_id, caller must match
                if caller_uid.is_none() {
                    if r.payload.user_id.is_some() {
                        return false;
                    }
                }
                true
            })
            .filter(|r| {
                if needs_client_filter {
                    matches_filters(&r.payload, effective_filters.as_ref())
                } else {
                    true
                }
            })
            .map(|r| r.to_scored_memory())
            .collect();

        // Filter by threshold before reranking (optional, but saves rerank quota)
        scored.retain(|m| m.score >= threshold);

        // Reranking
        if options.rerank {
            if let Some(reranker) = &self.reranker {
                scored = reranker.rerank(query, scored).await?;
            } else {
                 warn!("Reranking requested but no reranker configured");
            }
        }
        
        // Final sort and limit
        scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(limit);

        Ok(SearchResult { results: scored })
    }

    /// Get a memory by ID
    pub async fn get(&self, id: &str) -> Result<Option<MemoryRecord>, MemoryError> {
        let result = self.vector_store.get(id).await?;
        Ok(result.map(|r| r.to_memory_record()))
    }

    /// Get all memories
    pub async fn get_all(&self, options: GetAllOptions) -> Result<Vec<MemoryRecord>, MemoryError> {
        let limit = options.limit.unwrap_or(100);
        let scope_filters = build_scope_filters(
            options.user_id.as_deref(),
            options.agent_id.as_deref(),
            options.request_id.as_deref(),
            options.memory_type,
        );
        let results = self.vector_store.list(scope_filters.as_ref(), limit).await?;

        // PRIV-01: Post-filter for user-scoped privacy (per D-04)
        // When caller provides user_id, scope filter already handles this server-side.
        let caller_uid = options.user_id.as_deref();
        let records: Vec<MemoryRecord> = results
            .into_iter()
            .filter(|r| {
                if caller_uid.is_none() && r.payload.user_id.is_some() {
                    return false;
                }
                true
            })
            .map(|r| r.to_memory_record())
            .collect();

        Ok(records)
    }

    /// Update a memory
    pub async fn update(&self, id: &str, content: &str, options: UpdateOptions) -> Result<MemoryRecord, MemoryError> {
        // Get existing record
        let existing = self
            .vector_store
            .get(id)
            .await?
            .ok_or_else(|| MemoryError::NotFound(id.to_string()))?;

        // PRIV-02: Validate caller ownership before allowing update (per D-06)
        validate_ownership(id, existing.payload.user_id.as_deref(), options.user_id.as_deref())?;

        let mut record = existing.to_memory_record();
        let previous_content = record.content.clone();
        record.update_content(content);

        let embedding = self.embedder.embed(content).await?;
        let payload = Payload::from(&record);

        self.vector_store
            .update(id, Some(embedding), payload)
            .await?;

        if let Some(history) = &self.history {
            let _ = history.add_history(
                record.id,
                Some(previous_content),
                record.content.clone(),
                EventType::Update,
                Utc::now(),
                record.user_id.clone(),
                record.agent_id.clone(),
                record.request_id.clone(),
            );
        }

        Ok(record)
    }

    /// Delete a memory
    pub async fn delete(&self, id: &str, options: DeleteOptions) -> Result<(), MemoryError> {
        // Get record first for history
        let record = self.get(id).await?;

        // PRIV-03: Validate caller ownership before allowing deletion (per D-06)
        if let Some(ref record) = record {
            validate_ownership(id, record.user_id.as_deref(), options.user_id.as_deref())?;
        }

        self.vector_store.delete(id).await?;

        if let Some(record) = record {
            if let Some(history) = &self.history {
                let _ = history.add_history(
                    record.id,
                    Some(record.content),
                    "DELETED".to_string(),
                    EventType::Delete,
                    Utc::now(),
                    record.user_id,
                    record.agent_id,
                    record.request_id,
                );
            }
        }
        
        Ok(())
    }

    /// Get memory history
    pub async fn history(&self, id: &str) -> Result<Vec<HistoryEntry>, MemoryError> {
        if let Some(history) = &self.history {
            let memory_id = Uuid::parse_str(id).map_err(|e| MemoryError::InvalidInput(e.to_string()))?;
            history.get_history(memory_id)
        } else {
            Ok(Vec::new())
        }
    }

    /// Reset all memories
    pub async fn reset(&self, options: ResetOptions) -> Result<(), MemoryError> {
        // Build filters based on options
        let filters = if options.user_id.is_some() || options.agent_id.is_some() || options.request_id.is_some() {
            build_scope_filters(
                options.user_id.as_deref(),
                options.agent_id.as_deref(),
                options.request_id.as_deref(),
                None,
            )
        } else {
            None
        };

        self.vector_store.delete_all(filters.as_ref()).await?;
        
        if let Some(history) = &self.history {
            // If global reset, clear history too
            if filters.is_none() {
                history.reset()?;
            }
        }
        
        Ok(())
    }
}

/// Embed text using the cache to avoid duplicate API calls within a single add() call.
async fn cached_embed(
    embedder: &dyn crate::embeddings::Embedder,
    cache: &mut HashMap<String, Vec<f32>>,
    text: &str,
) -> Result<Vec<f32>, MemoryError> {
    if let Some(cached) = cache.get(text) {
        return Ok(cached.clone());
    }
    let emb = embedder.embed(text).await?;
    cache.insert(text.to_string(), emb.clone());
    Ok(emb)
}

/// Build a Filters struct from scoping options (D-07).
///
/// Adds FilterCondition(Eq) for each Some field: user_id, agent_id, request_id, memory_type.
/// Returns None if no conditions are present.
fn build_scope_filters(
    user_id: Option<&str>,
    agent_id: Option<&str>,
    request_id: Option<&str>,
    memory_type: Option<MemoryType>,
) -> Option<Filters> {
    let mut conditions = Vec::new();

    if let Some(uid) = user_id {
        conditions.push(FilterCondition {
            field: "user_id".to_string(),
            operator: FilterOperator::Eq,
            value: serde_json::Value::String(uid.to_string()),
        });
    }
    if let Some(aid) = agent_id {
        conditions.push(FilterCondition {
            field: "agent_id".to_string(),
            operator: FilterOperator::Eq,
            value: serde_json::Value::String(aid.to_string()),
        });
    }
    if let Some(rid) = request_id {
        conditions.push(FilterCondition {
            field: REQUEST_ID_FIELD.to_string(),
            operator: FilterOperator::Eq,
            value: serde_json::Value::String(rid.to_string()),
        });
    }
    if let Some(mt) = memory_type {
        conditions.push(FilterCondition {
            field: "memory_type".to_string(),
            operator: FilterOperator::Eq,
            value: serde_json::Value::String(mt.to_string()),
        });
    }

    if conditions.is_empty() {
        None
    } else {
        Some(Filters {
            conditions,
            logic: FilterLogic::And,
        })
    }
}

/// Check if caller owns a memory based on user_id.
/// Returns true if memory has no user_id (anyone can access) or caller's user_id matches.
fn is_owner(memory_user_id: Option<&str>, caller_user_id: Option<&str>) -> bool {
    match memory_user_id {
        Some(uid) => caller_user_id == Some(uid),
        None => true,
    }
}

/// Validate that the caller has ownership of the memory (PRIV-02/PRIV-03, per D-06).
/// Returns Err(Unauthorized) if the memory has user_id and caller doesn't match.
fn validate_ownership(
    memory_id: &str,
    memory_user_id: Option<&str>,
    caller_user_id: Option<&str>,
) -> Result<(), MemoryError> {
    if is_owner(memory_user_id, caller_user_id) {
        Ok(())
    } else {
        Err(MemoryError::Unauthorized {
            memory_id: memory_id.to_string(),
            reason: format!(
                "caller's user_id ({:?}) does not match memory's user_id ({:?})",
                caller_user_id, memory_user_id
            ),
        })
    }
}

/// Check if the filter set contains any operators that may need client-side evaluation
/// (Contains, IContains are not supported by all backends, notably S3 Vectors).
///
/// Per Pitfall 5 from research: if logic is OR with mixed supported/unsupported operators,
/// we must evaluate ALL conditions client-side. For AND logic, we can safely re-apply
/// just the unsupported conditions, but for simplicity we re-apply all conditions since
/// matches_filters is cheap and idempotent.
fn needs_client_side_filter(filters: Option<&Filters>) -> bool {
    filters.map_or(false, |f| {
        f.conditions.iter().any(|c| {
            matches!(
                c.operator,
                FilterOperator::Contains | FilterOperator::IContains
            )
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_memory_creation() {
        let config = MemoryConfig::default();
        let memory = Memory::new(config).await;
        assert!(memory.is_ok());
    }

    #[tokio::test]
    async fn test_add_raw() {
        let config = MemoryConfig::default();
        let memory = Memory::new(config).await.unwrap();

        let result = memory
            .add(
                "Test memory content",
                AddOptions {
                    user_id: Some("test_user".to_string()),
                    infer: false,
                    ..Default::default()
                },
            )
            .await;

        assert!(result.is_ok());
        assert_eq!(result.unwrap().results.len(), 1);
    }

    #[tokio::test]
    async fn test_search() {
        let config = MemoryConfig::default();
        let memory = Memory::new(config).await.unwrap();

        // Add a memory
        memory
            .add(
                "I love programming in Rust",
                AddOptions {
                    user_id: Some("test_user".to_string()),
                    infer: false,
                    ..Default::default()
                },
            )
            .await
            .unwrap();

        // Search for it
        let results = memory
            .search(
                "Rust programming",
                SearchOptions {
                    user_id: Some("test_user".to_string()),
                    ..Default::default()
                },
            )
            .await
            .unwrap();

        assert!(!results.results.is_empty());
    }
}

#[cfg(test)]
mod privacy_tests {
    use super::*;
    use crate::errors::MemoryError;

    // ==========================================
    // TST-05: Privacy enforcement tests
    // ==========================================

    #[tokio::test]
    async fn test_search_privacy_user_scoped_invisible_to_agent_only() {
        // PRIV-01: User-scoped memories invisible to agent-only queries
        let config = MemoryConfig::default();
        let memory = Memory::new(config).await.unwrap();

        // Add a user-scoped memory
        memory
            .add(
                "Alice secret preference",
                AddOptions {
                    user_id: Some("alice".to_string()),
                    infer: false,
                    ..Default::default()
                },
            )
            .await
            .unwrap();

        // Search WITHOUT user_id (agent-only query)
        let results = memory
            .search(
                "preference",
                SearchOptions {
                    agent_id: Some("agent-1".to_string()),
                    ..Default::default()
                },
            )
            .await
            .unwrap();

        assert!(
            results.results.is_empty(),
            "User-scoped memories must be invisible to agent-only queries"
        );
    }

    #[tokio::test]
    async fn test_search_privacy_cross_user_isolation() {
        // PRIV-01: Alice's memories invisible to Bob
        let config = MemoryConfig::default();
        let memory = Memory::new(config).await.unwrap();

        memory
            .add(
                "Alice likes Rust",
                AddOptions {
                    user_id: Some("alice".to_string()),
                    infer: false,
                    ..Default::default()
                },
            )
            .await
            .unwrap();

        let results = memory
            .search(
                "Rust",
                SearchOptions {
                    user_id: Some("bob".to_string()),
                    ..Default::default()
                },
            )
            .await
            .unwrap();

        assert!(
            results.results.is_empty(),
            "Alice's memories must be invisible to Bob"
        );
    }

    #[tokio::test]
    async fn test_search_privacy_same_user_visible() {
        // PRIV-01: Alice's memories visible to Alice
        let config = MemoryConfig::default();
        let memory = Memory::new(config).await.unwrap();

        memory
            .add(
                "Alice likes Rust",
                AddOptions {
                    user_id: Some("alice".to_string()),
                    infer: false,
                    ..Default::default()
                },
            )
            .await
            .unwrap();

        let results = memory
            .search(
                "Rust",
                SearchOptions {
                    user_id: Some("alice".to_string()),
                    ..Default::default()
                },
            )
            .await
            .unwrap();

        assert!(
            !results.results.is_empty(),
            "Alice's memories must be visible to Alice"
        );
    }

    #[tokio::test]
    async fn test_get_all_privacy_filter() {
        // PRIV-01: get_all with no user_id hides user-scoped memories (per D-04)
        let config = MemoryConfig::default();
        let memory = Memory::new(config).await.unwrap();

        memory
            .add(
                "Alice private data",
                AddOptions {
                    user_id: Some("alice".to_string()),
                    infer: false,
                    ..Default::default()
                },
            )
            .await
            .unwrap();

        let results = memory
            .get_all(GetAllOptions {
                agent_id: Some("agent-1".to_string()),
                ..Default::default()
            })
            .await
            .unwrap();

        assert!(
            results.is_empty(),
            "get_all without user_id must not return user-scoped memories"
        );
    }

    #[tokio::test]
    async fn test_no_user_memory_visible_to_all() {
        // Memories without user_id are visible to everyone
        let config = MemoryConfig::default();
        let memory = Memory::new(config).await.unwrap();

        memory
            .add(
                "Shared knowledge",
                AddOptions {
                    agent_id: Some("agent-1".to_string()),
                    infer: false,
                    ..Default::default()
                },
            )
            .await
            .unwrap();

        let results = memory
            .search(
                "knowledge",
                SearchOptions {
                    agent_id: Some("agent-1".to_string()),
                    ..Default::default()
                },
            )
            .await
            .unwrap();

        assert!(
            !results.results.is_empty(),
            "Memories without user_id must be visible to any caller"
        );
    }

    // ==========================================
    // TST-06: Ownership validation tests
    // ==========================================

    #[tokio::test]
    async fn test_update_ownership_rejects_wrong_user() {
        // PRIV-02: update with wrong user_id returns Unauthorized
        let config = MemoryConfig::default();
        let memory = Memory::new(config).await.unwrap();

        let result = memory
            .add(
                "Alice memory",
                AddOptions {
                    user_id: Some("alice".to_string()),
                    infer: false,
                    ..Default::default()
                },
            )
            .await
            .unwrap();

        let memory_id = result.results[0].id.to_string();

        let err = memory
            .update(
                &memory_id,
                "Bob edit",
                UpdateOptions {
                    user_id: Some("bob".to_string()),
                    ..Default::default()
                },
            )
            .await;

        assert!(err.is_err());
        assert!(
            matches!(err.unwrap_err(), MemoryError::Unauthorized { .. }),
            "Must return Unauthorized, not NotFound or other error"
        );
    }

    #[tokio::test]
    async fn test_delete_ownership_rejects_wrong_user() {
        // PRIV-03: delete with wrong user_id returns Unauthorized
        let config = MemoryConfig::default();
        let memory = Memory::new(config).await.unwrap();

        let result = memory
            .add(
                "Alice memory",
                AddOptions {
                    user_id: Some("alice".to_string()),
                    infer: false,
                    ..Default::default()
                },
            )
            .await
            .unwrap();

        let memory_id = result.results[0].id.to_string();

        let err = memory
            .delete(
                &memory_id,
                DeleteOptions {
                    user_id: Some("bob".to_string()),
                    ..Default::default()
                },
            )
            .await;

        assert!(err.is_err());
        assert!(
            matches!(err.unwrap_err(), MemoryError::Unauthorized { .. }),
            "Must return Unauthorized, not NotFound or other error"
        );
    }

    #[tokio::test]
    async fn test_update_ownership_allows_same_user() {
        // PRIV-02: update with matching user_id succeeds
        let config = MemoryConfig::default();
        let memory = Memory::new(config).await.unwrap();

        let result = memory
            .add(
                "Alice memory",
                AddOptions {
                    user_id: Some("alice".to_string()),
                    infer: false,
                    ..Default::default()
                },
            )
            .await
            .unwrap();

        let memory_id = result.results[0].id.to_string();

        let updated = memory
            .update(
                &memory_id,
                "Alice updated",
                UpdateOptions {
                    user_id: Some("alice".to_string()),
                    ..Default::default()
                },
            )
            .await;

        assert!(updated.is_ok(), "Same user update must succeed");
        assert_eq!(updated.unwrap().content, "Alice updated");
    }

    #[tokio::test]
    async fn test_delete_ownership_allows_same_user() {
        // PRIV-03: delete with matching user_id succeeds
        let config = MemoryConfig::default();
        let memory = Memory::new(config).await.unwrap();

        let result = memory
            .add(
                "Alice memory to delete",
                AddOptions {
                    user_id: Some("alice".to_string()),
                    infer: false,
                    ..Default::default()
                },
            )
            .await
            .unwrap();

        let memory_id = result.results[0].id.to_string();

        let deleted = memory
            .delete(
                &memory_id,
                DeleteOptions {
                    user_id: Some("alice".to_string()),
                    ..Default::default()
                },
            )
            .await;

        assert!(deleted.is_ok(), "Same user delete must succeed");
    }

    #[tokio::test]
    async fn test_update_no_user_memory_modifiable_by_anyone() {
        // Memories without user_id can be updated by anyone
        let config = MemoryConfig::default();
        let memory = Memory::new(config).await.unwrap();

        let result = memory
            .add(
                "Shared memory",
                AddOptions {
                    agent_id: Some("agent-1".to_string()),
                    infer: false,
                    ..Default::default()
                },
            )
            .await
            .unwrap();

        let memory_id = result.results[0].id.to_string();

        let updated = memory
            .update(
                &memory_id,
                "Updated by bob",
                UpdateOptions {
                    user_id: Some("bob".to_string()),
                    ..Default::default()
                },
            )
            .await;

        assert!(
            updated.is_ok(),
            "Memories without user_id must be modifiable by anyone"
        );
    }

    // ==========================================
    // TST-08: Contains/IContains filter test
    // ==========================================

    #[tokio::test]
    async fn test_contains_filter_client_side_evaluation() {
        // PRIV-05: Contains filter returns results via client-side eval
        let config = MemoryConfig::default();
        let memory = Memory::new(config).await.unwrap();

        memory
            .add(
                "I love Rust programming language",
                AddOptions {
                    user_id: Some("alice".to_string()),
                    infer: false,
                    ..Default::default()
                },
            )
            .await
            .unwrap();

        // Search with a Contains filter on the data field
        let results = memory
            .search(
                "programming",
                SearchOptions {
                    user_id: Some("alice".to_string()),
                    filters: Some(Filters {
                        conditions: vec![FilterCondition {
                            field: "data".to_string(),
                            operator: FilterOperator::Contains,
                            value: serde_json::Value::String("Rust".to_string()),
                        }],
                        logic: FilterLogic::And,
                    }),
                    ..Default::default()
                },
            )
            .await
            .unwrap();

        assert!(
            !results.results.is_empty(),
            "Contains filter must return matching results via client-side evaluation"
        );
    }
}
