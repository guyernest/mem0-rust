//! S3 Vectors vector store backend.

use async_trait::async_trait;
use std::collections::HashMap;

use super::traits::{VectorSearchResult, VectorStore};
use crate::config::S3VectorsConfig;
use crate::errors::VectorStoreError;
use crate::models::{FilterCondition, FilterLogic, FilterOperator, Filters, Payload};

use aws_smithy_types::Document;
use aws_smithy_types::Number;

/// Maximum number of vectors per batch operation (S3 Vectors limit)
const BATCH_LIMIT: usize = 500;

/// S3 Vectors vector store
pub struct S3VectorsStore {
    client: aws_sdk_s3vectors::Client,
    vector_bucket_name: String,
    index_name: String,
    dimensions: usize,
    distance_metric: String,
}

impl S3VectorsStore {
    /// Create a new S3VectorsStore, auto-creating the bucket and index if needed (per D-04).
    pub async fn new(
        config: S3VectorsConfig,
        index_name: &str,
        dimensions: usize,
    ) -> Result<Self, VectorStoreError> {
        use aws_config::BehaviorVersion;
        use aws_sdk_s3vectors::config::Region;

        let region = config
            .region
            .clone()
            .unwrap_or_else(|| "us-east-1".to_string());

        let sdk_config = aws_config::defaults(BehaviorVersion::latest())
            .region(Region::new(region))
            .load()
            .await;

        let client = aws_sdk_s3vectors::Client::new(&sdk_config);

        let store = Self {
            client,
            vector_bucket_name: config.bucket_name.clone(),
            index_name: index_name.to_string(),
            dimensions,
            distance_metric: config
                .distance_metric
                .clone()
                .unwrap_or_else(|| "cosine".to_string()),
        };

        // Auto-create bucket and index on initialization (D-04)
        store.ensure_bucket_exists().await?;
        if !store.collection_exists().await? {
            store.create_collection().await?;
        }

        Ok(store)
    }

    /// Ensure the vector bucket exists, creating it if necessary (per D-04).
    async fn ensure_bucket_exists(&self) -> Result<(), VectorStoreError> {
        match self
            .client
            .get_vector_bucket()
            .vector_bucket_name(&self.vector_bucket_name)
            .send()
            .await
        {
            Ok(_) => {
                tracing::debug!(
                    bucket = %self.vector_bucket_name,
                    "S3 Vectors bucket already exists"
                );
                Ok(())
            }
            Err(e) => {
                if is_not_found_error(&e) {
                    tracing::info!(
                        bucket = %self.vector_bucket_name,
                        "S3 Vectors bucket not found, creating"
                    );
                    self.client
                        .create_vector_bucket()
                        .vector_bucket_name(&self.vector_bucket_name)
                        .send()
                        .await
                        .map_err(|ce| {
                            VectorStoreError::Connection(format!(
                                "failed to create bucket '{}': {}",
                                self.vector_bucket_name, ce
                            ))
                        })?;
                    tracing::info!(bucket = %self.vector_bucket_name, "S3 Vectors bucket created");
                    Ok(())
                } else {
                    Err(VectorStoreError::Connection(format!(
                        "failed to check bucket '{}': {}",
                        self.vector_bucket_name, e
                    )))
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Helper: NotFoundException detection
// ---------------------------------------------------------------------------

/// Return true if the SDK error represents a "resource not found" condition.
///
/// Uses string matching on the error Display representation as a robust
/// approach across SDK versions. The `NotFoundException` code appears in the
/// Display output for service errors from S3 Vectors.
fn is_not_found_error<E: std::fmt::Display>(
    e: &aws_sdk_s3vectors::error::SdkError<E>,
) -> bool {
    let error_str = e.to_string();
    error_str.contains("NotFoundException")
        || error_str.contains("not found")
        || error_str.contains("Not Found")
}

// ---------------------------------------------------------------------------
// Payload <-> Document conversion (per D-01, D-02)
// ---------------------------------------------------------------------------

/// Serialize a `Payload` into a flat `aws_smithy_types::Document` object.
///
/// Filterable fields (D-01): user_id, agent_id, run_id, hash, created_at, memory_type
/// Non-filterable fields (D-02): data (content), plus all extra metadata entries
///
/// The filterable/non-filterable split is enforced at index-creation time via
/// `MetadataConfiguration::non_filterable_metadata_keys`. Here we simply emit
/// all fields into a single Document; the index schema handles the policy.
pub(crate) fn payload_to_document(payload: &Payload) -> Document {
    let mut map: HashMap<String, Document> = HashMap::with_capacity(8 + payload.metadata.len());

    // Filterable fields (D-01)
    if let Some(uid) = &payload.user_id {
        map.insert("user_id".to_string(), Document::String(uid.clone()));
    }
    if let Some(aid) = &payload.agent_id {
        map.insert("agent_id".to_string(), Document::String(aid.clone()));
    }
    if let Some(rid) = &payload.run_id {
        map.insert("run_id".to_string(), Document::String(rid.clone()));
    }
    map.insert("hash".to_string(), Document::String(payload.hash.clone()));
    map.insert(
        "created_at".to_string(),
        Document::String(payload.created_at.to_rfc3339()),
    );

    // memory_type: filterable (D-01) — now a first-class Payload field (D-05, Phase 2)
    if let Some(mt) = payload.memory_type {
        map.insert("memory_type".to_string(), Document::String(mt.to_string()));
    }

    // Non-filterable fields (D-02): data is declared non-filterable at create_index time
    map.insert("data".to_string(), Document::String(payload.data.clone()));

    // Remaining metadata entries
    for (k, v) in &payload.metadata {
        map.insert(k.clone(), json_value_to_document(v));
    }

    Document::Object(map)
}

/// Convert `aws_smithy_types::Document` back to a `Payload`.
///
/// Expects the document to be a `Document::Object` produced by `payload_to_document`.
pub(crate) fn document_to_payload(doc: &Document) -> Payload {
    use chrono::{DateTime, Utc};

    let map = match doc {
        Document::Object(m) => m,
        _ => {
            return Payload {
                data: String::new(),
                hash: String::new(),
                created_at: Utc::now(),
                user_id: None,
                agent_id: None,
                run_id: None,
                memory_type: None,
                metadata: HashMap::new(),
            }
        }
    };

    let get_str = |key: &str| -> Option<String> {
        match map.get(key) {
            Some(Document::String(s)) => Some(s.clone()),
            _ => None,
        }
    };

    let data = get_str("data").unwrap_or_default();
    let hash = get_str("hash").unwrap_or_default();
    let created_at = get_str("created_at")
        .and_then(|s| DateTime::parse_from_rfc3339(&s).ok())
        .map(|dt| dt.with_timezone(&Utc))
        .unwrap_or_else(Utc::now);
    let user_id = get_str("user_id");
    let agent_id = get_str("agent_id");
    let run_id = get_str("run_id");

    // Deserialize memory_type from its string representation (D-05)
    let memory_type = get_str("memory_type").and_then(|s| {
        serde_json::from_value::<crate::models::MemoryType>(serde_json::Value::String(s)).ok()
    });

    // All remaining fields go into metadata
    let known_keys = [
        "data", "hash", "created_at", "user_id", "agent_id", "run_id", "memory_type",
    ];
    let metadata: HashMap<String, serde_json::Value> = map
        .iter()
        .filter(|(k, _v): &(&String, &Document)| !known_keys.contains(&k.as_str()))
        .map(|(k, v): (&String, &Document)| (k.clone(), document_to_json_value(v)))
        .collect();

    Payload {
        data,
        hash,
        created_at,
        user_id,
        agent_id,
        run_id,
        memory_type,
        metadata,
    }
}

/// Convert a `serde_json::Value` to `aws_smithy_types::Document`.
pub(crate) fn json_value_to_document(v: &serde_json::Value) -> Document {
    match v {
        serde_json::Value::Null => Document::Null,
        serde_json::Value::Bool(b) => Document::Bool(*b),
        serde_json::Value::Number(n) => {
            Document::Number(Number::Float(n.as_f64().unwrap_or(0.0)))
        }
        serde_json::Value::String(s) => Document::String(s.clone()),
        serde_json::Value::Array(arr) => {
            Document::Array(arr.iter().map(json_value_to_document).collect())
        }
        serde_json::Value::Object(map) => Document::Object(
            map.iter()
                .map(|(k, v)| (k.clone(), json_value_to_document(v)))
                .collect(),
        ),
    }
}

/// Convert `aws_smithy_types::Document` to `serde_json::Value`.
pub(crate) fn document_to_json_value(doc: &Document) -> serde_json::Value {
    match doc {
        Document::Null => serde_json::Value::Null,
        Document::Bool(b) => serde_json::Value::Bool(*b),
        Document::Number(n) => {
            // Number::Float is the primary variant; convert via f64
            let f = match n {
                Number::Float(f) => *f,
                Number::PosInt(i) => *i as f64,
                Number::NegInt(i) => *i as f64,
            };
            serde_json::Number::from_f64(f)
                .map(serde_json::Value::Number)
                .unwrap_or(serde_json::Value::Null)
        }
        Document::String(s) => serde_json::Value::String(s.clone()),
        Document::Array(arr) => {
            serde_json::Value::Array(arr.iter().map(document_to_json_value).collect())
        }
        Document::Object(map) => serde_json::Value::Object(
            map.iter()
                .map(|(k, v): (&String, &Document)| (k.clone(), document_to_json_value(v)))
                .collect(),
        ),
    }
}

// ---------------------------------------------------------------------------
// Filter translation (per S3V-02, Pattern 4 from RESEARCH.md)
// ---------------------------------------------------------------------------

/// Build an S3 Vectors filter `Document` from a `Filters` struct.
///
/// Returns `None` if there are no applicable conditions (empty input or all
/// conditions used `Contains`/`IContains` which have no S3 Vectors equivalent).
pub(crate) fn build_filter(filters: &Filters) -> Option<Document> {
    if filters.conditions.is_empty() {
        return None;
    }

    let conditions: Vec<Document> = filters
        .conditions
        .iter()
        .filter_map(condition_to_document)
        .collect();

    if conditions.is_empty() {
        return None;
    }

    if conditions.len() == 1 {
        return Some(conditions.into_iter().next().unwrap());
    }

    // Multiple conditions: wrap in $and or $or
    let logic_key = match filters.logic {
        FilterLogic::And => "$and",
        FilterLogic::Or => "$or",
    };

    let mut map = HashMap::new();
    map.insert(logic_key.to_string(), Document::Array(conditions));
    Some(Document::Object(map))
}

/// Convert a single `FilterCondition` to an S3 Vectors filter `Document`.
///
/// Returns `None` for operators without an S3 Vectors equivalent (`Contains`, `IContains`).
pub(crate) fn condition_to_document(cond: &FilterCondition) -> Option<Document> {
    let op_key = match cond.operator {
        FilterOperator::Eq => "$eq",
        FilterOperator::Ne => "$ne",
        FilterOperator::Gt => "$gt",
        FilterOperator::Gte => "$gte",
        FilterOperator::Lt => "$lt",
        FilterOperator::Lte => "$lte",
        FilterOperator::In => "$in",
        FilterOperator::Nin => "$nin",
        // No S3 Vectors equivalent — skip condition rather than error
        FilterOperator::Contains | FilterOperator::IContains => return None,
    };

    let doc_value = json_value_to_document(&cond.value);

    let mut inner = HashMap::new();
    inner.insert(op_key.to_string(), doc_value);

    let mut outer = HashMap::new();
    outer.insert(cond.field.clone(), Document::Object(inner));
    Some(Document::Object(outer))
}

// ---------------------------------------------------------------------------
// Client-side filter matching for list() (per S3V-05, Pattern 6 from RESEARCH.md)
// ---------------------------------------------------------------------------

// Client-side filter evaluation is provided by the shared `filter_eval` module.
use super::filter_eval::matches_filters as matches_payload_filters;

// ---------------------------------------------------------------------------
// VectorStore trait implementation
// ---------------------------------------------------------------------------

#[async_trait]
impl VectorStore for S3VectorsStore {
    /// Insert a single vector with its payload metadata.
    ///
    /// Uses `put_vectors` which acts as an upsert — if a vector with the same
    /// key already exists it is overwritten.
    async fn insert(
        &self,
        id: &str,
        embedding: Vec<f32>,
        payload: Payload,
    ) -> Result<(), VectorStoreError> {
        use aws_sdk_s3vectors::types::{PutInputVector, VectorData};

        let vector = PutInputVector::builder()
            .key(id)
            .data(VectorData::Float32(embedding))
            .metadata(payload_to_document(&payload))
            .build()
            .map_err(|e| VectorStoreError::Insert(format!("failed to build PutInputVector: {}", e)))?;

        self.client
            .put_vectors()
            .vector_bucket_name(&self.vector_bucket_name)
            .index_name(&self.index_name)
            .vectors(vector)
            .send()
            .await
            .map_err(|e| VectorStoreError::Insert(format!("put_vectors failed: {}", e)))?;

        Ok(())
    }

    /// Search for vectors similar to the given embedding.
    ///
    /// Applies S3 Vectors server-side filtering when filters are present and
    /// translatable to Document form. Converts returned distance to a score
    /// (1.0 - distance for cosine) so higher score = more similar.
    async fn search(
        &self,
        embedding: &[f32],
        limit: usize,
        filters: Option<&Filters>,
    ) -> Result<Vec<VectorSearchResult>, VectorStoreError> {
        use aws_sdk_s3vectors::types::VectorData;

        let mut req = self
            .client
            .query_vectors()
            .vector_bucket_name(&self.vector_bucket_name)
            .index_name(&self.index_name)
            .query_vector(VectorData::Float32(embedding.to_vec()))
            .top_k(limit as i32)
            .return_metadata(true)
            .return_distance(true);

        // Apply server-side filter when available
        if let Some(f) = filters {
            if let Some(filter_doc) = build_filter(f) {
                req = req.filter(filter_doc);
            }
        }

        let response = req
            .send()
            .await
            .map_err(|e| VectorStoreError::Search(format!("query_vectors failed: {}", e)))?;

        let results = response
            .vectors()
            .iter()
            .map(|v| {
                let id = v.key().to_string();
                // Convert distance to score: for cosine, distance = 1 - similarity,
                // so score = 1.0 - distance gives higher values for more similar vectors.
                let distance = v.distance().unwrap_or(0.0);
                let score = if distance >= 0.0 { 1.0 - distance } else { 0.0 };
                let payload = v
                    .metadata()
                    .map(document_to_payload)
                    .unwrap_or_else(|| document_to_payload(&Document::Null));
                VectorSearchResult { id, score, payload }
            })
            .collect();

        Ok(results)
    }

    /// Retrieve a single vector record by its key.
    ///
    /// Returns metadata only (no vector data) — the VectorSearchResult score is
    /// set to 1.0 as there is no distance for a direct key lookup.
    async fn get(&self, id: &str) -> Result<Option<VectorSearchResult>, VectorStoreError> {
        let response = self
            .client
            .get_vectors()
            .vector_bucket_name(&self.vector_bucket_name)
            .index_name(&self.index_name)
            .keys(id)
            .return_data(false)
            .return_metadata(true)
            .send()
            .await
            .map_err(|e| VectorStoreError::Search(format!("get_vectors failed: {}", e)))?;

        let vectors = response.vectors();
        if vectors.is_empty() {
            return Ok(None);
        }

        let v = &vectors[0];
        let payload = v
            .metadata()
            .map(document_to_payload)
            .unwrap_or_else(|| document_to_payload(&Document::Null));

        Ok(Some(VectorSearchResult {
            id: v.key().to_string(),
            score: 1.0,
            payload,
        }))
    }

    /// Delete a vector by its key.
    async fn delete(&self, id: &str) -> Result<(), VectorStoreError> {
        self.client
            .delete_vectors()
            .vector_bucket_name(&self.vector_bucket_name)
            .index_name(&self.index_name)
            .keys(id)
            .send()
            .await
            .map_err(|e| VectorStoreError::Delete(format!("delete_vectors failed: {}", e)))?;

        Ok(())
    }

    /// Update an existing vector record.
    ///
    /// When `embedding` is `None` the existing vector data is fetched first to
    /// preserve its embedding. Using a zero vector as a fallback is explicitly
    /// avoided (Pitfall 4 from RESEARCH.md) since it would destroy semantic content.
    async fn update(
        &self,
        id: &str,
        embedding: Option<Vec<f32>>,
        payload: Payload,
    ) -> Result<(), VectorStoreError> {
        let embedding_vec = match embedding {
            Some(emb) => emb,
            None => {
                // Fetch the existing vector data to preserve its embedding.
                let response = self
                    .client
                    .get_vectors()
                    .vector_bucket_name(&self.vector_bucket_name)
                    .index_name(&self.index_name)
                    .keys(id)
                    .return_data(true)
                    .return_metadata(false)
                    .send()
                    .await
                    .map_err(|e| {
                        VectorStoreError::Update(format!(
                            "get_vectors failed while fetching embedding for update: {}",
                            e
                        ))
                    })?;

                let vectors = response.vectors();
                if vectors.is_empty() {
                    return Err(VectorStoreError::NotFound(id.to_string()));
                }

                let v = &vectors[0];
                match v.data() {
                    Some(aws_sdk_s3vectors::types::VectorData::Float32(data)) => data.clone(),
                    _ => {
                        return Err(VectorStoreError::Update(format!(
                            "vector '{}' has no Float32 data",
                            id
                        )));
                    }
                }
            }
        };

        // S3 Vectors put_vectors acts as upsert — re-use insert for the actual write.
        self.insert(id, embedding_vec, payload).await.map_err(|e| {
            VectorStoreError::Update(format!("insert during update failed: {}", e))
        })
    }

    /// List all vectors, applying client-side filtering when filters are present.
    ///
    /// `list_vectors` does not support metadata filter parameters, so all vectors
    /// are fetched and filtered in memory using `matches_payload_filters`.
    async fn list(
        &self,
        filters: Option<&Filters>,
        limit: usize,
    ) -> Result<Vec<VectorSearchResult>, VectorStoreError> {
        let response = self
            .client
            .list_vectors()
            .vector_bucket_name(&self.vector_bucket_name)
            .index_name(&self.index_name)
            .return_metadata(true)
            .send()
            .await
            .map_err(|e| VectorStoreError::Search(format!("list_vectors failed: {}", e)))?;

        let total = response.vectors().len();

        let filtered: Vec<VectorSearchResult> = response
            .vectors()
            .iter()
            .map(|v| {
                let payload = v
                    .metadata()
                    .map(document_to_payload)
                    .unwrap_or_else(|| document_to_payload(&Document::Null));
                VectorSearchResult {
                    id: v.key().to_string(),
                    score: 1.0,
                    payload,
                }
            })
            .filter(|r| matches_payload_filters(&r.payload, filters))
            .take(limit)
            .collect();

        tracing::debug!(
            "S3 Vectors list: fetched {} vectors, {} after filtering",
            total,
            filtered.len()
        );

        Ok(filtered)
    }

    /// Delete all vectors matching the given filters.
    ///
    /// Two strategies:
    /// - **No filters:** destroy and recreate the index (fast, O(1) API calls).
    /// - **With filters:** list matching vectors then batch-delete in 500-item chunks.
    async fn delete_all(&self, filters: Option<&Filters>) -> Result<usize, VectorStoreError> {
        match filters {
            None => {
                // Fastest path: delete the entire index and recreate it.
                self.client
                    .delete_index()
                    .vector_bucket_name(&self.vector_bucket_name)
                    .index_name(&self.index_name)
                    .send()
                    .await
                    .map_err(|e| {
                        VectorStoreError::Delete(format!("delete_index failed: {}", e))
                    })?;

                self.create_collection().await?;

                tracing::info!(
                    index = %self.index_name,
                    "S3 Vectors index reset (delete + recreate)"
                );

                // Exact count unavailable — delete_index is an atomic wipe.
                Ok(0)
            }
            Some(f) => {
                // Targeted delete: list matching IDs then batch-delete in chunks.
                let matching = self.list(Some(f), usize::MAX).await?;
                let ids: Vec<String> = matching.into_iter().map(|r| r.id).collect();

                if ids.is_empty() {
                    return Ok(0);
                }

                let total = ids.len();

                for chunk in ids.chunks(BATCH_LIMIT) {
                    self.client
                        .delete_vectors()
                        .vector_bucket_name(&self.vector_bucket_name)
                        .index_name(&self.index_name)
                        .set_keys(Some(chunk.to_vec()))
                        .send()
                        .await
                        .map_err(|e| {
                            VectorStoreError::Delete(format!(
                                "delete_vectors batch failed: {}",
                                e
                            ))
                        })?;
                }

                tracing::info!(
                    index = %self.index_name,
                    deleted = total,
                    "S3 Vectors filtered delete_all complete"
                );

                Ok(total)
            }
        }
    }

    /// Check whether the vector index exists.
    async fn collection_exists(&self) -> Result<bool, VectorStoreError> {
        match self
            .client
            .get_index()
            .vector_bucket_name(&self.vector_bucket_name)
            .index_name(&self.index_name)
            .send()
            .await
        {
            Ok(_) => Ok(true),
            Err(e) if is_not_found_error(&e) => Ok(false),
            Err(e) => Err(VectorStoreError::Collection(format!(
                "failed to check index '{}': {}",
                self.index_name, e
            ))),
        }
    }

    /// Create the vector index with filterable/non-filterable metadata configuration.
    ///
    /// Per D-01/D-02: `data` (memory content) is declared non-filterable; all
    /// scoping fields (user_id, agent_id, run_id, hash, memory_type, created_at)
    /// remain filterable by default.
    async fn create_collection(&self) -> Result<(), VectorStoreError> {
        use aws_sdk_s3vectors::types::{DataType, DistanceMetric, MetadataConfiguration};

        // D-01: filterable fields: user_id, agent_id, run_id, hash, memory_type, created_at
        // D-02: non-filterable fields: data (content string, can be arbitrarily large)
        let metadata_config = MetadataConfiguration::builder()
            .non_filterable_metadata_keys("data")
            .build()
            .map_err(|e| VectorStoreError::Collection(format!("failed to build metadata config: {}", e)))?;

        // Map config string to SDK DistanceMetric. The SDK currently defines
        // Cosine and Euclidean; forward anything else through the From<&str> impl.
        let distance_metric = DistanceMetric::from(self.distance_metric.to_lowercase().as_str());

        self.client
            .create_index()
            .vector_bucket_name(&self.vector_bucket_name)
            .index_name(&self.index_name)
            .data_type(DataType::Float32)
            .dimension(self.dimensions as i32)
            .distance_metric(distance_metric)
            .metadata_configuration(metadata_config)
            .send()
            .await
            .map_err(|e| {
                VectorStoreError::Collection(format!(
                    "failed to create index '{}': {}",
                    self.index_name, e
                ))
            })?;

        tracing::info!(
            index = %self.index_name,
            bucket = %self.vector_bucket_name,
            dimensions = self.dimensions,
            "S3 Vectors index created"
        );

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn make_payload(data: &str, user_id: Option<&str>) -> Payload {
        Payload {
            data: data.to_string(),
            hash: "abc123".to_string(),
            created_at: Utc::now(),
            user_id: user_id.map(|s| s.to_string()),
            agent_id: None,
            run_id: None,
            memory_type: None,
            metadata: HashMap::new(),
        }
    }

    // -----------------------------------------------------------------------
    // payload_to_document / document_to_payload round-trip
    // -----------------------------------------------------------------------

    #[test]
    fn test_payload_to_document_contains_required_fields() {
        let payload = make_payload("hello world", Some("user-1"));
        let doc = payload_to_document(&payload);

        let Document::Object(ref map) = doc else {
            panic!("expected Document::Object");
        };

        assert!(matches!(map.get("data"), Some(Document::String(s)) if s == "hello world"));
        assert!(matches!(map.get("hash"), Some(Document::String(s)) if s == "abc123"));
        assert!(matches!(map.get("user_id"), Some(Document::String(s)) if s == "user-1"));
        assert!(map.contains_key("created_at"));
    }

    #[test]
    fn test_payload_to_document_omits_absent_optional_fields() {
        let payload = make_payload("data", None);
        let doc = payload_to_document(&payload);
        let Document::Object(ref map) = doc else {
            panic!("expected Document::Object");
        };
        assert!(!map.contains_key("user_id"));
        assert!(!map.contains_key("agent_id"));
        assert!(!map.contains_key("run_id"));
    }

    #[test]
    fn test_payload_to_document_memory_type_filterable() {
        // memory_type is now a first-class Payload field (D-05, Phase 2)
        let payload = Payload {
            data: "data".to_string(),
            hash: "h".to_string(),
            created_at: Utc::now(),
            user_id: None,
            agent_id: None,
            run_id: None,
            memory_type: Some(crate::models::MemoryType::Semantic),
            metadata: HashMap::new(),
        };
        let doc = payload_to_document(&payload);
        let Document::Object(ref map) = doc else {
            panic!("expected Document::Object");
        };
        // memory_type should appear as top-level filterable field with Python-compatible string
        assert!(
            matches!(map.get("memory_type"), Some(Document::String(s)) if s == "semantic_memory"),
            "memory_type should be filterable at top level with Python-compatible string value"
        );
    }

    #[test]
    fn test_document_to_payload_round_trip() {
        let original = Payload {
            data: "round trip data".to_string(),
            hash: "deadbeef".to_string(),
            created_at: chrono::DateTime::parse_from_rfc3339("2024-01-01T00:00:00Z")
                .unwrap()
                .with_timezone(&Utc),
            user_id: Some("u1".to_string()),
            agent_id: Some("a1".to_string()),
            run_id: Some("r1".to_string()),
            memory_type: None,
            metadata: HashMap::new(),
        };

        let doc = payload_to_document(&original);
        let restored = document_to_payload(&doc);

        assert_eq!(restored.data, original.data);
        assert_eq!(restored.hash, original.hash);
        assert_eq!(restored.user_id, original.user_id);
        assert_eq!(restored.agent_id, original.agent_id);
        assert_eq!(restored.run_id, original.run_id);
    }

    // -----------------------------------------------------------------------
    // json_value_to_document / document_to_json_value round-trip
    // -----------------------------------------------------------------------

    #[test]
    fn test_json_value_round_trip() {
        let values = vec![
            serde_json::json!(null),
            serde_json::json!(true),
            serde_json::json!(false),
            serde_json::json!(42.5),
            serde_json::json!("hello"),
            serde_json::json!(["a", "b", "c"]),
            serde_json::json!({"key": "value"}),
        ];

        for v in values {
            let doc = json_value_to_document(&v);
            let restored = document_to_json_value(&doc);
            // For number types, compare as f64 to avoid JSON type issues
            match &v {
                serde_json::Value::Number(_) => {
                    assert_eq!(v.as_f64(), restored.as_f64());
                }
                _ => assert_eq!(v, restored),
            }
        }
    }

    // -----------------------------------------------------------------------
    // build_filter / condition_to_document
    // -----------------------------------------------------------------------

    #[test]
    fn test_build_filter_empty_returns_none() {
        let filters = Filters {
            conditions: vec![],
            logic: FilterLogic::And,
        };
        assert!(build_filter(&filters).is_none());
    }

    #[test]
    fn test_build_filter_eq_single_condition() {
        let filters = Filters {
            conditions: vec![FilterCondition {
                field: "user_id".to_string(),
                operator: FilterOperator::Eq,
                value: serde_json::json!("user-123"),
            }],
            logic: FilterLogic::And,
        };

        let doc = build_filter(&filters).expect("should produce a filter document");
        let Document::Object(ref outer) = doc else {
            panic!("expected outer Document::Object, got {:?}", doc);
        };
        let Some(Document::Object(ref inner)) = outer.get("user_id") else {
            panic!("expected inner Document::Object for field 'user_id'");
        };
        assert!(
            matches!(inner.get("$eq"), Some(Document::String(s)) if s == "user-123"),
            "expected $eq operator with value 'user-123'"
        );
    }

    #[test]
    fn test_build_filter_multiple_conditions_wrapped_in_and() {
        let filters = Filters {
            conditions: vec![
                FilterCondition {
                    field: "user_id".to_string(),
                    operator: FilterOperator::Eq,
                    value: serde_json::json!("u1"),
                },
                FilterCondition {
                    field: "agent_id".to_string(),
                    operator: FilterOperator::Eq,
                    value: serde_json::json!("a1"),
                },
            ],
            logic: FilterLogic::And,
        };

        let doc = build_filter(&filters).expect("should produce a filter document");
        let Document::Object(ref map) = doc else {
            panic!("expected Document::Object for $and wrapper");
        };
        let Some(Document::Array(ref conditions)) = map.get("$and") else {
            panic!("expected $and array");
        };
        assert_eq!(conditions.len(), 2);
    }

    #[test]
    fn test_build_filter_or_logic() {
        let filters = Filters {
            conditions: vec![
                FilterCondition {
                    field: "user_id".to_string(),
                    operator: FilterOperator::Eq,
                    value: serde_json::json!("u1"),
                },
                FilterCondition {
                    field: "user_id".to_string(),
                    operator: FilterOperator::Eq,
                    value: serde_json::json!("u2"),
                },
            ],
            logic: FilterLogic::Or,
        };

        let doc = build_filter(&filters).expect("should produce a filter document");
        let Document::Object(ref map) = doc else {
            panic!("expected Document::Object");
        };
        assert!(map.contains_key("$or"), "expected $or key");
    }

    #[test]
    fn test_build_filter_contains_skipped() {
        // Contains has no S3 Vectors equivalent — should return None when only such operators
        let filters = Filters {
            conditions: vec![FilterCondition {
                field: "data".to_string(),
                operator: FilterOperator::Contains,
                value: serde_json::json!("substring"),
            }],
            logic: FilterLogic::And,
        };
        assert!(
            build_filter(&filters).is_none(),
            "Contains operator should be skipped (no S3 Vectors equivalent)"
        );
    }

    // -----------------------------------------------------------------------
    // matches_payload_filters
    // -----------------------------------------------------------------------

    #[test]
    fn test_matches_payload_filters_no_filters() {
        let payload = make_payload("data", Some("u1"));
        assert!(matches_payload_filters(&payload, None));
    }

    #[test]
    fn test_matches_payload_filters_eq_match() {
        let payload = make_payload("data", Some("u1"));
        let filters = Filters {
            conditions: vec![FilterCondition {
                field: "user_id".to_string(),
                operator: FilterOperator::Eq,
                value: serde_json::json!("u1"),
            }],
            logic: FilterLogic::And,
        };
        assert!(matches_payload_filters(&payload, Some(&filters)));
    }

    #[test]
    fn test_matches_payload_filters_eq_no_match() {
        let payload = make_payload("data", Some("u1"));
        let filters = Filters {
            conditions: vec![FilterCondition {
                field: "user_id".to_string(),
                operator: FilterOperator::Eq,
                value: serde_json::json!("u2"),
            }],
            logic: FilterLogic::And,
        };
        assert!(!matches_payload_filters(&payload, Some(&filters)));
    }

    #[test]
    fn test_matches_payload_filters_metadata_field() {
        // memory_type is now a first-class Payload field (D-05, Phase 2) — use it directly
        let mut payload = make_payload("data", None);
        payload.memory_type = Some(crate::models::MemoryType::Semantic);
        let filters = Filters {
            conditions: vec![FilterCondition {
                field: "memory_type".to_string(),
                operator: FilterOperator::Eq,
                value: serde_json::json!("semantic_memory"),
            }],
            logic: FilterLogic::And,
        };
        assert!(matches_payload_filters(&payload, Some(&filters)));
    }

    #[test]
    fn test_batch_limit_constant() {
        assert_eq!(BATCH_LIMIT, 500);
    }

    // -----------------------------------------------------------------------
    // Additional filter tests (Task 1 — TST-04 coverage)
    // -----------------------------------------------------------------------

    /// Create a test Payload with configurable scoping fields.
    fn create_test_payload(
        data: &str,
        user_id: Option<&str>,
        agent_id: Option<&str>,
        run_id: Option<&str>,
    ) -> Payload {
        Payload {
            data: data.to_string(),
            hash: "abc123hash".to_string(),
            created_at: Utc::now(),
            user_id: user_id.map(String::from),
            agent_id: agent_id.map(String::from),
            run_id: run_id.map(String::from),
            memory_type: None,
            metadata: HashMap::new(),
        }
    }

    #[test]
    fn test_build_filter_single_eq_condition() {
        let filters = Filters {
            conditions: vec![FilterCondition {
                field: "user_id".to_string(),
                operator: FilterOperator::Eq,
                value: serde_json::json!("alice"),
            }],
            logic: FilterLogic::And,
        };
        let doc = build_filter(&filters).expect("should produce a filter document");
        let Document::Object(ref outer) = doc else {
            panic!("expected Document::Object");
        };
        let Some(Document::Object(ref inner)) = outer.get("user_id") else {
            panic!("expected inner Document::Object for field 'user_id'");
        };
        assert!(
            matches!(inner.get("$eq"), Some(Document::String(s)) if s == "alice"),
            "expected $eq operator with value 'alice'"
        );
    }

    #[test]
    fn test_build_filter_all_scoping_fields() {
        // All 4 scoping fields: user_id, agent_id, run_id, memory_type
        let filters = Filters {
            conditions: vec![
                FilterCondition {
                    field: "user_id".to_string(),
                    operator: FilterOperator::Eq,
                    value: serde_json::json!("u1"),
                },
                FilterCondition {
                    field: "agent_id".to_string(),
                    operator: FilterOperator::Eq,
                    value: serde_json::json!("a1"),
                },
                FilterCondition {
                    field: "run_id".to_string(),
                    operator: FilterOperator::Eq,
                    value: serde_json::json!("r1"),
                },
                FilterCondition {
                    field: "memory_type".to_string(),
                    operator: FilterOperator::Eq,
                    value: serde_json::json!("long_term"),
                },
            ],
            logic: FilterLogic::And,
        };
        let doc = build_filter(&filters).expect("should produce a filter document");
        let Document::Object(ref map) = doc else {
            panic!("expected Document::Object for $and wrapper");
        };
        let Some(Document::Array(ref conditions)) = map.get("$and") else {
            panic!("expected $and array");
        };
        assert_eq!(conditions.len(), 4, "all 4 scoping fields should appear");

        // Check each field name appears in one of the conditions
        let field_names: Vec<String> = conditions
            .iter()
            .filter_map(|c| {
                if let Document::Object(m) = c {
                    m.keys().next().cloned()
                } else {
                    None
                }
            })
            .collect();
        assert!(field_names.contains(&"user_id".to_string()));
        assert!(field_names.contains(&"agent_id".to_string()));
        assert!(field_names.contains(&"run_id".to_string()));
        assert!(field_names.contains(&"memory_type".to_string()));
    }

    #[test]
    fn test_build_filter_empty_conditions() {
        let filters = Filters {
            conditions: vec![],
            logic: FilterLogic::And,
        };
        assert!(
            build_filter(&filters).is_none(),
            "empty conditions should produce None"
        );
    }

    #[test]
    fn test_condition_contains_returns_none() {
        let cond = FilterCondition {
            field: "data".to_string(),
            operator: FilterOperator::Contains,
            value: serde_json::json!("substring"),
        };
        assert!(
            condition_to_document(&cond).is_none(),
            "Contains operator has no S3 Vectors equivalent — should return None"
        );
    }

    #[test]
    fn test_build_filter_ne_gt_lt_operators() {
        // Ne -> $ne
        let ne_cond = FilterCondition {
            field: "score".to_string(),
            operator: FilterOperator::Ne,
            value: serde_json::json!(0),
        };
        let ne_doc = condition_to_document(&ne_cond).expect("Ne should produce a doc");
        let Document::Object(ref outer) = ne_doc else {
            panic!("expected outer object for Ne");
        };
        let Document::Object(ref inner) = outer["score"] else {
            panic!("expected inner object for 'score'");
        };
        assert!(inner.contains_key("$ne"), "expected $ne key");

        // Gt -> $gt
        let gt_cond = FilterCondition {
            field: "score".to_string(),
            operator: FilterOperator::Gt,
            value: serde_json::json!(5),
        };
        let gt_doc = condition_to_document(&gt_cond).expect("Gt should produce a doc");
        let Document::Object(ref o) = gt_doc else {
            panic!("expected outer object for Gt");
        };
        let Document::Object(ref i) = o["score"] else {
            panic!("expected inner object for Gt");
        };
        assert!(i.contains_key("$gt"), "expected $gt key");

        // Lt -> $lt
        let lt_cond = FilterCondition {
            field: "score".to_string(),
            operator: FilterOperator::Lt,
            value: serde_json::json!(10),
        };
        let lt_doc = condition_to_document(&lt_cond).expect("Lt should produce a doc");
        let Document::Object(ref o2) = lt_doc else {
            panic!("expected outer object for Lt");
        };
        let Document::Object(ref i2) = o2["score"] else {
            panic!("expected inner object for Lt");
        };
        assert!(i2.contains_key("$lt"), "expected $lt key");
    }

    // -----------------------------------------------------------------------
    // Additional payload serialization tests (Task 1 — S3V-03)
    // -----------------------------------------------------------------------

    #[test]
    fn test_payload_to_document_includes_filterable_fields() {
        let payload = create_test_payload("hello", Some("alice"), Some("bot"), None);
        let doc = payload_to_document(&payload);
        let Document::Object(ref map) = doc else {
            panic!("expected Document::Object");
        };
        assert!(map.contains_key("user_id"), "user_id missing");
        assert!(map.contains_key("agent_id"), "agent_id missing");
        assert!(map.contains_key("hash"), "hash missing");
        assert!(map.contains_key("created_at"), "created_at missing");
        assert!(map.contains_key("data"), "data missing");
    }

    #[test]
    fn test_payload_to_document_omits_none_optional_fields() {
        let payload = create_test_payload("data", None, None, None);
        let doc = payload_to_document(&payload);
        let Document::Object(ref map) = doc else {
            panic!("expected Document::Object");
        };
        assert!(!map.contains_key("user_id"), "user_id should be absent");
        assert!(!map.contains_key("agent_id"), "agent_id should be absent");
        assert!(!map.contains_key("run_id"), "run_id should be absent");
    }

    #[test]
    fn test_payload_to_document_includes_extra_metadata() {
        let mut payload = create_test_payload("data", None, None, None);
        payload
            .metadata
            .insert("category".to_string(), serde_json::json!("work"));
        let doc = payload_to_document(&payload);
        let Document::Object(ref map) = doc else {
            panic!("expected Document::Object");
        };
        assert!(
            matches!(map.get("category"), Some(Document::String(s)) if s == "work"),
            "extra metadata key 'category' should appear in Document"
        );
    }

    #[test]
    fn test_payload_document_roundtrip() {
        let original = Payload {
            data: "roundtrip content".to_string(),
            hash: "cafebabe".to_string(),
            created_at: chrono::DateTime::parse_from_rfc3339("2025-06-01T12:00:00Z")
                .unwrap()
                .with_timezone(&Utc),
            user_id: Some("user42".to_string()),
            agent_id: Some("agent7".to_string()),
            run_id: Some("run99".to_string()),
            memory_type: None,
            metadata: HashMap::new(),
        };
        let doc = payload_to_document(&original);
        let restored = document_to_payload(&doc);
        assert_eq!(restored.data, original.data);
        assert_eq!(restored.hash, original.hash);
        assert_eq!(restored.user_id, original.user_id);
        assert_eq!(restored.agent_id, original.agent_id);
        assert_eq!(restored.run_id, original.run_id);
        // Compare created_at via RFC3339 to avoid sub-second precision differences
        assert_eq!(
            restored.created_at.to_rfc3339(),
            original.created_at.to_rfc3339()
        );
    }

    // -----------------------------------------------------------------------
    // Additional client-side filter matching tests (Task 1 — S3V-05)
    // -----------------------------------------------------------------------

    #[test]
    fn test_matches_payload_filters_user_id_eq_match() {
        let payload = create_test_payload("data", Some("alice"), None, None);
        let filters = Filters {
            conditions: vec![FilterCondition {
                field: "user_id".to_string(),
                operator: FilterOperator::Eq,
                value: serde_json::json!("alice"),
            }],
            logic: FilterLogic::And,
        };
        assert!(
            matches_payload_filters(&payload, Some(&filters)),
            "user_id alice should match filter alice"
        );
    }

    #[test]
    fn test_matches_payload_filters_user_id_eq_no_match() {
        let payload = create_test_payload("data", Some("alice"), None, None);
        let filters = Filters {
            conditions: vec![FilterCondition {
                field: "user_id".to_string(),
                operator: FilterOperator::Eq,
                value: serde_json::json!("bob"),
            }],
            logic: FilterLogic::And,
        };
        assert!(
            !matches_payload_filters(&payload, Some(&filters)),
            "user_id alice should NOT match filter bob"
        );
    }

    #[test]
    fn test_matches_payload_filters_none_filters_matches_all() {
        let payload = create_test_payload("any data", Some("any_user"), None, None);
        assert!(
            matches_payload_filters(&payload, None),
            "None filters should match any payload"
        );
    }

    #[test]
    fn test_matches_payload_filters_and_logic() {
        let payload = create_test_payload("data", Some("alice"), Some("bot"), None);

        // Both conditions match → true
        let filters_match = Filters {
            conditions: vec![
                FilterCondition {
                    field: "user_id".to_string(),
                    operator: FilterOperator::Eq,
                    value: serde_json::json!("alice"),
                },
                FilterCondition {
                    field: "agent_id".to_string(),
                    operator: FilterOperator::Eq,
                    value: serde_json::json!("bot"),
                },
            ],
            logic: FilterLogic::And,
        };
        assert!(
            matches_payload_filters(&payload, Some(&filters_match)),
            "both conditions match → should return true"
        );

        // Second condition does not match → false
        let filters_no_match = Filters {
            conditions: vec![
                FilterCondition {
                    field: "user_id".to_string(),
                    operator: FilterOperator::Eq,
                    value: serde_json::json!("alice"),
                },
                FilterCondition {
                    field: "agent_id".to_string(),
                    operator: FilterOperator::Eq,
                    value: serde_json::json!("other"),
                },
            ],
            logic: FilterLogic::And,
        };
        assert!(
            !matches_payload_filters(&payload, Some(&filters_no_match)),
            "one condition does not match → should return false for AND logic"
        );
    }

    #[test]
    fn test_matches_payload_filters_extra_metadata_field() {
        let mut payload = create_test_payload("data", None, None, None);
        payload
            .metadata
            .insert("priority".to_string(), serde_json::json!("high"));
        let filters = Filters {
            conditions: vec![FilterCondition {
                field: "priority".to_string(),
                operator: FilterOperator::Eq,
                value: serde_json::json!("high"),
            }],
            logic: FilterLogic::And,
        };
        assert!(
            matches_payload_filters(&payload, Some(&filters)),
            "metadata field 'priority' = 'high' should match filter"
        );
    }

    // -----------------------------------------------------------------------
    // Task 2: new_for_test constructor + mocked AWS SDK tests
    // -----------------------------------------------------------------------

    impl S3VectorsStore {
        /// Construct a store with a pre-built client — for use in unit tests only.
        pub(crate) fn new_for_test(
            client: aws_sdk_s3vectors::Client,
            bucket: &str,
            index: &str,
            dimensions: usize,
        ) -> Self {
            Self {
                client,
                vector_bucket_name: bucket.to_string(),
                index_name: index.to_string(),
                dimensions,
                distance_metric: "cosine".to_string(),
            }
        }
    }

    #[tokio::test]
    async fn test_insert_calls_put_vectors() {
        use aws_smithy_mocks::{mock, mock_client, RuleMode};
        use aws_sdk_s3vectors::operation::put_vectors::PutVectorsOutput;

        let put_rule = mock!(aws_sdk_s3vectors::Client::put_vectors)
            .then_output(|| PutVectorsOutput::builder().build());

        let client = mock_client!(aws_sdk_s3vectors, RuleMode::Sequential, &[&put_rule]);
        let store = S3VectorsStore::new_for_test(client, "test-bucket", "test-index", 3);
        let payload = create_test_payload("hello", Some("alice"), None, None);
        let result = store.insert("id1", vec![0.1, 0.2, 0.3], payload).await;
        assert!(result.is_ok(), "insert should succeed: {:?}", result);
    }

    #[tokio::test]
    async fn test_search_returns_scored_results() {
        use aws_smithy_mocks::{mock, mock_client, RuleMode};
        use aws_sdk_s3vectors::operation::query_vectors::QueryVectorsOutput;
        use aws_sdk_s3vectors::types::QueryOutputVector;

        // Build a metadata document for "id1"
        let meta_payload = create_test_payload("hello world", Some("alice"), None, None);
        let meta_doc = payload_to_document(&meta_payload);

        let vec_result = QueryOutputVector::builder()
            .key("id1")
            .distance(0.1_f32)
            .metadata(meta_doc)
            .build()
            .expect("QueryOutputVector build");

        let query_rule = mock!(aws_sdk_s3vectors::Client::query_vectors)
            .then_output(move || {
                QueryVectorsOutput::builder()
                    .vectors(vec_result.clone())
                    .build()
                    .expect("QueryVectorsOutput build")
            });

        let client = mock_client!(aws_sdk_s3vectors, RuleMode::Sequential, &[&query_rule]);
        let store = S3VectorsStore::new_for_test(client, "test-bucket", "test-index", 3);
        let results = store.search(&[0.1, 0.2, 0.3], 5, None).await;
        assert!(results.is_ok(), "search should succeed: {:?}", results);
        let results = results.unwrap();
        assert_eq!(results.len(), 1, "should return exactly 1 result");
        assert_eq!(results[0].id, "id1");
        // score = 1.0 - distance = 1.0 - 0.1 = 0.9
        let score = results[0].score;
        assert!(
            (score - 0.9_f32).abs() < 1e-5,
            "score should be ~0.9, got {score}"
        );
    }

    #[tokio::test]
    async fn test_get_returns_some_when_found() {
        use aws_smithy_mocks::{mock, mock_client, RuleMode};
        use aws_sdk_s3vectors::operation::get_vectors::GetVectorsOutput;
        use aws_sdk_s3vectors::types::GetOutputVector;

        let meta_payload = create_test_payload("stored content", Some("u1"), None, None);
        let meta_doc = payload_to_document(&meta_payload);

        let get_vec = GetOutputVector::builder()
            .key("id1")
            .metadata(meta_doc)
            .build()
            .expect("GetOutputVector build");

        let get_rule = mock!(aws_sdk_s3vectors::Client::get_vectors)
            .then_output(move || {
                GetVectorsOutput::builder()
                    .vectors(get_vec.clone())
                    .build()
                    .expect("GetVectorsOutput build")
            });

        let client = mock_client!(aws_sdk_s3vectors, RuleMode::Sequential, &[&get_rule]);
        let store = S3VectorsStore::new_for_test(client, "test-bucket", "test-index", 3);
        let result = store.get("id1").await;
        assert!(result.is_ok(), "get should succeed: {:?}", result);
        let result = result.unwrap();
        assert!(result.is_some(), "should return Some when vector is found");
        let vsr = result.unwrap();
        assert_eq!(vsr.id, "id1");
        assert_eq!(vsr.payload.data, "stored content");
    }

    #[tokio::test]
    async fn test_get_returns_none_when_not_found() {
        use aws_smithy_mocks::{mock, mock_client, RuleMode};
        use aws_sdk_s3vectors::operation::get_vectors::GetVectorsOutput;

        let get_rule = mock!(aws_sdk_s3vectors::Client::get_vectors)
            .then_output(|| {
                GetVectorsOutput::builder()
                    .set_vectors(Some(vec![]))
                    .build()
                    .expect("empty GetVectorsOutput build")
            });

        let client = mock_client!(aws_sdk_s3vectors, RuleMode::Sequential, &[&get_rule]);
        let store = S3VectorsStore::new_for_test(client, "test-bucket", "test-index", 3);
        let result = store.get("missing").await;
        assert!(result.is_ok(), "get should succeed: {:?}", result);
        assert!(result.unwrap().is_none(), "should return None when not found");
    }

    #[tokio::test]
    async fn test_delete_calls_delete_vectors() {
        use aws_smithy_mocks::{mock, mock_client, RuleMode};
        use aws_sdk_s3vectors::operation::delete_vectors::DeleteVectorsOutput;

        let del_rule = mock!(aws_sdk_s3vectors::Client::delete_vectors)
            .then_output(|| DeleteVectorsOutput::builder().build());

        let client = mock_client!(aws_sdk_s3vectors, RuleMode::Sequential, &[&del_rule]);
        let store = S3VectorsStore::new_for_test(client, "test-bucket", "test-index", 3);
        let result = store.delete("id1").await;
        assert!(result.is_ok(), "delete should succeed: {:?}", result);
    }

    #[tokio::test]
    async fn test_collection_exists_returns_true() {
        use aws_smithy_mocks::{mock, mock_client, RuleMode};
        use aws_sdk_s3vectors::operation::get_index::GetIndexOutput;

        let get_index_rule = mock!(aws_sdk_s3vectors::Client::get_index)
            .then_output(|| GetIndexOutput::builder().build());

        let client = mock_client!(aws_sdk_s3vectors, RuleMode::Sequential, &[&get_index_rule]);
        let store = S3VectorsStore::new_for_test(client, "test-bucket", "test-index", 3);
        let result = store.collection_exists().await;
        assert!(result.is_ok(), "collection_exists should succeed: {:?}", result);
        assert!(result.unwrap(), "should return true when index exists");
    }

    #[tokio::test]
    async fn test_create_collection_calls_create_index() {
        use aws_smithy_mocks::{mock, mock_client, RuleMode};
        use aws_sdk_s3vectors::operation::create_index::CreateIndexOutput;

        let create_rule = mock!(aws_sdk_s3vectors::Client::create_index)
            .then_output(|| CreateIndexOutput::builder().build());

        let client = mock_client!(aws_sdk_s3vectors, RuleMode::Sequential, &[&create_rule]);
        let store = S3VectorsStore::new_for_test(client, "test-bucket", "test-index", 3);
        let result = store.create_collection().await;
        assert!(result.is_ok(), "create_collection should succeed: {:?}", result);
    }

    // -----------------------------------------------------------------------
    // Conformance suite (Plan 04 — D-09)
    // -----------------------------------------------------------------------

    /// Run the shared VectorStore conformance suite against S3VectorsStore using
    /// a fully-mocked AWS client.
    ///
    /// The conformance suite exercises all 7 primary VectorStore behaviors:
    /// insert, get (found/not-found), list (filtered/unfiltered), update,
    /// search, delete, and delete_all.
    ///
    /// The sequential mock ordering maps exactly to the conformance_suite call
    /// sequence:
    ///   1. put_vectors  — insert id-a
    ///   2. put_vectors  — insert id-b
    ///   3. get_vectors  — get id-a (found, returns metadata)
    ///   4. get_vectors  — get id-missing (not found, returns empty)
    ///   5. list_vectors — list all (returns id-a + id-b)
    ///   6. list_vectors — list with user_id=user-alice (client-side filtered)
    ///   7. get_vectors  — update id-a: fetch existing embedding (return_data=true)
    ///   8. put_vectors  — update id-a: write updated payload
    ///   9. get_vectors  — get id-a after update (returns updated metadata)
    ///  10. query_vectors — search
    ///  11. delete_vectors — delete id-b
    ///  12. get_vectors  — get id-b after delete (not found, returns empty)
    ///  13. list_vectors — delete_all(alice filter): list matching IDs
    ///  14. delete_vectors — delete_all(alice filter): batch delete
    ///  15. get_vectors  — get id-a after delete_all (not found, returns empty)
    #[tokio::test]
    async fn test_conformance_suite() {
        use aws_smithy_mocks::{mock, mock_client, RuleMode};
        use aws_sdk_s3vectors::operation::delete_vectors::DeleteVectorsOutput;
        use aws_sdk_s3vectors::operation::get_vectors::GetVectorsOutput;
        use aws_sdk_s3vectors::operation::list_vectors::ListVectorsOutput;
        use aws_sdk_s3vectors::operation::put_vectors::PutVectorsOutput;
        use aws_sdk_s3vectors::operation::query_vectors::QueryVectorsOutput;
        use aws_sdk_s3vectors::types::{
            GetOutputVector, ListOutputVector, QueryOutputVector, VectorData,
        };

        // Payload documents used in mock responses.
        let alice_payload = create_test_payload("memory A", Some("user-alice"), None, None);
        let alice_doc = payload_to_document(&alice_payload);
        let bob_payload = create_test_payload("memory B", Some("user-bob"), None, None);
        let bob_doc = payload_to_document(&bob_payload);
        let alice_updated_payload =
            create_test_payload("memory A updated", Some("user-alice"), None, None);
        let alice_updated_doc = payload_to_document(&alice_updated_payload);

        // 1. put_vectors — insert id-a
        let put_rule_1 = mock!(aws_sdk_s3vectors::Client::put_vectors)
            .then_output(|| PutVectorsOutput::builder().build());

        // 2. put_vectors — insert id-b
        let put_rule_2 = mock!(aws_sdk_s3vectors::Client::put_vectors)
            .then_output(|| PutVectorsOutput::builder().build());

        // 3. get_vectors — get id-a (found)
        let get_vec_a = GetOutputVector::builder()
            .key("id-a")
            .metadata(alice_doc.clone())
            .build()
            .expect("GetOutputVector id-a");
        let get_vec_a_clone = get_vec_a.clone();
        let get_rule_3 = mock!(aws_sdk_s3vectors::Client::get_vectors).then_output(move || {
            GetVectorsOutput::builder()
                .vectors(get_vec_a_clone.clone())
                .build()
                .expect("GetVectorsOutput id-a")
        });

        // 4. get_vectors — get id-missing (not found, empty)
        let get_rule_4 = mock!(aws_sdk_s3vectors::Client::get_vectors).then_output(|| {
            GetVectorsOutput::builder()
                .set_vectors(Some(vec![]))
                .build()
                .expect("empty GetVectorsOutput")
        });

        // 5. list_vectors — list all (returns id-a + id-b)
        let list_vec_a = ListOutputVector::builder()
            .key("id-a")
            .metadata(alice_doc.clone())
            .build()
            .expect("ListOutputVector id-a");
        let list_vec_b = ListOutputVector::builder()
            .key("id-b")
            .metadata(bob_doc.clone())
            .build()
            .expect("ListOutputVector id-b");
        let (list_vec_a2, list_vec_b2) = (list_vec_a.clone(), list_vec_b.clone());
        let list_rule_5 = mock!(aws_sdk_s3vectors::Client::list_vectors).then_output(move || {
            ListVectorsOutput::builder()
                .vectors(list_vec_a2.clone())
                .vectors(list_vec_b2.clone())
                .build()
                .expect("ListVectorsOutput all")
        });

        // 6. list_vectors — list with alice filter (returns both; client filters to alice only)
        let (list_vec_a3, list_vec_b3) = (list_vec_a.clone(), list_vec_b.clone());
        let list_rule_6 = mock!(aws_sdk_s3vectors::Client::list_vectors).then_output(move || {
            ListVectorsOutput::builder()
                .vectors(list_vec_a3.clone())
                .vectors(list_vec_b3.clone())
                .build()
                .expect("ListVectorsOutput alice filter")
        });

        // 7. get_vectors — update id-a: fetch existing embedding (return_data=true)
        let get_vec_a_with_data = GetOutputVector::builder()
            .key("id-a")
            .data(VectorData::Float32(vec![0.1_f32; 4]))
            .build()
            .expect("GetOutputVector id-a with data");
        let get_vec_a_with_data_clone = get_vec_a_with_data.clone();
        let get_rule_7 = mock!(aws_sdk_s3vectors::Client::get_vectors).then_output(move || {
            GetVectorsOutput::builder()
                .vectors(get_vec_a_with_data_clone.clone())
                .build()
                .expect("GetVectorsOutput id-a with data")
        });

        // 8. put_vectors — update id-a: write updated payload
        let put_rule_8 = mock!(aws_sdk_s3vectors::Client::put_vectors)
            .then_output(|| PutVectorsOutput::builder().build());

        // 9. get_vectors — get id-a after update (returns updated metadata)
        let get_vec_a_updated = GetOutputVector::builder()
            .key("id-a")
            .metadata(alice_updated_doc.clone())
            .build()
            .expect("GetOutputVector id-a updated");
        let get_vec_a_updated_clone = get_vec_a_updated.clone();
        let get_rule_9 = mock!(aws_sdk_s3vectors::Client::get_vectors).then_output(move || {
            GetVectorsOutput::builder()
                .vectors(get_vec_a_updated_clone.clone())
                .build()
                .expect("GetVectorsOutput id-a updated")
        });

        // 10. query_vectors — search
        let query_result = QueryOutputVector::builder()
            .key("id-a")
            .distance(0.1_f32)
            .metadata(alice_updated_doc.clone())
            .build()
            .expect("QueryOutputVector");
        let query_result_clone = query_result.clone();
        let query_rule_10 = mock!(aws_sdk_s3vectors::Client::query_vectors).then_output(
            move || {
                QueryVectorsOutput::builder()
                    .vectors(query_result_clone.clone())
                    .build()
                    .expect("QueryVectorsOutput")
            },
        );

        // 11. delete_vectors — delete id-b
        let del_rule_11 = mock!(aws_sdk_s3vectors::Client::delete_vectors)
            .then_output(|| DeleteVectorsOutput::builder().build());

        // 12. get_vectors — get id-b after delete (not found, empty)
        let get_rule_12 = mock!(aws_sdk_s3vectors::Client::get_vectors).then_output(|| {
            GetVectorsOutput::builder()
                .set_vectors(Some(vec![]))
                .build()
                .expect("empty GetVectorsOutput id-b after delete")
        });

        // 13. list_vectors — delete_all(alice filter): list matching (only id-a remains)
        let list_vec_a4 = list_vec_a.clone();
        let list_rule_13 =
            mock!(aws_sdk_s3vectors::Client::list_vectors).then_output(move || {
                ListVectorsOutput::builder()
                    .vectors(list_vec_a4.clone())
                    .build()
                    .expect("ListVectorsOutput for delete_all")
            });

        // 14. delete_vectors — delete_all(alice filter): batch delete id-a
        let del_rule_14 = mock!(aws_sdk_s3vectors::Client::delete_vectors)
            .then_output(|| DeleteVectorsOutput::builder().build());

        // 15. get_vectors — get id-a after delete_all (not found, empty)
        let get_rule_15 = mock!(aws_sdk_s3vectors::Client::get_vectors).then_output(|| {
            GetVectorsOutput::builder()
                .set_vectors(Some(vec![]))
                .build()
                .expect("empty GetVectorsOutput id-a after delete_all")
        });

        let client = mock_client!(
            aws_sdk_s3vectors,
            RuleMode::Sequential,
            &[
                &put_rule_1,
                &put_rule_2,
                &get_rule_3,
                &get_rule_4,
                &list_rule_5,
                &list_rule_6,
                &get_rule_7,
                &put_rule_8,
                &get_rule_9,
                &query_rule_10,
                &del_rule_11,
                &get_rule_12,
                &list_rule_13,
                &del_rule_14,
                &get_rule_15,
            ]
        );

        let store = S3VectorsStore::new_for_test(client, "test-bucket", "test-index", 4);
        crate::vector_stores::conformance::conformance_suite(&store).await;
    }

    #[tokio::test]
    async fn test_search_with_filter_passes_filter_document() {
        use aws_smithy_mocks::{mock, mock_client, RuleMode};
        use aws_sdk_s3vectors::operation::query_vectors::QueryVectorsOutput;
        use aws_sdk_s3vectors::types::QueryOutputVector;

        let meta_payload = create_test_payload("filtered result", Some("alice"), None, None);
        let meta_doc = payload_to_document(&meta_payload);

        let vec_result = QueryOutputVector::builder()
            .key("id2")
            .distance(0.2_f32)
            .metadata(meta_doc)
            .build()
            .expect("QueryOutputVector build");

        let query_rule = mock!(aws_sdk_s3vectors::Client::query_vectors)
            .then_output(move || {
                QueryVectorsOutput::builder()
                    .vectors(vec_result.clone())
                    .build()
                    .expect("QueryVectorsOutput build")
            });

        let client = mock_client!(aws_sdk_s3vectors, RuleMode::Sequential, &[&query_rule]);
        let store = S3VectorsStore::new_for_test(client, "test-bucket", "test-index", 3);

        let filters = Filters {
            conditions: vec![FilterCondition {
                field: "user_id".to_string(),
                operator: FilterOperator::Eq,
                value: serde_json::json!("alice"),
            }],
            logic: FilterLogic::And,
        };

        let results = store.search(&[0.1, 0.2, 0.3], 5, Some(&filters)).await;
        assert!(results.is_ok(), "search with filter should succeed: {:?}", results);
        let results = results.unwrap();
        assert_eq!(results.len(), 1, "should return the mocked result");
        assert_eq!(results[0].id, "id2");
    }
}
