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
    let mut map: HashMap<String, Document> = HashMap::new();

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

    // memory_type: filterable (D-01) — stored in payload.metadata until Phase 2
    // adds it as a first-class Payload field. Check metadata HashMap and promote
    // it to a top-level filterable field.
    if let Some(mt) = payload.metadata.get("memory_type") {
        if let Some(s) = mt.as_str() {
            map.insert("memory_type".to_string(), Document::String(s.to_string()));
        }
    }

    // Non-filterable fields (D-02): data is declared non-filterable at create_index time
    map.insert("data".to_string(), Document::String(payload.data.clone()));

    // Remaining metadata entries (excluding memory_type already handled above)
    for (k, v) in &payload.metadata {
        if k != "memory_type" {
            map.insert(k.clone(), json_value_to_document(v));
        }
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

    // All remaining fields go into metadata
    let known_keys = ["data", "hash", "created_at", "user_id", "agent_id", "run_id"];
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

/// Check whether a `Payload` matches the given `Filters` using client-side evaluation.
///
/// S3 Vectors `list_vectors` does not support filter parameters. This function
/// replicates the `InMemoryStore::matches_filters` logic on the payload-side.
pub(crate) fn matches_payload_filters(payload: &Payload, filters: Option<&Filters>) -> bool {
    let Some(filters) = filters else {
        return true;
    };

    if filters.conditions.is_empty() {
        return true;
    }

    let results: Vec<bool> = filters
        .conditions
        .iter()
        .map(|cond| {
            let field_value = resolve_payload_field(payload, &cond.field);
            evaluate_payload_condition(field_value.as_ref(), &cond.operator, &cond.value)
        })
        .collect();

    match filters.logic {
        FilterLogic::And => results.iter().all(|&r| r),
        FilterLogic::Or => results.iter().any(|&r| r),
    }
}

/// Resolve a field name to its JSON value from a `Payload`.
///
/// Checks first-class Payload fields, then falls back to the metadata HashMap.
fn resolve_payload_field(payload: &Payload, field: &str) -> Option<serde_json::Value> {
    match field {
        "user_id" => payload.user_id.as_ref().map(|s| serde_json::Value::String(s.clone())),
        "agent_id" => payload.agent_id.as_ref().map(|s| serde_json::Value::String(s.clone())),
        "run_id" => payload.run_id.as_ref().map(|s| serde_json::Value::String(s.clone())),
        "hash" => Some(serde_json::Value::String(payload.hash.clone())),
        "data" => Some(serde_json::Value::String(payload.data.clone())),
        _ => payload.metadata.get(field).cloned(),
    }
}

/// Evaluate a single filter condition against an optional field value.
fn evaluate_payload_condition(
    field_value: Option<&serde_json::Value>,
    operator: &FilterOperator,
    filter_value: &serde_json::Value,
) -> bool {
    match operator {
        FilterOperator::Eq => field_value == Some(filter_value),
        FilterOperator::Ne => field_value != Some(filter_value),
        FilterOperator::Gt => compare_numeric(field_value, filter_value, |a, b| a > b),
        FilterOperator::Gte => compare_numeric(field_value, filter_value, |a, b| a >= b),
        FilterOperator::Lt => compare_numeric(field_value, filter_value, |a, b| a < b),
        FilterOperator::Lte => compare_numeric(field_value, filter_value, |a, b| a <= b),
        FilterOperator::In => {
            if let Some(arr) = filter_value.as_array() {
                field_value.map(|v| arr.contains(v)).unwrap_or(false)
            } else {
                false
            }
        }
        FilterOperator::Nin => {
            if let Some(arr) = filter_value.as_array() {
                field_value.map(|v| !arr.contains(v)).unwrap_or(true)
            } else {
                true
            }
        }
        FilterOperator::Contains => {
            if let (Some(field_str), Some(filter_str)) = (
                field_value.and_then(|v| v.as_str()),
                filter_value.as_str(),
            ) {
                field_str.contains(filter_str)
            } else {
                false
            }
        }
        FilterOperator::IContains => {
            if let (Some(field_str), Some(filter_str)) = (
                field_value.and_then(|v| v.as_str()),
                filter_value.as_str(),
            ) {
                field_str.to_lowercase().contains(&filter_str.to_lowercase())
            } else {
                false
            }
        }
    }
}

/// Compare two JSON values as f64 numbers using the provided comparator.
fn compare_numeric<F>(
    field_value: Option<&serde_json::Value>,
    filter_value: &serde_json::Value,
    cmp: F,
) -> bool
where
    F: Fn(f64, f64) -> bool,
{
    let field_num = field_value.and_then(|v| v.as_f64());
    let filter_num = filter_value.as_f64();
    match (field_num, filter_num) {
        (Some(a), Some(b)) => cmp(a, b),
        _ => false,
    }
}

// ---------------------------------------------------------------------------
// VectorStore trait implementation (stub — Plan 02 fills in real bodies)
// ---------------------------------------------------------------------------

#[async_trait]
impl VectorStore for S3VectorsStore {
    async fn insert(
        &self,
        _id: &str,
        _embedding: Vec<f32>,
        _payload: Payload,
    ) -> Result<(), VectorStoreError> {
        Err(VectorStoreError::Connection(
            "S3VectorsStore::insert not yet implemented — will be filled in Plan 02".into(),
        ))
    }

    async fn search(
        &self,
        _embedding: &[f32],
        _limit: usize,
        _filters: Option<&Filters>,
    ) -> Result<Vec<VectorSearchResult>, VectorStoreError> {
        Err(VectorStoreError::Connection(
            "S3VectorsStore::search not yet implemented — will be filled in Plan 02".into(),
        ))
    }

    async fn get(&self, _id: &str) -> Result<Option<VectorSearchResult>, VectorStoreError> {
        Err(VectorStoreError::Connection(
            "S3VectorsStore::get not yet implemented — will be filled in Plan 02".into(),
        ))
    }

    async fn delete(&self, _id: &str) -> Result<(), VectorStoreError> {
        Err(VectorStoreError::Connection(
            "S3VectorsStore::delete not yet implemented — will be filled in Plan 02".into(),
        ))
    }

    async fn update(
        &self,
        _id: &str,
        _embedding: Option<Vec<f32>>,
        _payload: Payload,
    ) -> Result<(), VectorStoreError> {
        Err(VectorStoreError::Connection(
            "S3VectorsStore::update not yet implemented — will be filled in Plan 02".into(),
        ))
    }

    async fn list(
        &self,
        _filters: Option<&Filters>,
        _limit: usize,
    ) -> Result<Vec<VectorSearchResult>, VectorStoreError> {
        Err(VectorStoreError::Connection(
            "S3VectorsStore::list not yet implemented — will be filled in Plan 02".into(),
        ))
    }

    async fn delete_all(&self, _filters: Option<&Filters>) -> Result<usize, VectorStoreError> {
        Err(VectorStoreError::Connection(
            "S3VectorsStore::delete_all not yet implemented — will be filled in Plan 02".into(),
        ))
    }

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
        let mut metadata = HashMap::new();
        metadata.insert(
            "memory_type".to_string(),
            serde_json::Value::String("long_term".to_string()),
        );
        let payload = Payload {
            data: "data".to_string(),
            hash: "h".to_string(),
            created_at: Utc::now(),
            user_id: None,
            agent_id: None,
            run_id: None,
            metadata,
        };
        let doc = payload_to_document(&payload);
        let Document::Object(ref map) = doc else {
            panic!("expected Document::Object");
        };
        // memory_type should be promoted to top-level filterable field
        assert!(
            matches!(map.get("memory_type"), Some(Document::String(s)) if s == "long_term"),
            "memory_type should be filterable at top level"
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
        let mut payload = make_payload("data", None);
        payload
            .metadata
            .insert("memory_type".to_string(), serde_json::json!("long_term"));
        let filters = Filters {
            conditions: vec![FilterCondition {
                field: "memory_type".to_string(),
                operator: FilterOperator::Eq,
                value: serde_json::json!("long_term"),
            }],
            logic: FilterLogic::And,
        };
        assert!(matches_payload_filters(&payload, Some(&filters)));
    }

    #[test]
    fn test_batch_limit_constant() {
        assert_eq!(BATCH_LIMIT, 500);
    }
}
