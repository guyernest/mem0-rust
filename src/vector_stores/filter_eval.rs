//! Shared client-side filter evaluation for vector store backends.
//!
//! Both `InMemoryStore` and `S3VectorsStore` need to evaluate `Filters` against
//! `Payload` structs client-side (InMemory does all filtering locally; S3 Vectors'
//! `list_vectors` API doesn't support server-side filters). This module provides
//! the shared implementation to avoid duplication.

use crate::models::{FilterLogic, FilterOperator, Filters, Payload, REQUEST_ID_FIELD};

/// Check whether a `Payload` matches the given `Filters`.
///
/// Returns `true` if filters is `None` or empty.
/// Uses short-circuit evaluation for AND (stops on first `false`)
/// and OR (stops on first `true`).
pub fn matches_filters(payload: &Payload, filters: Option<&Filters>) -> bool {
    let Some(filters) = filters else {
        return true;
    };

    if filters.conditions.is_empty() {
        return true;
    }

    match filters.logic {
        FilterLogic::And => filters.conditions.iter().all(|cond| {
            let value = resolve_payload_field(payload, &cond.field);
            evaluate_condition(value.as_ref(), &cond.operator, &cond.value)
        }),
        FilterLogic::Or => filters.conditions.iter().any(|cond| {
            let value = resolve_payload_field(payload, &cond.field);
            evaluate_condition(value.as_ref(), &cond.operator, &cond.value)
        }),
    }
}

/// Resolve a field name to its JSON value from a `Payload`.
///
/// Checks first-class Payload fields first, then falls back to the metadata HashMap.
fn resolve_payload_field(payload: &Payload, field: &str) -> Option<serde_json::Value> {
    match field {
        "user_id" => payload
            .user_id
            .as_ref()
            .map(|s| serde_json::Value::String(s.clone())),
        "agent_id" => payload
            .agent_id
            .as_ref()
            .map(|s| serde_json::Value::String(s.clone())),
        REQUEST_ID_FIELD => payload
            .request_id
            .as_ref()
            .map(|s| serde_json::Value::String(s.clone())),
        "memory_type" => payload
            .memory_type
            .map(|mt| serde_json::Value::String(mt.to_string())),
        "hash" => Some(serde_json::Value::String(payload.hash.clone())),
        "data" => Some(serde_json::Value::String(payload.data.clone())),
        _ => payload.metadata.get(field).cloned(),
    }
}

/// Evaluate a single filter condition against an optional field value.
fn evaluate_condition(
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
            if let (Some(field_str), Some(filter_str)) =
                (field_value.and_then(|v| v.as_str()), filter_value.as_str())
            {
                field_str.contains(filter_str)
            } else {
                false
            }
        }
        FilterOperator::IContains => {
            if let (Some(field_str), Some(filter_str)) =
                (field_value.and_then(|v| v.as_str()), filter_value.as_str())
            {
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
