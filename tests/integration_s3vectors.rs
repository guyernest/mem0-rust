//! Integration tests for S3VectorsStore against a real S3 Vectors endpoint.
//!
//! Run with: cargo test --features s3vectors,integration-tests --test integration_s3vectors
//!
//! Requires AWS credentials and a real S3 Vectors bucket. These tests are
//! excluded from CI — they are intended for manual verification only.

#![cfg(feature = "integration-tests")]

// TODO (Phase 1 Plan 04): implement conformance suite tests here once
// the shared VectorStore conformance module exists.
