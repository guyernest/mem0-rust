//! Lambda wrapper for the mem0 MCP Memory Server
//!
//! Runs the MCP server as a background HTTP server and proxies Lambda requests to it.
//! Security (CORS, DNS rebinding, security headers) is handled by the SDK's Tower layers
//! applied automatically by StreamableHttpServer::start().

use lambda_http::{run, service_fn, Body, Error, Request, Response};
use mem0_rust::config::{HistoryStoreConfig, OpenAIEmbedderConfig, OpenAILLMConfig, S3VectorsConfig};
use mem0_rust::{EmbedderConfig, LLMConfig, Memory, MemoryConfig, VectorStoreConfig};
use once_cell::sync::OnceCell;
use pmcp::server::streamable_http_server::{StreamableHttpServer, StreamableHttpServerConfig};
use reqwest::Client;
use std::net::SocketAddr;
use tracing_subscriber::EnvFilter;

static BASE_URL: OnceCell<String> = OnceCell::new();
static HTTP: OnceCell<Client> = OnceCell::new();

/// Build a MemoryConfig from environment variables.
///
/// Secrets (OPENAI_API_KEY) are injected by the pmcp.run platform as env vars.
/// Use `cargo pmcp secret set mem0-rust/OPENAI_API_KEY --prompt --remote` to configure.
fn build_memory_config() -> MemoryConfig {
    // Validate required secret is present (gives actionable error if missing)
    pmcp::secrets::require("OPENAI_API_KEY")
        .expect("OPENAI_API_KEY secret not configured");

    let bucket = std::env::var("S3_VECTORS_BUCKET").expect(
        "S3_VECTORS_BUCKET env var is required. Set it in deploy/lib/stack.ts environment.",
    );
    let region = std::env::var("AWS_REGION")
        .ok()
        .or_else(|| std::env::var("AWS_DEFAULT_REGION").ok());
    let collection = std::env::var("MEM0_COLLECTION_NAME")
        .unwrap_or_else(|_| "mem0".to_string());

    MemoryConfig {
        embedder: EmbedderConfig::OpenAI(OpenAIEmbedderConfig::default()),
        vector_store: VectorStoreConfig::S3Vectors(S3VectorsConfig {
            bucket_name: bucket,
            region,
            index_name: collection.clone(),
            dimensions: 1536,
            distance_metric: Some("cosine".to_string()),
        }),
        llm: Some(LLMConfig::OpenAI(OpenAILLMConfig::default())),
        history_store: HistoryStoreConfig::default(), // Disabled for Lambda (no persistent filesystem)
        custom_prompts: None,
        reranker: None,
        version: "1.1".to_string(),
        collection_name: collection,
    }
}

async fn start_http_in_background() -> pmcp::Result<SocketAddr> {
    let config = build_memory_config();
    tracing::info!(
        bucket = %match &config.vector_store {
            VectorStoreConfig::S3Vectors(c) => c.bucket_name.as_str(),
            _ => "N/A",
        },
        collection = %config.collection_name,
        "Initializing mem0 Memory with S3 Vectors backend"
    );
    let memory = Memory::new(config)
        .await
        .map_err(|e| pmcp::Error::internal(e.to_string()))?;
    let server = mem0_mcp_core::build_memory_server(memory).await?;
    let server = std::sync::Arc::new(tokio::sync::Mutex::new(server));

    let addr: SocketAddr = "127.0.0.1:8080".parse().unwrap();
    let config = StreamableHttpServerConfig::stateless();
    let http_server = StreamableHttpServer::with_config(addr, server, config);

    let (bound, handle) = http_server.start().await?;
    tracing::info!("mem0 Memory MCP server started on {}", bound);

    tokio::spawn(async move {
        if let Err(e) = handle.await {
            tracing::error!("HTTP server error: {}", e);
        }
    });

    Ok(bound)
}

async fn ensure_server_started() -> Result<String, Error> {
    if let Some(url) = BASE_URL.get() {
        return Ok(url.clone());
    }

    let bound = start_http_in_background()
        .await
        .map_err(|e| lambda_http::Error::from(e.to_string()))?;

    let base = format!("http://{}", bound);
    let _ = BASE_URL.set(base.clone());
    let _ = HTTP.set(Client::builder().build().unwrap());
    Ok(base)
}

async fn handler(event: Request) -> Result<Response<Body>, Error> {
    let method = event.method().clone();
    let path_q = event
        .uri()
        .path_and_query()
        .map(|pq| pq.as_str().to_string())
        .unwrap_or_else(|| "/".to_string());

    // Health check
    if method.as_str() == "GET" {
        let body = serde_json::json!({
            "ok": true,
            "server": "mem0-memory",
            "message": "mem0 Memory MCP Server. POST JSON-RPC to '/' for MCP requests."
        })
        .to_string();
        return Ok(Response::builder()
            .status(200)
            .header("content-type", "application/json")
            .body(Body::Text(body))
            .unwrap());
    }

    let base = ensure_server_started().await?;
    let client = HTTP.get().expect("client");

    let url = format!("{}{}", base, path_q);
    let reqwest_method = reqwest::Method::from_bytes(method.as_str().as_bytes())
        .map_err(|e| lambda_http::Error::from(e.to_string()))?;

    let mut req = client.request(reqwest_method, &url);

    for (name, value) in event.headers() {
        if let Ok(val) = value.to_str() {
            if name.as_str().eq_ignore_ascii_case("host") {
                continue;
            }
            req = req.header(name.as_str(), val);
        }
    }

    let body_bytes = match event.body() {
        Body::Empty => Vec::new(),
        Body::Text(s) => s.as_bytes().to_vec(),
        Body::Binary(b) => b.clone(),
    };
    req = req.body(body_bytes);

    let resp = req
        .send()
        .await
        .map_err(|e| lambda_http::Error::from(e.to_string()))?;
    let status = resp.status();
    let headers = resp.headers().clone();
    let bytes = resp
        .bytes()
        .await
        .map_err(|e| lambda_http::Error::from(e.to_string()))?;

    let mut builder = Response::builder().status(status.as_u16());
    for (name, value) in headers.iter() {
        if let Ok(val) = value.to_str() {
            if name.as_str().eq_ignore_ascii_case("transfer-encoding")
                || name.as_str().eq_ignore_ascii_case("content-length")
            {
                continue;
            }
            builder = builder.header(name.as_str(), val);
        }
    }

    Ok(builder.body(Body::Binary(bytes.to_vec())).unwrap())
}

#[tokio::main]
async fn main() -> Result<(), Error> {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .with_ansi(false)
        .try_init();

    run(service_fn(handler)).await
}
