//! Lambda wrapper for the mem0 MCP Memory Server
//!
//! Runs the MCP server as a background HTTP server and proxies Lambda requests to it.
//! Security (CORS, DNS rebinding, security headers) is handled by the SDK's Tower layers
//! applied automatically by StreamableHttpServer::start().

use aws_config::BehaviorVersion;
use aws_sdk_secretsmanager::Client as SecretsClient;
use lambda_http::{run, service_fn, Body, Error, Request, Response};
use mem0_rust::config::{OpenAIEmbedderConfig, OpenAILLMConfig, S3VectorsConfig};
use mem0_rust::{EmbedderConfig, LLMConfig, Memory, MemoryConfig, VectorStoreConfig};
use once_cell::sync::OnceCell;
use pmcp::server::streamable_http_server::{StreamableHttpServer, StreamableHttpServerConfig};
use reqwest::Client;
use std::collections::HashMap;
use std::net::SocketAddr;
use tracing_subscriber::EnvFilter;

static BASE_URL: OnceCell<String> = OnceCell::new();
static HTTP: OnceCell<Client> = OnceCell::new();

/// Load secrets from the pmcp.run org-level Secrets Manager secret.
///
/// Reads `PMCP_SECRETS_PATH` (e.g., "pmcp/orgs/{org_id}/credentials") and
/// `PMCP_SERVER_ID` (e.g., "mem0-rust") to extract server-specific secrets.
/// Sets them as environment variables so downstream code (e.g., OpenAI SDK)
/// can read them via std::env::var.
async fn load_pmcp_secrets() {
    let secrets_path = match std::env::var("PMCP_SECRETS_PATH") {
        Ok(path) => path,
        Err(_) => {
            tracing::debug!("PMCP_SECRETS_PATH not set, skipping secrets loading");
            return;
        }
    };
    let server_id = std::env::var("PMCP_SERVER_ID").unwrap_or_else(|_| "mem0-rust".to_string());

    tracing::info!(
        path = %secrets_path,
        server_id = %server_id,
        "Loading secrets from org-level Secrets Manager"
    );

    let aws_config = aws_config::load_defaults(BehaviorVersion::latest()).await;
    let client = SecretsClient::new(&aws_config);

    let response = match client
        .get_secret_value()
        .secret_id(&secrets_path)
        .send()
        .await
    {
        Ok(resp) => resp,
        Err(e) => {
            tracing::error!(error = %e, "Failed to fetch org secret from Secrets Manager");
            return;
        }
    };

    let secret_string = match response.secret_string() {
        Some(s) => s,
        None => {
            tracing::error!("Org secret has no string value");
            return;
        }
    };

    // Parse as { "server-id": { "KEY": "value" } }
    let all_secrets: HashMap<String, serde_json::Value> = match serde_json::from_str(secret_string)
    {
        Ok(v) => v,
        Err(e) => {
            tracing::error!(error = %e, "Org secret is not valid JSON");
            return;
        }
    };

    // Extract secrets for this server
    if let Some(serde_json::Value::Object(server_secrets)) = all_secrets.get(&server_id) {
        let mut count = 0;
        for (key, value) in server_secrets {
            if key.starts_with('_') {
                continue;
            }
            if let Some(s) = value.as_str() {
                if !s.is_empty() && s != "PLACEHOLDER_UPDATE_REQUIRED" {
                    // SAFETY: single-threaded init before any concurrent access
                    unsafe { std::env::set_var(key, s) };
                    count += 1;
                }
            }
        }
        tracing::info!(count, server_id = %server_id, "Loaded secrets into environment");
    } else {
        tracing::warn!(
            server_id = %server_id,
            "No secrets found for this server in org secret"
        );
    }
}

/// Build a MemoryConfig from environment variables.
///
/// Secrets (OPENAI_API_KEY) are loaded from Secrets Manager by `load_pmcp_secrets()`
/// before this function is called.
fn build_memory_config() -> MemoryConfig {
    let bucket = std::env::var("S3_VECTORS_BUCKET").expect(
        "S3_VECTORS_BUCKET env var is required. Set it in .pmcp/deploy.toml [environment] \
         to a globally unique name like 'mem0-vectors-{account_id}-{region}'."
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
        history_db_path: None, // DEP-02: Disabled for Lambda (no writable filesystem)
        custom_prompts: None,
        reranker: None,
        version: "1.1".to_string(),
        collection_name: collection,
    }
}

async fn start_http_in_background() -> pmcp::Result<SocketAddr> {
    // Load secrets from pmcp.run org-level Secrets Manager before building config
    load_pmcp_secrets().await;
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
    // stateless() uses AllowedOrigins::any() — safe behind Lambda/API Gateway proxy
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

    // Copy headers (skip host)
    for (name, value) in event.headers() {
        if let Ok(val) = value.to_str() {
            if name.as_str().eq_ignore_ascii_case("host") {
                continue;
            }
            req = req.header(name.as_str(), val);
        }
    }

    // Copy body
    let body_bytes = match event.body() {
        Body::Empty => Vec::new(),
        Body::Text(s) => s.as_bytes().to_vec(),
        Body::Binary(b) => b.clone(),
    };
    req = req.body(body_bytes);

    // Forward and return response
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
