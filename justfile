# mem0-rust project commands

# Deploy DSQL infrastructure stack (run before cargo pmcp deploy)
deploy-infra:
    cd infrastructure/deploy && npm install && npx cdk deploy --require-approval never --profile ze-kasher-dev

# Deploy MCP server (requires deploy-infra to have run at least once)
deploy-server:
    cargo pmcp deploy

# Deploy everything in order
deploy-all: deploy-infra deploy-server
