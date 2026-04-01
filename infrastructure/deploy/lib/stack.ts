import * as cdk from 'aws-cdk-lib';
import * as dsql from 'aws-cdk-lib/aws-dsql';
import * as ssm from 'aws-cdk-lib/aws-ssm';
import { Construct } from 'constructs';

/**
 * Mem0Rust Infrastructure Stack
 *
 * Provisions Aurora DSQL cluster for mem0-rust history storage and exports
 * the cluster endpoint to SSM Parameter Store for the MCP server stack to consume.
 *
 * Deploy this stack before `cargo pmcp deploy`:
 *   cd infrastructure/deploy && npm install && npx cdk deploy --profile ze-kasher-dev
 */
export class Mem0RustInfraStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

    // ========================================================================
    // Aurora DSQL Cluster
    // Single-region cluster for mem0-rust history storage.
    // Deletion protection disabled for dev — enable for production.
    // ========================================================================
    const cluster = new dsql.CfnCluster(this, 'DsqlCluster', {
      deletionProtectionEnabled: false,
    });
    cluster.applyRemovalPolicy(cdk.RemovalPolicy.DESTROY);

    // ========================================================================
    // SSM Parameter — endpoint for MCP server stack to discover at deploy time
    // Path: /pmcp/mem0-rust/dsql-endpoint
    // ========================================================================
    new ssm.StringParameter(this, 'DsqlEndpointParam', {
      parameterName: '/pmcp/mem0-rust/dsql-endpoint',
      stringValue: cluster.attrEndpoint,
      description: 'Aurora DSQL cluster endpoint for mem0-rust history store',
    });

    // ========================================================================
    // Outputs
    // ========================================================================
    new cdk.CfnOutput(this, 'ClusterEndpoint', {
      value: cluster.attrEndpoint,
      description: 'Aurora DSQL cluster connection endpoint',
    });

    new cdk.CfnOutput(this, 'ClusterArn', {
      value: cluster.attrResourceArn,
      description: 'Aurora DSQL cluster ARN (use for scoped IAM policies)',
    });
  }
}
