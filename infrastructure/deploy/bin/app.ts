#!/usr/bin/env node
import * as cdk from 'aws-cdk-lib';
import { Mem0RustInfraStack } from '../lib/stack';

const app = new cdk.App();
const region = process.env.CDK_DEFAULT_REGION || process.env.AWS_REGION || 'us-east-1';

const stack = new Mem0RustInfraStack(app, 'mem0-rust-infra', {
  env: { account: process.env.CDK_DEFAULT_ACCOUNT, region },
  description: 'Aurora DSQL infrastructure for mem0-rust history store',
});

cdk.Tags.of(stack).add('project', 'agents');

app.synth();
