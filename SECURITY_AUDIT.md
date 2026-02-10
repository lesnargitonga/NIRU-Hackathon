# Security Architecture Review: Operation Sentinel

Status: Amber
Date: 2026-02-10

This document captures a focused security review of the local stack, with emphasis on authentication, inter process communication, and operational robustness.

## Findings

### 1. Command and telemetry transport

Finding: A file based handoff was previously used for command overrides and coordination.

Risk: Filesystem based IPC is prone to race conditions, provides poor observability, and is difficult to secure.

Remediation: Replaced file based IPC with Redis pubsub channels for commands and telemetry.

### 2. API authentication coverage

Finding: The backend supports API key authentication, but authorization must be consistently enforced across all endpoints.

Risk: Unauthenticated access can expose telemetry and enable unsafe command execution.

Remediation: Added role checks on previously unauthenticated endpoints and documented required headers.

### 3. Operational robustness of the PX4 bridge script

Finding: Control and data collection scripts must avoid silent failure paths.

Risk: Suppressing exceptions in flight control loops can hide failures and complicate incident analysis.

Remediation: Removed duplicated code blocks, improved configurability, and updated the bridge to publish telemetry and consume commands via Redis.

### 4. Frontend configuration

Finding: Development proxies and hard coded endpoints tend to fail when the deployment environment changes.

Risk: A build that works only in a narrow development configuration can break during evaluation.

Remediation: Updated frontend configuration to support environment driven backend URLs and API key injection.

## Current posture

1. Backend stack starts via Docker Compose.
2. Backend API requires `X-API-Key` for protected endpoints.
3. Redis is used for command and telemetry exchange.

## Remaining work

1. Add integration tests for end to end command propagation.
2. Add structured logging and explicit error handling around control loop failures.
