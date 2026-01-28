# Project Aether: Distributed ML Inference Engine

A high-performance Pipeline Parallelism engine for sharding Transformers across distributed nodes.

## 🚀 Current Progress: Day 1 (The Plumbing)
- [x] Protobuf contract for Zero-Copy tensor transfer.
- [x] Standardized serialization/deserialization utility.
- [x] Basic gRPC Worker-Coordinator handshake.

## 📁 Repository Map
- `/protos`: The communication contract (.proto).
- `/src/common`: Shared utilities (Serialization, Config).
- `/src/worker`: The engine room for model shards.
- `/src/coordinator`: The brain that batches and routes requests.
- `/generated`: Python code auto-built from gRPC.