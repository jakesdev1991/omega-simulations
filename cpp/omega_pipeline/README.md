# Omega Native Pipeline Skeleton

This directory contains a dependency-light C++20 skeleton for the latency-aware route pipeline discussed in the Omega Protocol notes.  It is intentionally structured as a sandbox/replay component, not as a live trading executor: it does not manage private keys, build transactions, or submit bundles.

## Pipeline stages

1. **Yellowstone-style ingestion adapter** decodes binary account updates into `MarketUpdate` records and publishes them into a single-producer/single-consumer ring buffer.
2. **Structure-of-arrays route batches** keep hot-path fields contiguous for AVX-512-friendly evaluation.
3. **Constant-product evaluator** processes multi-hop route batches with an AVX-512 implementation when the compiler supports it and a scalar fallback everywhere else.
4. **Block-level predictor** projects reserves by estimated latency before scoring a route.
5. **Survivability scorer** exposes a stable interface for an eventual XDNA/MLIR-AIE model while providing a deterministic heuristic fallback for replay tests.

## Build and test

```bash
cmake -S cpp/omega_pipeline -B /tmp/omega_pipeline_build
cmake --build /tmp/omega_pipeline_build
ctest --test-dir /tmp/omega_pipeline_build --output-on-failure
```

Disable AVX-512-specific compilation if building on a compiler or host that should use only the scalar path:

```bash
cmake -S cpp/omega_pipeline -B /tmp/omega_pipeline_build -DOMEGA_PIPELINE_ENABLE_AVX512=OFF
```

## Integration notes

- Replace `YellowstoneIngestionClient::ingest_account_update` with a real async gRPC completion-queue client once the target Geyser proto and provider authentication are selected.
- Keep the gRPC callback limited to decode-and-publish work; route generation, prediction, and scoring should happen on dedicated consumer threads.
- Replace `XdnaNpuScorer::score` with the compiled model runtime call when an XDNA deployment artifact is available. The feature contract is captured by `FeatureVector`.
- Preserve the sandbox boundary during development: replay historical account updates and shadow live streams before wiring any executor.
