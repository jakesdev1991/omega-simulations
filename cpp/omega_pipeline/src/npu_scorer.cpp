#include "omega_pipeline/npu_scorer.hpp"

#include <algorithm>
#include <cmath>

namespace omega::pipeline {
namespace {

float clamp01(float value) noexcept {
  return std::clamp(value, 0.0F, 1.0F);
}

}  // namespace

float HeuristicSurvivabilityScorer::score(const FeatureVector& features) noexcept {
  const float liquidity = std::max(features.liquidity_score, 0.0F);
  const float volatility_penalty = 1.0F + std::max(features.volatility_score, 0.0F);
  const float latency_penalty = 1.0F + std::max(features.latency_ms, 0.0F) / 400.0F;
  const float compute_penalty = 1.0F + std::max(features.cu_cost, 0.0F) / 1'400'000.0F;
  const float slot_penalty = 1.0F + clamp01(features.slot_progress) * 0.35F;
  return clamp01(liquidity / (volatility_penalty * latency_penalty * compute_penalty * slot_penalty));
}

XdnaNpuScorer::XdnaNpuScorer(HeuristicSurvivabilityScorer fallback) : fallback_(fallback) {}

bool XdnaNpuScorer::load_model(const char* compiled_model_path) noexcept {
  model_ready_ = compiled_model_path != nullptr && compiled_model_path[0] != '\0';
  return model_ready_;
}

float XdnaNpuScorer::score(const FeatureVector& features) noexcept {
  if (!model_ready_) {
    return fallback_.score(features);
  }

  // Placeholder for a compiled XDNA/MLIR-AIE runtime call. Keep the same feature
  // contract as the fallback so callers can switch to NPU scoring without changing
  // the hot-path pipeline.
  return fallback_.score(features);
}

void score_batch(RouteBatch& batch, SurvivabilityScorer& scorer, float latency_ms, float slot_progress) noexcept {
  const auto safe_count = std::min(batch.count, kMaxBatchRoutes);
  for (std::size_t i = 0; i < safe_count; ++i) {
    batch.survivability[i] = scorer.score(FeatureVector{
        .predicted_profit = batch.predicted_profit[i],
        .liquidity_score = batch.liquidity_score[i],
        .volatility_score = batch.volatility_score[i],
        .cu_cost = batch.cu_cost[i],
        .latency_ms = latency_ms,
        .slot_progress = slot_progress,
    });
  }
}

}  // namespace omega::pipeline
