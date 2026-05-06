#include "omega_pipeline/block_predictor.hpp"

#include <algorithm>

namespace omega::pipeline {

float predict_reserve(float current, float delta_per_ms, float latency_ms) noexcept {
  return std::max(current + delta_per_ms * latency_ms, 1.0e-8F);
}

void apply_block_prediction(RouteBatch& batch, const MarketVelocity& velocity) noexcept {
  const auto safe_count = std::min(batch.count, kMaxBatchRoutes);
  for (std::size_t i = 0; i < safe_count; ++i) {
    float amount = batch.input_size[i];
    const float initial = amount;
    const auto hops = std::min<std::size_t>(batch.hops[i], kMaxHops);

    for (std::size_t hop = 0; hop < hops; ++hop) {
      const float projected_in = predict_reserve(batch.reserve_in[hop][i], velocity.reserve_in_delta_per_ms, velocity.latency_ms);
      const float projected_out = predict_reserve(batch.reserve_out[hop][i], velocity.reserve_out_delta_per_ms, velocity.latency_ms);
      const float net = amount * (1.0F - batch.fee[hop][i]);
      amount = projected_out * net / std::max(projected_in + net, 1.0e-8F);
    }

    batch.predicted_profit[i] = amount - initial;
  }
}

bool should_execute(float predicted_profit, float survivability, float cu_cost) noexcept {
  constexpr float kMinimumRiskAdjustedProfit = 0.001F;
  constexpr float kMaxComputeUnits = 1'400'000.0F;
  return predicted_profit * survivability > kMinimumRiskAdjustedProfit && cu_cost < kMaxComputeUnits;
}

}  // namespace omega::pipeline
