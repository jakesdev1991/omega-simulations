#pragma once

#include "omega_pipeline/types.hpp"

namespace omega::pipeline {

float predict_reserve(float current, float delta_per_ms, float latency_ms) noexcept;
void apply_block_prediction(RouteBatch& batch, const MarketVelocity& velocity) noexcept;
bool should_execute(float predicted_profit, float survivability, float cu_cost) noexcept;

}  // namespace omega::pipeline
