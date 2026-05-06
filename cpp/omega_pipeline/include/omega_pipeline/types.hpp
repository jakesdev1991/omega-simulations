#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace omega::pipeline {

inline constexpr std::size_t kMaxBatchRoutes = 4096;
inline constexpr std::size_t kMaxHops = 4;

struct alignas(64) RouteBatch {
  std::array<std::uint64_t, kMaxBatchRoutes> id{};
  std::array<std::uint8_t, kMaxBatchRoutes> hops{};
  std::array<float, kMaxBatchRoutes> input_size{};
  std::array<float, kMaxBatchRoutes> liquidity_score{};
  std::array<float, kMaxBatchRoutes> volatility_score{};
  std::array<float, kMaxBatchRoutes> cu_cost{};
  std::array<float, kMaxBatchRoutes> expected_profit{};
  std::array<float, kMaxBatchRoutes> predicted_profit{};
  std::array<float, kMaxBatchRoutes> survivability{};
  std::array<std::array<float, kMaxBatchRoutes>, kMaxHops> reserve_in{};
  std::array<std::array<float, kMaxBatchRoutes>, kMaxHops> reserve_out{};
  std::array<std::array<float, kMaxBatchRoutes>, kMaxHops> fee{};
  std::size_t count{0};
};

struct MarketUpdate {
  std::uint64_t pool_id{};
  std::uint64_t slot{};
  float reserve_in{};
  float reserve_out{};
};

struct MarketVelocity {
  float reserve_in_delta_per_ms{};
  float reserve_out_delta_per_ms{};
  float latency_ms{200.0F};
};

struct FeatureVector {
  float predicted_profit{};
  float liquidity_score{};
  float volatility_score{};
  float cu_cost{};
  float latency_ms{};
  float slot_progress{};
};

}  // namespace omega::pipeline
