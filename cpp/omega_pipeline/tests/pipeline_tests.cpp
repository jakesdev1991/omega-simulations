#include "omega_pipeline/block_predictor.hpp"
#include "omega_pipeline/evaluator.hpp"
#include "omega_pipeline/ingestion.hpp"
#include "omega_pipeline/npu_scorer.hpp"
#include "omega_pipeline/ring_buffer.hpp"

#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>

using namespace omega::pipeline;

namespace {

void write_u64_le(std::array<std::uint8_t, 32>& bytes, std::size_t offset, std::uint64_t value) {
  std::memcpy(bytes.data() + offset, &value, sizeof(value));
}

void test_ring_buffer() {
  SpscRingBuffer<int, 3> queue;
  assert(queue.empty());
  assert(queue.push(1));
  assert(queue.push(2));
  assert(!queue.push(3));
  auto first = queue.pop();
  auto second = queue.pop();
  auto third = queue.pop();
  assert(first && *first == 1);
  assert(second && *second == 2);
  assert(!third);
}

void test_ingestion_decode() {
  std::array<std::uint8_t, 32> bytes{};
  write_u64_le(bytes, 8, 1234);
  write_u64_le(bytes, 16, 5678);
  MarketUpdate update{};
  assert(decode_pool_update(bytes, PoolLayout{.reserve_in_offset = 8, .reserve_out_offset = 16}, 42, 99, update));
  assert(update.pool_id == 42);
  assert(update.slot == 99);
  assert(update.reserve_in == 1234.0F);
  assert(update.reserve_out == 5678.0F);
}

void test_pipeline_scores() {
  RouteBatch batch{};
  batch.count = 1;
  batch.id[0] = 7;
  batch.hops[0] = 1;
  batch.input_size[0] = 10.0F;
  batch.reserve_in[0][0] = 1000.0F;
  batch.reserve_out[0][0] = 1100.0F;
  batch.fee[0][0] = 0.003F;
  batch.liquidity_score[0] = 0.9F;
  batch.volatility_score[0] = 0.1F;
  batch.cu_cost[0] = 800'000.0F;

  evaluate_constant_product_batch(batch, 0.0001F);
  assert(batch.expected_profit[0] > 0.0F);

  apply_block_prediction(batch, MarketVelocity{.reserve_in_delta_per_ms = 0.0F,
                                               .reserve_out_delta_per_ms = 0.0F,
                                               .latency_ms = 100.0F});
  assert(std::fabs((batch.predicted_profit[0] - batch.expected_profit[0]) - 0.0001F) < 0.001F);

  HeuristicSurvivabilityScorer scorer;
  score_batch(batch, scorer, 100.0F, 0.25F);
  assert(batch.survivability[0] > 0.0F);
  assert(batch.survivability[0] <= 1.0F);
}

}  // namespace

int main() {
  test_ring_buffer();
  test_ingestion_decode();
  test_pipeline_scores();
  std::cout << "omega_pipeline_tests passed\n";
}
