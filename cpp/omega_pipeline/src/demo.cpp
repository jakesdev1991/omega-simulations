#include "omega_pipeline/block_predictor.hpp"
#include "omega_pipeline/evaluator.hpp"
#include "omega_pipeline/npu_scorer.hpp"
#include "omega_pipeline/types.hpp"

#include <iostream>

using omega::pipeline::HeuristicSurvivabilityScorer;
using omega::pipeline::MarketVelocity;
using omega::pipeline::RouteBatch;

int main() {
  RouteBatch batch{};
  batch.count = 2;
  for (std::size_t i = 0; i < batch.count; ++i) {
    batch.id[i] = 1000 + i;
    batch.hops[i] = 2;
    batch.input_size[i] = 10.0F + static_cast<float>(i);
    batch.reserve_in[0][i] = 10'000.0F;
    batch.reserve_out[0][i] = 10'250.0F;
    batch.reserve_in[1][i] = 10'250.0F;
    batch.reserve_out[1][i] = 10'550.0F;
    batch.fee[0][i] = 0.0025F;
    batch.fee[1][i] = 0.0025F;
    batch.liquidity_score[i] = 0.85F;
    batch.volatility_score[i] = 0.15F;
    batch.cu_cost[i] = 780'000.0F;
  }

  omega::pipeline::evaluate_constant_product_batch(batch, 0.0001F);
  omega::pipeline::apply_block_prediction(batch, MarketVelocity{.reserve_in_delta_per_ms = 0.01F,
                                                                .reserve_out_delta_per_ms = -0.005F,
                                                                .latency_ms = 180.0F});
  HeuristicSurvivabilityScorer scorer;
  omega::pipeline::score_batch(batch, scorer, 180.0F, 0.42F);

  for (std::size_t i = 0; i < batch.count; ++i) {
    std::cout << "route=" << batch.id[i]
              << " expected=" << batch.expected_profit[i]
              << " predicted=" << batch.predicted_profit[i]
              << " survivability=" << batch.survivability[i]
              << " execute=" << omega::pipeline::should_execute(batch.predicted_profit[i], batch.survivability[i], batch.cu_cost[i])
              << '\n';
  }
}
