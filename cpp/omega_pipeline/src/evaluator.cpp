#include "omega_pipeline/evaluator.hpp"

#include <algorithm>
#include <cmath>

#if defined(OMEGA_PIPELINE_ENABLE_AVX512)
#include <immintrin.h>
#endif

namespace omega::pipeline {
namespace {

constexpr float kMinDenominator = 1.0e-8F;

float evaluate_scalar_hop(float input, float reserve_in, float reserve_out, float fee) noexcept {
  const float denominator = std::max(reserve_in + input * (1.0F - fee), kMinDenominator);
  const float net_input = input * (1.0F - fee);
  return reserve_out * net_input / denominator;
}

#if defined(OMEGA_PIPELINE_ENABLE_AVX512)
__m512 fast_rcp(__m512 value) noexcept {
  const __m512 min_denominator = _mm512_set1_ps(kMinDenominator);
  value = _mm512_max_ps(value, min_denominator);
  const __m512 estimate = _mm512_rcp14_ps(value);
  const __m512 two = _mm512_set1_ps(2.0F);
  return _mm512_mul_ps(estimate, _mm512_fnmadd_ps(value, estimate, two));
}
#endif

}  // namespace

void evaluate_constant_product_batch(RouteBatch& batch, float priority_fee) {
  const auto safe_count = std::min(batch.count, kMaxBatchRoutes);
  std::size_t i = 0;

#if defined(OMEGA_PIPELINE_ENABLE_AVX512)
  const __m512 one = _mm512_set1_ps(1.0F);
  const __m512 priority = _mm512_set1_ps(priority_fee);
  for (; i + 16 <= safe_count; i += 16) {
    __m512 amount = _mm512_loadu_ps(batch.input_size.data() + i);
    __m512 initial = amount;

    for (std::size_t hop = 0; hop < kMaxHops; ++hop) {
      __mmask16 active = 0;
      for (std::size_t lane = 0; lane < 16; ++lane) {
        if (batch.hops[i + lane] > hop) {
          active |= static_cast<__mmask16>(1U << lane);
        }
      }
      if (active == 0) {
        continue;
      }

      const __m512 reserve_in = _mm512_maskz_loadu_ps(active, batch.reserve_in[hop].data() + i);
      const __m512 reserve_out = _mm512_maskz_loadu_ps(active, batch.reserve_out[hop].data() + i);
      const __m512 fee = _mm512_maskz_loadu_ps(active, batch.fee[hop].data() + i);
      const __m512 net = _mm512_mul_ps(amount, _mm512_sub_ps(one, fee));
      const __m512 denominator = _mm512_add_ps(reserve_in, net);
      const __m512 output = _mm512_mul_ps(_mm512_mul_ps(reserve_out, net), fast_rcp(denominator));
      amount = _mm512_mask_mov_ps(amount, active, output);
    }

    const __m512 profit = _mm512_sub_ps(_mm512_sub_ps(amount, initial), priority);
    _mm512_storeu_ps(batch.expected_profit.data() + i, profit);
  }
#endif

  for (; i < safe_count; ++i) {
    float amount = batch.input_size[i];
    const float initial = amount;
    const auto hops = std::min<std::size_t>(batch.hops[i], kMaxHops);
    for (std::size_t hop = 0; hop < hops; ++hop) {
      amount = evaluate_scalar_hop(amount, batch.reserve_in[hop][i], batch.reserve_out[hop][i], batch.fee[hop][i]);
    }
    batch.expected_profit[i] = amount - initial - priority_fee;
  }
}

}  // namespace omega::pipeline
