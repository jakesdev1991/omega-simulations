#pragma once

#include "omega_pipeline/types.hpp"

#include <span>

namespace omega::pipeline {

class SurvivabilityScorer {
 public:
  virtual ~SurvivabilityScorer() = default;
  virtual float score(const FeatureVector& features) noexcept = 0;
};

class HeuristicSurvivabilityScorer final : public SurvivabilityScorer {
 public:
  float score(const FeatureVector& features) noexcept override;
};

class XdnaNpuScorer final : public SurvivabilityScorer {
 public:
  explicit XdnaNpuScorer(HeuristicSurvivabilityScorer fallback = {});
  bool load_model(const char* compiled_model_path) noexcept;
  float score(const FeatureVector& features) noexcept override;

 private:
  HeuristicSurvivabilityScorer fallback_{};
  bool model_ready_{false};
};

void score_batch(RouteBatch& batch, SurvivabilityScorer& scorer, float latency_ms, float slot_progress) noexcept;

}  // namespace omega::pipeline
