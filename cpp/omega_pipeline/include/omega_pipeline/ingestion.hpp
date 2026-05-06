#pragma once

#include "omega_pipeline/ring_buffer.hpp"
#include "omega_pipeline/types.hpp"

#include <cstdint>
#include <span>

namespace omega::pipeline {

struct PoolLayout {
  std::size_t reserve_in_offset{};
  std::size_t reserve_out_offset{};
};

class MarketUpdateSink {
 public:
  virtual ~MarketUpdateSink() = default;
  virtual bool publish(const MarketUpdate& update) noexcept = 0;
};

template <std::size_t Capacity>
class RingMarketUpdateSink final : public MarketUpdateSink {
 public:
  explicit RingMarketUpdateSink(SpscRingBuffer<MarketUpdate, Capacity>& queue) : queue_(queue) {}

  bool publish(const MarketUpdate& update) noexcept override { return queue_.push(update); }

 private:
  SpscRingBuffer<MarketUpdate, Capacity>& queue_;
};

bool decode_pool_update(std::span<const std::uint8_t> account_data,
                        const PoolLayout& layout,
                        std::uint64_t pool_id,
                        std::uint64_t slot,
                        MarketUpdate& out) noexcept;

class YellowstoneIngestionClient {
 public:
  explicit YellowstoneIngestionClient(MarketUpdateSink& sink) : sink_(sink) {}
  bool ingest_account_update(std::span<const std::uint8_t> account_data,
                             const PoolLayout& layout,
                             std::uint64_t pool_id,
                             std::uint64_t slot) noexcept;

 private:
  MarketUpdateSink& sink_;
};

}  // namespace omega::pipeline
