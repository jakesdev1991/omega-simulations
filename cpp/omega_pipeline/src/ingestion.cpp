#include "omega_pipeline/ingestion.hpp"

#include <cstring>

namespace omega::pipeline {
namespace {

bool read_u64_le(std::span<const std::uint8_t> data, std::size_t offset, std::uint64_t& value) noexcept {
  if (offset + sizeof(std::uint64_t) > data.size()) {
    return false;
  }
  std::memcpy(&value, data.data() + offset, sizeof(std::uint64_t));
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  value = __builtin_bswap64(value);
#endif
  return true;
}

}  // namespace

bool decode_pool_update(std::span<const std::uint8_t> account_data,
                        const PoolLayout& layout,
                        std::uint64_t pool_id,
                        std::uint64_t slot,
                        MarketUpdate& out) noexcept {
  std::uint64_t reserve_in = 0;
  std::uint64_t reserve_out = 0;
  if (!read_u64_le(account_data, layout.reserve_in_offset, reserve_in) ||
      !read_u64_le(account_data, layout.reserve_out_offset, reserve_out)) {
    return false;
  }

  out = MarketUpdate{
      .pool_id = pool_id,
      .slot = slot,
      .reserve_in = static_cast<float>(reserve_in),
      .reserve_out = static_cast<float>(reserve_out),
  };
  return true;
}

bool YellowstoneIngestionClient::ingest_account_update(std::span<const std::uint8_t> account_data,
                                                       const PoolLayout& layout,
                                                       std::uint64_t pool_id,
                                                       std::uint64_t slot) noexcept {
  MarketUpdate update{};
  if (!decode_pool_update(account_data, layout, pool_id, slot, update)) {
    return false;
  }
  return sink_.publish(update);
}

}  // namespace omega::pipeline
