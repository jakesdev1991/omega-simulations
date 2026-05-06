#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <optional>

namespace omega::pipeline {

template <typename T, std::size_t Capacity>
class SpscRingBuffer {
 public:
  static_assert(Capacity > 1, "SPSC ring buffer capacity must exceed one slot");

  bool push(const T& item) noexcept {
    const auto head = head_.load(std::memory_order_relaxed);
    const auto next = increment(head);
    if (next == tail_.load(std::memory_order_acquire)) {
      return false;
    }
    buffer_[head] = item;
    head_.store(next, std::memory_order_release);
    return true;
  }

  std::optional<T> pop() noexcept {
    const auto tail = tail_.load(std::memory_order_relaxed);
    if (tail == head_.load(std::memory_order_acquire)) {
      return std::nullopt;
    }
    T item = buffer_[tail];
    tail_.store(increment(tail), std::memory_order_release);
    return item;
  }

  bool empty() const noexcept {
    return tail_.load(std::memory_order_acquire) == head_.load(std::memory_order_acquire);
  }

 private:
  static constexpr std::size_t increment(std::size_t value) noexcept {
    return (value + 1) % Capacity;
  }

  alignas(64) std::atomic<std::size_t> head_{0};
  alignas(64) std::atomic<std::size_t> tail_{0};
  std::array<T, Capacity> buffer_{};
};

}  // namespace omega::pipeline
