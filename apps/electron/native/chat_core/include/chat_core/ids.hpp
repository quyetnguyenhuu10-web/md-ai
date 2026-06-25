#pragma once

#include <atomic>
#include <chrono>
#include <sstream>
#include <string>

namespace chat_core {

inline std::string make_id(const std::string& prefix) {
  static std::atomic<unsigned long long> seq{0};
  const auto now = std::chrono::duration_cast<std::chrono::milliseconds>(
                       std::chrono::system_clock::now().time_since_epoch())
                       .count();
  std::ostringstream out;
  out << prefix << "_" << now << "_" << ++seq;
  return out.str();
}

inline long long now_ms() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

}  // namespace chat_core
