#include <chat_core/error.hpp>

#include <utility>

namespace chat_core {

CoreError::CoreError(std::string code, std::string message) : std::runtime_error(std::move(message)), code_(std::move(code)) {}

const std::string& CoreError::code() const {
  return code_;
}

}  // namespace chat_core
