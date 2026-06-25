#pragma once

#include <stdexcept>
#include <string>

namespace chat_core {

class CoreError : public std::runtime_error {
 public:
  CoreError(std::string code, std::string message);
  const std::string& code() const;

 private:
  std::string code_;
};

}  // namespace chat_core
