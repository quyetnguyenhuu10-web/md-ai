#pragma once

#include <chat_core/message.hpp>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace chat_core {

class TokenBudgetManager {
 public:
  int estimate_tokens(const std::string& text) const;
  int estimate_messages(const std::vector<Message>& messages) const;
  std::vector<Message> fit_messages_into_budget(const std::vector<Message>& messages, int max_tokens) const;
  nlohmann::json build_budget(const std::vector<Message>& messages, const std::string& current_input, int max_tokens) const;
};

}  // namespace chat_core
