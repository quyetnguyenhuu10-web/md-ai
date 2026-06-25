#include <chat_core/token_budget.hpp>

#include <algorithm>

namespace chat_core {

int TokenBudgetManager::estimate_tokens(const std::string& text) const {
  return std::max(1, static_cast<int>((text.size() + 3) / 4));
}

int TokenBudgetManager::estimate_messages(const std::vector<Message>& messages) const {
  int total = 0;
  for (const auto& message : messages) {
    total += message.token_estimate > 0 ? message.token_estimate : estimate_tokens(message.content);
  }
  return total;
}

std::vector<Message> TokenBudgetManager::fit_messages_into_budget(const std::vector<Message>& messages, int max_tokens) const {
  std::vector<Message> selected;
  int used = 0;
  for (auto it = messages.rbegin(); it != messages.rend(); ++it) {
    const int tokens = it->token_estimate > 0 ? it->token_estimate : estimate_tokens(it->content);
    if (used + tokens > max_tokens) {
      break;
    }
    used += tokens;
    selected.push_back(*it);
  }
  std::reverse(selected.begin(), selected.end());
  return selected;
}

nlohmann::json TokenBudgetManager::build_budget(const std::vector<Message>& messages, const std::string& current_input,
                                                int max_tokens) const {
  const int input = estimate_tokens(current_input);
  const int history = estimate_messages(messages);
  const int total = input + history;
  return {{"inputTokens", input},
          {"historyTokens", history},
          {"totalTokens", total},
          {"maxTokens", max_tokens},
          {"remainingTokens", std::max(0, max_tokens - total)}};
}

}  // namespace chat_core
