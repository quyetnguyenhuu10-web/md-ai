#include <chat_core/context_builder.hpp>

namespace chat_core {

ContextBuilder::ContextBuilder(StorageSQLite& storage, ConversationIndex& index, TokenBudgetManager& token_budget)
    : storage_(storage), index_(index), token_budget_(token_budget) {}

nlohmann::json ContextBuilder::build_context(const std::string& conversation_id, const std::string& current_user_message,
                                             int max_tokens) {
  const auto recent = token_budget_.fit_messages_into_budget(storage_.load_recent_messages(conversation_id, 24), max_tokens / 2);
  auto relevant_result = index_.search_messages(current_user_message, 5);

  nlohmann::json recent_json = nlohmann::json::array();
  for (const auto& message : recent) {
    recent_json.push_back(to_json(message));
  }

  nlohmann::json relevant_messages = nlohmann::json::array();
  for (const auto& item : relevant_result) {
    relevant_messages.push_back(item["message"]);
  }

  const int tokens = token_budget_.estimate_messages(recent) + token_budget_.estimate_tokens(current_user_message) + 32;
  return {{"systemPrompt", "You are a helpful assistant inside HighPerf Chat UI."},
          {"summary", storage_.get_summary(conversation_id)},
          {"memoryFacts", nlohmann::json::array()},
          {"recentMessages", recent_json},
          {"relevantMessages", relevant_messages},
          {"currentUserMessage", current_user_message},
          {"tokenEstimate", tokens},
          {"maxTokens", max_tokens}};
}

}  // namespace chat_core
