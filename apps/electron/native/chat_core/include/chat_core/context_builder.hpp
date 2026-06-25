#pragma once

#include <chat_core/conversation_index.hpp>
#include <chat_core/storage_sqlite.hpp>
#include <chat_core/token_budget.hpp>
#include <nlohmann/json.hpp>

namespace chat_core {

class ContextBuilder {
 public:
  ContextBuilder(StorageSQLite& storage, ConversationIndex& index, TokenBudgetManager& token_budget);
  nlohmann::json build_context(const std::string& conversation_id, const std::string& current_user_message,
                               int max_tokens);

 private:
  StorageSQLite& storage_;
  ConversationIndex& index_;
  TokenBudgetManager& token_budget_;
};

}  // namespace chat_core
