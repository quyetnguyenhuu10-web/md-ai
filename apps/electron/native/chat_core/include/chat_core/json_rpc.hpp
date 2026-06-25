#pragma once

#include <chat_core/context_builder.hpp>
#include <chat_core/conversation_index.hpp>
#include <chat_core/message_store.hpp>
#include <chat_core/token_budget.hpp>
#include <nlohmann/json.hpp>

namespace chat_core {

class JsonRpcServer {
 public:
  explicit JsonRpcServer(StorageSQLite& storage);
  nlohmann::json handle(const nlohmann::json& request);

 private:
  StorageSQLite& storage_;
  MessageStore messages_;
  ConversationIndex index_;
  TokenBudgetManager token_budget_;
  ContextBuilder context_;

  nlohmann::json dispatch(const std::string& method, const nlohmann::json& params);
};

}  // namespace chat_core
