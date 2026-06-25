#pragma once

#include <chat_core/storage_sqlite.hpp>
#include <nlohmann/json.hpp>

namespace chat_core {

class ConversationIndex {
 public:
  explicit ConversationIndex(StorageSQLite& storage);
  void index_message(const Message& message);
  nlohmann::json search_messages(const std::string& query, int limit);

 private:
  StorageSQLite& storage_;
};

}  // namespace chat_core
