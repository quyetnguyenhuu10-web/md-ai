#include <chat_core/conversation_index.hpp>

namespace chat_core {

ConversationIndex::ConversationIndex(StorageSQLite& storage) : storage_(storage) {}

void ConversationIndex::index_message(const Message& message) {
  (void)message;
}

nlohmann::json ConversationIndex::search_messages(const std::string& query, int limit) {
  nlohmann::json results = nlohmann::json::array();
  int rank = 1;
  for (const auto& message : storage_.search_messages(query, limit)) {
    results.push_back({{"message", to_json(message)}, {"rank", rank++}});
  }
  return results;
}

}  // namespace chat_core
