#pragma once

#include <chat_core/storage_sqlite.hpp>

namespace chat_core {

class MessageStore {
 public:
  explicit MessageStore(StorageSQLite& storage);
  Message append_message(const std::string& conversation_id, const std::string& role, const std::string& content,
                         const std::string& status);
  Message update_message_content(const std::string& message_id, const std::string& content);
  Message finalize_message(const std::string& message_id);
  std::vector<Message> load_recent_messages(const std::string& conversation_id, int limit);
  std::vector<Message> load_messages_before(const std::string& conversation_id, const std::string& before_message_id,
                                            int limit);
  Message get_message(const std::string& message_id);

 private:
  StorageSQLite& storage_;
};

}  // namespace chat_core
