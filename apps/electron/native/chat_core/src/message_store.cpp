#include <chat_core/message_store.hpp>

namespace chat_core {

MessageStore::MessageStore(StorageSQLite& storage) : storage_(storage) {}

Message MessageStore::append_message(const std::string& conversation_id, const std::string& role, const std::string& content,
                                     const std::string& status) {
  return storage_.append_message(conversation_id, role, content, status);
}

Message MessageStore::update_message_content(const std::string& message_id, const std::string& content) {
  return storage_.update_message_content(message_id, content);
}

Message MessageStore::finalize_message(const std::string& message_id) {
  return storage_.finalize_message(message_id);
}

std::vector<Message> MessageStore::load_recent_messages(const std::string& conversation_id, int limit) {
  return storage_.load_recent_messages(conversation_id, limit);
}

std::vector<Message> MessageStore::load_messages_before(const std::string& conversation_id, const std::string& before_message_id,
                                                        int limit) {
  return storage_.load_messages_before(conversation_id, before_message_id, limit);
}

Message MessageStore::get_message(const std::string& message_id) {
  return storage_.get_message(message_id);
}

}  // namespace chat_core
