#pragma once

#include <chat_core/conversation.hpp>
#include <chat_core/message.hpp>
#include <filesystem>
#include <mutex>
#include <sqlite3.h>
#include <string>
#include <unordered_map>
#include <vector>

namespace chat_core {

class StorageSQLite {
 public:
  explicit StorageSQLite(std::string storage_root, std::string legacy_db_path = "");
  ~StorageSQLite();

  StorageSQLite(const StorageSQLite&) = delete;
  StorageSQLite& operator=(const StorageSQLite&) = delete;

  Conversation create_conversation(const std::string& title);
  std::vector<Conversation> list_conversations();
  Conversation rename_conversation(const std::string& conversation_id, const std::string& title);
  void delete_conversation(const std::string& conversation_id);
  Message append_message(const std::string& conversation_id, const std::string& role, const std::string& content,
                         const std::string& status);
  Message update_message_content(const std::string& message_id, const std::string& content);
  Message finalize_message(const std::string& message_id);
  std::vector<Message> load_recent_messages(const std::string& conversation_id, int limit);
  std::vector<Message> load_messages_before(const std::string& conversation_id, const std::string& before_message_id,
                                            int limit);
  std::vector<Message> search_messages(const std::string& query, int limit);
  Message get_message(const std::string& message_id);
  std::string get_summary(const std::string& conversation_id);
  int message_count();
  int layout_count();

 private:
  std::filesystem::path storage_root_;
  std::filesystem::path legacy_db_path_;
  std::recursive_mutex mutex_;
  std::unordered_map<std::string, sqlite3*> databases_;
  std::unordered_map<std::string, std::string> message_conversations_;

  sqlite3* database_for_conversation(const std::string& conversation_id, bool create_if_missing = false);
  sqlite3* open_database(const std::filesystem::path& path);
  std::filesystem::path conversation_path(const std::string& conversation_id) const;
  std::vector<std::string> conversation_ids();
  void migrate(sqlite3* db);
  void migrate_legacy_database();
  void migrate_legacy_conversation(sqlite3* legacy_db, const Conversation& conversation);
  void exec(sqlite3* db, const std::string& sql);
  void bind_text(sqlite3* db, sqlite3_stmt* stmt, int index, const std::string& value);
  Message read_message(sqlite3_stmt* stmt);
  Conversation read_conversation(sqlite3_stmt* stmt);
  Conversation read_conversation(sqlite3* db, const std::string& conversation_id);
  Message get_message_unlocked(const std::string& message_id);
  std::string find_conversation_for_message(const std::string& message_id);
  void upsert_layout(sqlite3* db, const std::string& message_id, const LayoutMetadata& layout);
};

}  // namespace chat_core
