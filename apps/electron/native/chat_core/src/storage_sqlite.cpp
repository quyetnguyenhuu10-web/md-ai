#include <chat_core/storage_sqlite.hpp>

#include <chat_core/error.hpp>
#include <chat_core/ids.hpp>
#include <chat_core/layout_cache.hpp>
#include <chat_core/token_budget.hpp>

#include <algorithm>
#include <cctype>
#include <fstream>

namespace chat_core {

namespace {
void check_sqlite(int rc, sqlite3* db, const std::string& message) {
  if (rc != SQLITE_OK && rc != SQLITE_DONE && rc != SQLITE_ROW) {
    throw CoreError("SQLITE_ERROR", message + ": " + sqlite3_errmsg(db));
  }
}

int clamp_limit(int limit) {
  return std::clamp(limit, 1, 200);
}

bool has_valid_id_chars(const std::string& value) {
  if (value.empty()) {
    return false;
  }
  for (const unsigned char ch : value) {
    if (!std::isalnum(ch) && ch != '_' && ch != '-') {
      return false;
    }
  }
  return true;
}

bool ignored_storage_file(const std::filesystem::path& path) {
  const auto name = path.filename().string();
  return name.empty() || name[0] == '.' || name.ends_with("-journal") || name.ends_with("-wal") || name.ends_with("-shm");
}
}  // namespace

StorageSQLite::StorageSQLite(std::string storage_root, std::string legacy_db_path)
    : storage_root_(std::move(storage_root)), legacy_db_path_(std::move(legacy_db_path)) {
  std::filesystem::create_directories(storage_root_);
  migrate_legacy_database();
}

StorageSQLite::~StorageSQLite() {
  for (auto& item : databases_) {
    sqlite3_close(item.second);
  }
}

sqlite3* StorageSQLite::open_database(const std::filesystem::path& path) {
  sqlite3* db = nullptr;
  const int rc = sqlite3_open(path.string().c_str(), &db);
  if (rc != SQLITE_OK) {
    const std::string message = db != nullptr ? sqlite3_errmsg(db) : "unknown sqlite open error";
    if (db != nullptr) {
      sqlite3_close(db);
    }
    throw CoreError("SQLITE_OPEN_FAILED", message);
  }

  exec(db, "PRAGMA busy_timeout=5000;");
  exec(db, "PRAGMA journal_mode=DELETE;");
  exec(db, "PRAGMA synchronous=NORMAL;");
  migrate(db);
  return db;
}

std::filesystem::path StorageSQLite::conversation_path(const std::string& conversation_id) const {
  if (!has_valid_id_chars(conversation_id)) {
    throw CoreError("BAD_REQUEST", "Invalid conversationId");
  }
  return storage_root_ / conversation_id;
}

sqlite3* StorageSQLite::database_for_conversation(const std::string& conversation_id, bool create_if_missing) {
  const auto found = databases_.find(conversation_id);
  if (found != databases_.end()) {
    return found->second;
  }

  const auto path = conversation_path(conversation_id);
  if (!create_if_missing && !std::filesystem::exists(path)) {
    throw CoreError("NOT_FOUND", "Conversation not found");
  }

  sqlite3* db = open_database(path);
  databases_.emplace(conversation_id, db);
  return db;
}

std::vector<std::string> StorageSQLite::conversation_ids() {
  std::vector<std::string> ids;
  if (!std::filesystem::exists(storage_root_)) {
    return ids;
  }

  for (const auto& entry : std::filesystem::directory_iterator(storage_root_)) {
    if (!entry.is_regular_file() || ignored_storage_file(entry.path())) {
      continue;
    }
    const auto id = entry.path().filename().string();
    if (has_valid_id_chars(id)) {
      ids.push_back(id);
    }
  }
  return ids;
}

void StorageSQLite::exec(sqlite3* db, const std::string& sql) {
  char* error = nullptr;
  const int rc = sqlite3_exec(db, sql.c_str(), nullptr, nullptr, &error);
  if (rc != SQLITE_OK) {
    std::string message = error != nullptr ? error : "unknown sqlite error";
    sqlite3_free(error);
    throw CoreError("SQLITE_ERROR", message);
  }
}

void StorageSQLite::migrate(sqlite3* db) {
  exec(db, "CREATE TABLE IF NOT EXISTS conversations ("
           "id TEXT PRIMARY KEY,"
           "title TEXT NOT NULL,"
           "created_at INTEGER NOT NULL,"
           "updated_at INTEGER NOT NULL"
           ");");
  exec(db, "CREATE TABLE IF NOT EXISTS messages ("
           "id TEXT PRIMARY KEY,"
           "conversation_id TEXT NOT NULL,"
           "role TEXT NOT NULL,"
           "content TEXT NOT NULL,"
           "status TEXT NOT NULL,"
           "created_at INTEGER NOT NULL,"
           "updated_at INTEGER NOT NULL,"
           "token_estimate INTEGER NOT NULL DEFAULT 0"
           ");");
  exec(db, "CREATE INDEX IF NOT EXISTS idx_messages_conversation_created ON messages(conversation_id, created_at);");
  exec(db, "CREATE TABLE IF NOT EXISTS message_layout_cache ("
           "message_id TEXT PRIMARY KEY,"
           "estimated_height INTEGER NOT NULL,"
           "has_code_block INTEGER NOT NULL,"
           "has_markdown INTEGER NOT NULL,"
           "has_math INTEGER NOT NULL,"
           "has_image INTEGER NOT NULL,"
           "is_long_message INTEGER NOT NULL,"
           "updated_at INTEGER NOT NULL"
           ");");
  exec(db, "CREATE TABLE IF NOT EXISTS conversation_summaries ("
           "conversation_id TEXT PRIMARY KEY,"
           "summary TEXT NOT NULL,"
           "updated_at INTEGER NOT NULL"
           ");");
}

void StorageSQLite::bind_text(sqlite3* db, sqlite3_stmt* stmt, int index, const std::string& value) {
  check_sqlite(sqlite3_bind_text(stmt, index, value.c_str(), -1, SQLITE_TRANSIENT), db, "bind text");
}

Conversation StorageSQLite::create_conversation(const std::string& title) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  std::string id;
  do {
    id = make_id("conv");
  } while (std::filesystem::exists(conversation_path(id)));

  const auto timestamp = now_ms();
  const auto resolved_title = title.empty() ? "New Chat" : title;
  sqlite3* db = database_for_conversation(id, true);

  sqlite3_stmt* stmt = nullptr;
  check_sqlite(sqlite3_prepare_v2(db, "INSERT INTO conversations(id,title,created_at,updated_at) VALUES(?,?,?,?)", -1, &stmt,
                                  nullptr),
               db, "prepare create conversation");
  bind_text(db, stmt, 1, id);
  bind_text(db, stmt, 2, resolved_title);
  sqlite3_bind_int64(stmt, 3, timestamp);
  sqlite3_bind_int64(stmt, 4, timestamp);
  check_sqlite(sqlite3_step(stmt), db, "insert conversation");
  sqlite3_finalize(stmt);

  return {id, resolved_title, timestamp, timestamp};
}

std::vector<Conversation> StorageSQLite::list_conversations() {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  std::vector<Conversation> conversations;
  for (const auto& id : conversation_ids()) {
    try {
      conversations.push_back(read_conversation(database_for_conversation(id), id));
    } catch (const CoreError&) {
      continue;
    }
  }
  std::sort(conversations.begin(), conversations.end(), [](const Conversation& left, const Conversation& right) {
    return left.updated_at > right.updated_at;
  });
  return conversations;
}

Conversation StorageSQLite::rename_conversation(const std::string& conversation_id, const std::string& title) {
  if (conversation_id.empty() || title.empty()) {
    throw CoreError("BAD_REQUEST", "Missing conversationId or title");
  }

  std::lock_guard<std::recursive_mutex> lock(mutex_);
  sqlite3* db = database_for_conversation(conversation_id);
  const auto timestamp = now_ms();

  sqlite3_stmt* stmt = nullptr;
  check_sqlite(sqlite3_prepare_v2(db, "UPDATE conversations SET title=?, updated_at=? WHERE id=?", -1, &stmt, nullptr), db,
               "prepare rename conversation");
  bind_text(db, stmt, 1, title);
  sqlite3_bind_int64(stmt, 2, timestamp);
  bind_text(db, stmt, 3, conversation_id);
  check_sqlite(sqlite3_step(stmt), db, "rename conversation");
  const int changed = sqlite3_changes(db);
  sqlite3_finalize(stmt);

  if (changed == 0) {
    throw CoreError("NOT_FOUND", "Conversation not found");
  }
  return read_conversation(db, conversation_id);
}

void StorageSQLite::delete_conversation(const std::string& conversation_id) {
  if (conversation_id.empty()) {
    throw CoreError("BAD_REQUEST", "Missing conversationId");
  }

  std::lock_guard<std::recursive_mutex> lock(mutex_);
  const auto path = conversation_path(conversation_id);
  if (!std::filesystem::exists(path)) {
    throw CoreError("NOT_FOUND", "Conversation not found");
  }

  const auto found = databases_.find(conversation_id);
  if (found != databases_.end()) {
    sqlite3_close(found->second);
    databases_.erase(found);
  }

  for (auto it = message_conversations_.begin(); it != message_conversations_.end();) {
    if (it->second == conversation_id) {
      it = message_conversations_.erase(it);
    } else {
      ++it;
    }
  }

  std::filesystem::remove(path);
  std::filesystem::remove(path.string() + "-journal");
}

Message StorageSQLite::append_message(const std::string& conversation_id, const std::string& role, const std::string& content,
                                      const std::string& status) {
  if (conversation_id.empty() || role.empty()) {
    throw CoreError("BAD_REQUEST", "Missing conversationId or role");
  }

  std::lock_guard<std::recursive_mutex> lock(mutex_);
  sqlite3* db = database_for_conversation(conversation_id);
  (void)read_conversation(db, conversation_id);

  const auto id = make_id("msg");
  const auto timestamp = now_ms();
  TokenBudgetManager token_budget;
  LayoutCache layout_cache;
  const auto layout = layout_cache.estimate_message_layout(content);
  const auto tokens = token_budget.estimate_tokens(content);
  const auto resolved_status = status.empty() ? "complete" : status;

  exec(db, "BEGIN IMMEDIATE TRANSACTION;");
  sqlite3_stmt* stmt = nullptr;
  check_sqlite(sqlite3_prepare_v2(db,
                                  "INSERT INTO messages(id,conversation_id,role,content,status,created_at,updated_at,token_estimate) "
                                  "VALUES(?,?,?,?,?,?,?,?)",
                                  -1, &stmt, nullptr),
               db, "prepare append message");
  bind_text(db, stmt, 1, id);
  bind_text(db, stmt, 2, conversation_id);
  bind_text(db, stmt, 3, role);
  bind_text(db, stmt, 4, content);
  bind_text(db, stmt, 5, resolved_status);
  sqlite3_bind_int64(stmt, 6, timestamp);
  sqlite3_bind_int64(stmt, 7, timestamp);
  sqlite3_bind_int(stmt, 8, tokens);
  check_sqlite(sqlite3_step(stmt), db, "insert message");
  sqlite3_finalize(stmt);

  sqlite3_stmt* update = nullptr;
  check_sqlite(sqlite3_prepare_v2(db, "UPDATE conversations SET updated_at=?, title=CASE WHEN title='New Chat' THEN ? ELSE title END WHERE id=?",
                                  -1, &update, nullptr),
               db, "prepare conversation touch");
  sqlite3_bind_int64(update, 1, timestamp);
  bind_text(db, update, 2, content.substr(0, 48).empty() ? "New Chat" : content.substr(0, 48));
  bind_text(db, update, 3, conversation_id);
  check_sqlite(sqlite3_step(update), db, "touch conversation");
  sqlite3_finalize(update);

  upsert_layout(db, id, layout);
  exec(db, "COMMIT;");

  message_conversations_[id] = conversation_id;
  return {id, conversation_id, role, content, resolved_status, timestamp, timestamp, tokens, layout};
}

Message StorageSQLite::update_message_content(const std::string& message_id, const std::string& content) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  const auto conversation_id = find_conversation_for_message(message_id);
  sqlite3* db = database_for_conversation(conversation_id);
  TokenBudgetManager token_budget;
  LayoutCache layout_cache;
  const auto layout = layout_cache.estimate_message_layout(content);
  const auto tokens = token_budget.estimate_tokens(content);
  const auto timestamp = now_ms();

  sqlite3_stmt* stmt = nullptr;
  check_sqlite(sqlite3_prepare_v2(db, "UPDATE messages SET content=?, updated_at=?, token_estimate=? WHERE id=?", -1, &stmt, nullptr),
               db, "prepare update message");
  bind_text(db, stmt, 1, content);
  sqlite3_bind_int64(stmt, 2, timestamp);
  sqlite3_bind_int(stmt, 3, tokens);
  bind_text(db, stmt, 4, message_id);
  check_sqlite(sqlite3_step(stmt), db, "update message");
  const int changed = sqlite3_changes(db);
  sqlite3_finalize(stmt);

  if (changed == 0) {
    throw CoreError("NOT_FOUND", "Message not found");
  }
  upsert_layout(db, message_id, layout);
  return get_message_unlocked(message_id);
}

Message StorageSQLite::finalize_message(const std::string& message_id) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  const auto conversation_id = find_conversation_for_message(message_id);
  sqlite3* db = database_for_conversation(conversation_id);
  const auto timestamp = now_ms();

  sqlite3_stmt* stmt = nullptr;
  check_sqlite(sqlite3_prepare_v2(db, "UPDATE messages SET status='complete', updated_at=? WHERE id=?", -1, &stmt, nullptr), db,
               "prepare finalize message");
  sqlite3_bind_int64(stmt, 1, timestamp);
  bind_text(db, stmt, 2, message_id);
  check_sqlite(sqlite3_step(stmt), db, "finalize message");
  const int changed = sqlite3_changes(db);
  sqlite3_finalize(stmt);

  if (changed == 0) {
    throw CoreError("NOT_FOUND", "Message not found");
  }
  return get_message_unlocked(message_id);
}

std::vector<Message> StorageSQLite::load_recent_messages(const std::string& conversation_id, int limit) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  sqlite3* db = database_for_conversation(conversation_id);
  sqlite3_stmt* stmt = nullptr;
  check_sqlite(sqlite3_prepare_v2(db,
                                  "SELECT m.id,m.conversation_id,m.role,m.content,m.status,m.created_at,m.updated_at,m.token_estimate,"
                                  "l.estimated_height,l.has_code_block,l.has_markdown,l.has_math,l.has_image,l.is_long_message "
                                  "FROM messages m LEFT JOIN message_layout_cache l ON l.message_id=m.id "
                                  "WHERE m.conversation_id=? ORDER BY m.created_at DESC LIMIT ?",
                                  -1, &stmt, nullptr),
               db, "prepare recent messages");
  bind_text(db, stmt, 1, conversation_id);
  sqlite3_bind_int(stmt, 2, clamp_limit(limit));

  std::vector<Message> messages;
  while (sqlite3_step(stmt) == SQLITE_ROW) {
    auto message = read_message(stmt);
    message_conversations_[message.id] = message.conversation_id;
    messages.push_back(message);
  }
  sqlite3_finalize(stmt);
  std::reverse(messages.begin(), messages.end());
  return messages;
}

std::vector<Message> StorageSQLite::load_messages_before(const std::string& conversation_id, const std::string& before_message_id,
                                                         int limit) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  const auto before = get_message_unlocked(before_message_id);
  sqlite3* db = database_for_conversation(conversation_id);

  sqlite3_stmt* stmt = nullptr;
  check_sqlite(sqlite3_prepare_v2(db,
                                  "SELECT m.id,m.conversation_id,m.role,m.content,m.status,m.created_at,m.updated_at,m.token_estimate,"
                                  "l.estimated_height,l.has_code_block,l.has_markdown,l.has_math,l.has_image,l.is_long_message "
                                  "FROM messages m LEFT JOIN message_layout_cache l ON l.message_id=m.id "
                                  "WHERE m.conversation_id=? AND m.created_at<? ORDER BY m.created_at DESC LIMIT ?",
                                  -1, &stmt, nullptr),
               db, "prepare before messages");
  bind_text(db, stmt, 1, conversation_id);
  sqlite3_bind_int64(stmt, 2, before.created_at);
  sqlite3_bind_int(stmt, 3, clamp_limit(limit));

  std::vector<Message> messages;
  while (sqlite3_step(stmt) == SQLITE_ROW) {
    auto message = read_message(stmt);
    message_conversations_[message.id] = message.conversation_id;
    messages.push_back(message);
  }
  sqlite3_finalize(stmt);
  std::reverse(messages.begin(), messages.end());
  return messages;
}

std::vector<Message> StorageSQLite::search_messages(const std::string& query, int limit) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  std::vector<Message> messages;
  for (const auto& id : conversation_ids()) {
    sqlite3* db = database_for_conversation(id);
    sqlite3_stmt* stmt = nullptr;
    check_sqlite(sqlite3_prepare_v2(db,
                                    "SELECT m.id,m.conversation_id,m.role,m.content,m.status,m.created_at,m.updated_at,m.token_estimate,"
                                    "l.estimated_height,l.has_code_block,l.has_markdown,l.has_math,l.has_image,l.is_long_message "
                                    "FROM messages m LEFT JOIN message_layout_cache l ON l.message_id=m.id "
                                    "WHERE m.content LIKE ? ORDER BY m.updated_at DESC LIMIT ?",
                                    -1, &stmt, nullptr),
                 db, "prepare search messages");
    bind_text(db, stmt, 1, "%" + query + "%");
    sqlite3_bind_int(stmt, 2, clamp_limit(limit));
    while (sqlite3_step(stmt) == SQLITE_ROW) {
      auto message = read_message(stmt);
      message_conversations_[message.id] = message.conversation_id;
      messages.push_back(message);
    }
    sqlite3_finalize(stmt);
  }

  std::sort(messages.begin(), messages.end(), [](const Message& left, const Message& right) {
    return left.updated_at > right.updated_at;
  });
  if (messages.size() > static_cast<std::size_t>(clamp_limit(limit))) {
    messages.resize(static_cast<std::size_t>(clamp_limit(limit)));
  }
  return messages;
}

Message StorageSQLite::get_message(const std::string& message_id) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  return get_message_unlocked(message_id);
}

Message StorageSQLite::get_message_unlocked(const std::string& message_id) {
  const auto conversation_id = find_conversation_for_message(message_id);
  sqlite3* db = database_for_conversation(conversation_id);
  sqlite3_stmt* stmt = nullptr;
  check_sqlite(sqlite3_prepare_v2(db,
                                  "SELECT m.id,m.conversation_id,m.role,m.content,m.status,m.created_at,m.updated_at,m.token_estimate,"
                                  "l.estimated_height,l.has_code_block,l.has_markdown,l.has_math,l.has_image,l.is_long_message "
                                  "FROM messages m LEFT JOIN message_layout_cache l ON l.message_id=m.id WHERE m.id=?",
                                  -1, &stmt, nullptr),
               db, "prepare get message");
  bind_text(db, stmt, 1, message_id);
  if (sqlite3_step(stmt) != SQLITE_ROW) {
    sqlite3_finalize(stmt);
    throw CoreError("NOT_FOUND", "Message not found");
  }
  auto message = read_message(stmt);
  sqlite3_finalize(stmt);
  message_conversations_[message.id] = message.conversation_id;
  return message;
}

std::string StorageSQLite::find_conversation_for_message(const std::string& message_id) {
  const auto cached = message_conversations_.find(message_id);
  if (cached != message_conversations_.end()) {
    return cached->second;
  }

  for (const auto& id : conversation_ids()) {
    sqlite3* db = database_for_conversation(id);
    sqlite3_stmt* stmt = nullptr;
    check_sqlite(sqlite3_prepare_v2(db, "SELECT conversation_id FROM messages WHERE id=?", -1, &stmt, nullptr), db,
                 "prepare find message conversation");
    bind_text(db, stmt, 1, message_id);
    if (sqlite3_step(stmt) == SQLITE_ROW) {
      const auto conversation_id = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
      sqlite3_finalize(stmt);
      message_conversations_[message_id] = conversation_id;
      return conversation_id;
    }
    sqlite3_finalize(stmt);
  }

  throw CoreError("NOT_FOUND", "Message not found");
}

std::string StorageSQLite::get_summary(const std::string& conversation_id) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  sqlite3* db = database_for_conversation(conversation_id);
  sqlite3_stmt* stmt = nullptr;
  check_sqlite(sqlite3_prepare_v2(db, "SELECT summary FROM conversation_summaries WHERE conversation_id=?", -1, &stmt, nullptr), db,
               "prepare summary");
  bind_text(db, stmt, 1, conversation_id);
  std::string summary;
  if (sqlite3_step(stmt) == SQLITE_ROW && sqlite3_column_text(stmt, 0) != nullptr) {
    summary = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
  }
  sqlite3_finalize(stmt);
  return summary;
}

int StorageSQLite::message_count() {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  int count = 0;
  for (const auto& id : conversation_ids()) {
    sqlite3* db = database_for_conversation(id);
    sqlite3_stmt* stmt = nullptr;
    check_sqlite(sqlite3_prepare_v2(db, "SELECT COUNT(*) FROM messages", -1, &stmt, nullptr), db, "prepare count messages");
    sqlite3_step(stmt);
    count += sqlite3_column_int(stmt, 0);
    sqlite3_finalize(stmt);
  }
  return count;
}

int StorageSQLite::layout_count() {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  int count = 0;
  for (const auto& id : conversation_ids()) {
    sqlite3* db = database_for_conversation(id);
    sqlite3_stmt* stmt = nullptr;
    check_sqlite(sqlite3_prepare_v2(db, "SELECT COUNT(*) FROM message_layout_cache", -1, &stmt, nullptr), db,
                 "prepare count layouts");
    sqlite3_step(stmt);
    count += sqlite3_column_int(stmt, 0);
    sqlite3_finalize(stmt);
  }
  return count;
}

Message StorageSQLite::read_message(sqlite3_stmt* stmt) {
  Message message;
  message.id = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
  message.conversation_id = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
  message.role = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
  message.content = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3));
  message.status = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 4));
  message.created_at = sqlite3_column_int64(stmt, 5);
  message.updated_at = sqlite3_column_int64(stmt, 6);
  message.token_estimate = sqlite3_column_int(stmt, 7);
  message.layout.estimated_height = sqlite3_column_type(stmt, 8) == SQLITE_NULL ? 96 : sqlite3_column_int(stmt, 8);
  message.layout.has_code_block = sqlite3_column_int(stmt, 9) != 0;
  message.layout.has_markdown = sqlite3_column_int(stmt, 10) != 0;
  message.layout.has_math = sqlite3_column_int(stmt, 11) != 0;
  message.layout.has_image = sqlite3_column_int(stmt, 12) != 0;
  message.layout.is_long_message = sqlite3_column_int(stmt, 13) != 0;
  return message;
}

Conversation StorageSQLite::read_conversation(sqlite3_stmt* stmt) {
  Conversation conversation;
  conversation.id = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
  conversation.title = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
  conversation.created_at = sqlite3_column_int64(stmt, 2);
  conversation.updated_at = sqlite3_column_int64(stmt, 3);
  return conversation;
}

Conversation StorageSQLite::read_conversation(sqlite3* db, const std::string& conversation_id) {
  sqlite3_stmt* stmt = nullptr;
  check_sqlite(sqlite3_prepare_v2(db, "SELECT id,title,created_at,updated_at FROM conversations WHERE id=?", -1, &stmt, nullptr),
               db, "prepare get conversation");
  bind_text(db, stmt, 1, conversation_id);
  if (sqlite3_step(stmt) != SQLITE_ROW) {
    sqlite3_finalize(stmt);
    throw CoreError("NOT_FOUND", "Conversation not found");
  }
  auto conversation = read_conversation(stmt);
  sqlite3_finalize(stmt);
  return conversation;
}

void StorageSQLite::upsert_layout(sqlite3* db, const std::string& message_id, const LayoutMetadata& layout) {
  sqlite3_stmt* stmt = nullptr;
  check_sqlite(sqlite3_prepare_v2(db,
                                  "INSERT INTO message_layout_cache(message_id,estimated_height,has_code_block,has_markdown,has_math,"
                                  "has_image,is_long_message,updated_at) VALUES(?,?,?,?,?,?,?,?) "
                                  "ON CONFLICT(message_id) DO UPDATE SET estimated_height=excluded.estimated_height,"
                                  "has_code_block=excluded.has_code_block,has_markdown=excluded.has_markdown,has_math=excluded.has_math,"
                                  "has_image=excluded.has_image,is_long_message=excluded.is_long_message,updated_at=excluded.updated_at",
                                  -1, &stmt, nullptr),
               db, "prepare layout upsert");
  bind_text(db, stmt, 1, message_id);
  sqlite3_bind_int(stmt, 2, layout.estimated_height);
  sqlite3_bind_int(stmt, 3, layout.has_code_block ? 1 : 0);
  sqlite3_bind_int(stmt, 4, layout.has_markdown ? 1 : 0);
  sqlite3_bind_int(stmt, 5, layout.has_math ? 1 : 0);
  sqlite3_bind_int(stmt, 6, layout.has_image ? 1 : 0);
  sqlite3_bind_int(stmt, 7, layout.is_long_message ? 1 : 0);
  sqlite3_bind_int64(stmt, 8, now_ms());
  check_sqlite(sqlite3_step(stmt), db, "upsert layout");
  sqlite3_finalize(stmt);
}

void StorageSQLite::migrate_legacy_database() {
  if (legacy_db_path_.empty() || !std::filesystem::exists(legacy_db_path_)) {
    return;
  }

  const auto marker_path = std::filesystem::path(legacy_db_path_.string() + ".conversation-files-migrated");
  if (std::filesystem::exists(marker_path)) {
    return;
  }

  sqlite3* legacy_db = nullptr;
  if (sqlite3_open_v2(legacy_db_path_.string().c_str(), &legacy_db, SQLITE_OPEN_READONLY, nullptr) != SQLITE_OK) {
    if (legacy_db != nullptr) {
      sqlite3_close(legacy_db);
    }
    return;
  }

  sqlite3_stmt* stmt = nullptr;
  if (sqlite3_prepare_v2(legacy_db, "SELECT id,title,created_at,updated_at FROM conversations ORDER BY updated_at DESC", -1, &stmt,
                        nullptr) != SQLITE_OK) {
    sqlite3_close(legacy_db);
    return;
  }

  std::vector<Conversation> conversations;
  while (sqlite3_step(stmt) == SQLITE_ROW) {
    conversations.push_back(read_conversation(stmt));
  }
  sqlite3_finalize(stmt);

  for (const auto& conversation : conversations) {
    migrate_legacy_conversation(legacy_db, conversation);
  }

  sqlite3_close(legacy_db);
  std::ofstream marker(marker_path);
  marker << "migrated";
}

void StorageSQLite::migrate_legacy_conversation(sqlite3* legacy_db, const Conversation& conversation) {
  if (!has_valid_id_chars(conversation.id) || std::filesystem::exists(conversation_path(conversation.id))) {
    return;
  }

  sqlite3* db = database_for_conversation(conversation.id, true);
  exec(db, "BEGIN IMMEDIATE TRANSACTION;");

  sqlite3_stmt* conv_stmt = nullptr;
  check_sqlite(sqlite3_prepare_v2(db, "INSERT INTO conversations(id,title,created_at,updated_at) VALUES(?,?,?,?)", -1, &conv_stmt,
                                  nullptr),
               db, "prepare migrate conversation");
  bind_text(db, conv_stmt, 1, conversation.id);
  bind_text(db, conv_stmt, 2, conversation.title);
  sqlite3_bind_int64(conv_stmt, 3, conversation.created_at);
  sqlite3_bind_int64(conv_stmt, 4, conversation.updated_at);
  check_sqlite(sqlite3_step(conv_stmt), db, "migrate conversation");
  sqlite3_finalize(conv_stmt);

  sqlite3_stmt* message_stmt = nullptr;
  check_sqlite(sqlite3_prepare_v2(legacy_db,
                                  "SELECT m.id,m.conversation_id,m.role,m.content,m.status,m.created_at,m.updated_at,m.token_estimate,"
                                  "l.estimated_height,l.has_code_block,l.has_markdown,l.has_math,l.has_image,l.is_long_message "
                                  "FROM messages m LEFT JOIN message_layout_cache l ON l.message_id=m.id "
                                  "WHERE m.conversation_id=? ORDER BY m.created_at ASC",
                                  -1, &message_stmt, nullptr),
               legacy_db, "prepare migrate messages");
  bind_text(legacy_db, message_stmt, 1, conversation.id);
  while (sqlite3_step(message_stmt) == SQLITE_ROW) {
    const auto message = read_message(message_stmt);
    sqlite3_stmt* insert = nullptr;
    check_sqlite(sqlite3_prepare_v2(db,
                                    "INSERT INTO messages(id,conversation_id,role,content,status,created_at,updated_at,token_estimate) "
                                    "VALUES(?,?,?,?,?,?,?,?)",
                                    -1, &insert, nullptr),
                 db, "prepare migrate message");
    bind_text(db, insert, 1, message.id);
    bind_text(db, insert, 2, message.conversation_id);
    bind_text(db, insert, 3, message.role);
    bind_text(db, insert, 4, message.content);
    bind_text(db, insert, 5, message.status);
    sqlite3_bind_int64(insert, 6, message.created_at);
    sqlite3_bind_int64(insert, 7, message.updated_at);
    sqlite3_bind_int(insert, 8, message.token_estimate);
    check_sqlite(sqlite3_step(insert), db, "migrate message");
    sqlite3_finalize(insert);
    upsert_layout(db, message.id, message.layout);
    message_conversations_[message.id] = conversation.id;
  }
  sqlite3_finalize(message_stmt);

  sqlite3_stmt* summary_stmt = nullptr;
  check_sqlite(sqlite3_prepare_v2(legacy_db, "SELECT summary,updated_at FROM conversation_summaries WHERE conversation_id=?", -1,
                                  &summary_stmt, nullptr),
               legacy_db, "prepare migrate summary");
  bind_text(legacy_db, summary_stmt, 1, conversation.id);
  if (sqlite3_step(summary_stmt) == SQLITE_ROW) {
    sqlite3_stmt* insert_summary = nullptr;
    check_sqlite(sqlite3_prepare_v2(db,
                                    "INSERT INTO conversation_summaries(conversation_id,summary,updated_at) VALUES(?,?,?) "
                                    "ON CONFLICT(conversation_id) DO UPDATE SET summary=excluded.summary,updated_at=excluded.updated_at",
                                    -1, &insert_summary, nullptr),
                 db, "prepare insert summary");
    bind_text(db, insert_summary, 1, conversation.id);
    bind_text(db, insert_summary, 2, reinterpret_cast<const char*>(sqlite3_column_text(summary_stmt, 0)));
    sqlite3_bind_int64(insert_summary, 3, sqlite3_column_int64(summary_stmt, 1));
    check_sqlite(sqlite3_step(insert_summary), db, "insert summary");
    sqlite3_finalize(insert_summary);
  }
  sqlite3_finalize(summary_stmt);

  exec(db, "COMMIT;");
}

nlohmann::json to_json(const Conversation& conversation) {
  return {{"id", conversation.id},
          {"title", conversation.title},
          {"createdAt", conversation.created_at},
          {"updatedAt", conversation.updated_at}};
}

}  // namespace chat_core
