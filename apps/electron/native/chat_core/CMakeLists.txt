add_library(chat_core STATIC
  src/message_store.cpp
  src/conversation_index.cpp
  src/context_builder.cpp
  src/token_budget.cpp
  src/layout_cache.cpp
  src/storage_sqlite.cpp
  src/json_rpc.cpp
  src/error.cpp
)

target_include_directories(chat_core PUBLIC include)
target_link_libraries(chat_core PUBLIC sqlite3 nlohmann_json::nlohmann_json)
