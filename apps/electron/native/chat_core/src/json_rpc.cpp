#include <chat_core/json_rpc.hpp>

#include <chat_core/error.hpp>
#include <chat_core/layout_cache.hpp>

namespace chat_core {

namespace {
std::string require_string(const nlohmann::json& params, const char* key) {
  if (!params.contains(key) || !params.at(key).is_string()) {
    throw CoreError("BAD_REQUEST", std::string("Missing string param: ") + key);
  }
  return params.at(key).get<std::string>();
}

int optional_int(const nlohmann::json& params, const char* key, int fallback) {
  if (!params.contains(key) || !params.at(key).is_number_integer()) {
    return fallback;
  }
  return params.at(key).get<int>();
}

std::string optional_string(const nlohmann::json& params, const char* key, const std::string& fallback = "") {
  if (!params.contains(key) || !params.at(key).is_string()) {
    return fallback;
  }
  return params.at(key).get<std::string>();
}
}  // namespace

JsonRpcServer::JsonRpcServer(StorageSQLite& storage)
    : storage_(storage), messages_(storage), index_(storage), context_(storage, index_, token_budget_) {}

nlohmann::json JsonRpcServer::handle(const nlohmann::json& request) {
  const auto id = request.value("id", "");
  try {
    if (!request.contains("method") || !request.at("method").is_string()) {
      throw CoreError("BAD_REQUEST", "Missing method");
    }
    const auto method = request.at("method").get<std::string>();
    const auto params = request.value("params", nlohmann::json::object());
    return {{"id", id}, {"result", dispatch(method, params)}};
  } catch (const CoreError& error) {
    return {{"id", id}, {"error", {{"code", error.code()}, {"message", error.what()}}}};
  } catch (const std::exception& error) {
    return {{"id", id}, {"error", {{"code", "INTERNAL_ERROR"}, {"message", error.what()}}}};
  }
}

nlohmann::json JsonRpcServer::dispatch(const std::string& method, const nlohmann::json& params) {
  if (method == "ping") {
    return {{"ok", true}};
  }
  if (method == "conversation.create") {
    return to_json(storage_.create_conversation(optional_string(params, "title")));
  }
  if (method == "conversation.list") {
    nlohmann::json items = nlohmann::json::array();
    for (const auto& conversation : storage_.list_conversations()) {
      items.push_back(to_json(conversation));
    }
    return items;
  }
  if (method == "conversation.rename") {
    return to_json(storage_.rename_conversation(require_string(params, "conversationId"), require_string(params, "title")));
  }
  if (method == "conversation.delete") {
    storage_.delete_conversation(require_string(params, "conversationId"));
    return {{"ok", true}};
  }
  if (method == "message.append") {
    auto message = messages_.append_message(require_string(params, "conversationId"), require_string(params, "role"),
                                            optional_string(params, "content"), optional_string(params, "status", "complete"));
    index_.index_message(message);
    return to_json(message);
  }
  if (method == "message.update") {
    return to_json(messages_.update_message_content(require_string(params, "messageId"), optional_string(params, "content")));
  }
  if (method == "message.finalize") {
    return to_json(messages_.finalize_message(require_string(params, "messageId")));
  }
  if (method == "message.loadRecent") {
    nlohmann::json items = nlohmann::json::array();
    for (const auto& message : messages_.load_recent_messages(require_string(params, "conversationId"), optional_int(params, "limit", 50))) {
      items.push_back(to_json(message));
    }
    return items;
  }
  if (method == "message.loadBefore") {
    nlohmann::json items = nlohmann::json::array();
    for (const auto& message : messages_.load_messages_before(require_string(params, "conversationId"),
                                                               require_string(params, "beforeMessageId"),
                                                               optional_int(params, "limit", 50))) {
      items.push_back(to_json(message));
    }
    return items;
  }
  if (method == "message.search") {
    return index_.search_messages(optional_string(params, "query"), optional_int(params, "limit", 50));
  }
  if (method == "context.build") {
    return context_.build_context(require_string(params, "conversationId"), optional_string(params, "currentInput"),
                                  optional_int(params, "maxTokens", 4096));
  }
  if (method == "token.estimate") {
    const auto conversation_id = require_string(params, "conversationId");
    const auto current_input = optional_string(params, "currentInput");
    const auto messages = storage_.load_recent_messages(conversation_id, 50);
    return token_budget_.build_budget(messages, current_input, optional_int(params, "maxTokens", 4096));
  }
  if (method == "layout.estimate") {
    LayoutCache layout_cache;
    return to_json(layout_cache.estimate_message_layout(optional_string(params, "content")));
  }
  if (method == "stats.get") {
    return {{"visibleMessageCount", 0},
            {"cachedMessageCount", storage_.message_count()},
            {"estimatedLayoutCount", storage_.layout_count()},
            {"streamingFps", 0},
            {"streamUpdateIntervalMs", 0}};
  }

  throw CoreError("METHOD_NOT_FOUND", "Unknown method: " + method);
}

}  // namespace chat_core
