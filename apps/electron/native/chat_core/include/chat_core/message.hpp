#pragma once

#include <nlohmann/json.hpp>
#include <string>

namespace chat_core {

struct LayoutMetadata {
  int estimated_height = 96;
  bool has_code_block = false;
  bool has_markdown = false;
  bool has_math = false;
  bool has_image = false;
  bool is_long_message = false;
};

struct Message {
  std::string id;
  std::string conversation_id;
  std::string role;
  std::string content;
  std::string status;
  long long created_at = 0;
  long long updated_at = 0;
  int token_estimate = 0;
  LayoutMetadata layout;
};

nlohmann::json to_json(const LayoutMetadata& layout);
nlohmann::json to_json(const Message& message);

}  // namespace chat_core
