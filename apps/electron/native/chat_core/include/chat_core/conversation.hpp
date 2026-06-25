#pragma once

#include <nlohmann/json.hpp>
#include <string>

namespace chat_core {

struct Conversation {
  std::string id;
  std::string title;
  long long created_at = 0;
  long long updated_at = 0;
};

nlohmann::json to_json(const Conversation& conversation);

}  // namespace chat_core
