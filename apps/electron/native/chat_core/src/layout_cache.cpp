#include <chat_core/layout_cache.hpp>

#include <algorithm>

namespace chat_core {

LayoutMetadata LayoutCache::estimate_message_layout(const std::string& content) const {
  LayoutMetadata layout;
  layout.has_code_block = content.find("```") != std::string::npos;
  layout.has_markdown = content.find('#') != std::string::npos || content.find('*') != std::string::npos ||
                        content.find('`') != std::string::npos;
  layout.has_math = content.find('$') != std::string::npos;
  layout.has_image = content.find("![") != std::string::npos || content.find("<img") != std::string::npos;
  layout.is_long_message = content.size() > 900;
  const auto lines = std::count(content.begin(), content.end(), '\n') + 1;
  layout.estimated_height = static_cast<int>(std::min<std::size_t>(640, 72 + (content.size() / 58) * 20 + lines * 10));
  if (layout.has_code_block) {
    layout.estimated_height += 70;
  }
  return layout;
}

nlohmann::json to_json(const LayoutMetadata& layout) {
  return {
      {"estimatedHeight", layout.estimated_height},
      {"hasCodeBlock", layout.has_code_block},
      {"hasMarkdown", layout.has_markdown},
      {"hasMath", layout.has_math},
      {"hasImage", layout.has_image},
      {"isLongMessage", layout.is_long_message},
  };
}

nlohmann::json to_json(const Message& message) {
  return {
      {"id", message.id},
      {"conversationId", message.conversation_id},
      {"role", message.role},
      {"content", message.content},
      {"status", message.status},
      {"createdAt", message.created_at},
      {"updatedAt", message.updated_at},
      {"tokenEstimate", message.token_estimate},
      {"layout", to_json(message.layout)},
  };
}

}  // namespace chat_core
