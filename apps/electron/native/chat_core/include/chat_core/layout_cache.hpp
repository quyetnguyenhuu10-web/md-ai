#pragma once

#include <chat_core/message.hpp>
#include <string>

namespace chat_core {

class LayoutCache {
 public:
  LayoutMetadata estimate_message_layout(const std::string& content) const;
};

}  // namespace chat_core
