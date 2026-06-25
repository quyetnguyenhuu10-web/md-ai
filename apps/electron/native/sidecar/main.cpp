#include <chat_core/error.hpp>
#include <chat_core/json_rpc.hpp>
#include <chat_core/storage_sqlite.hpp>
#include <nlohmann/json.hpp>

#include <filesystem>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
  const std::string storage_root = argc > 1 ? argv[1] : "conversations";
  const std::string legacy_db_path = argc > 2 ? argv[2] : "";

  try {
    std::filesystem::create_directories(storage_root);

    chat_core::StorageSQLite storage(storage_root, legacy_db_path);
    chat_core::JsonRpcServer server(storage);

    std::string line;
    while (std::getline(std::cin, line)) {
      if (line.empty()) {
        continue;
      }

      try {
        auto request = nlohmann::json::parse(line);
        std::cout << server.handle(request).dump() << std::endl;
      } catch (const std::exception& error) {
        std::cout << nlohmann::json({{"id", ""},
                                     {"error", {{"code", "BAD_JSON"}, {"message", error.what()}}}})
                         .dump()
                  << std::endl;
      }
    }
  } catch (const std::exception& error) {
    std::cerr << "sidecar fatal: " << error.what() << std::endl;
    return 1;
  }

  return 0;
}
