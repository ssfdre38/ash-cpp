// Ollama-based tokenizer bridge
// Temporary solution until we integrate llama.cpp tokenizer
#include "ollama_tokenizer.h"
#include "logger.h"
#include <sstream>
#include <curl/curl.h>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace ash {

// Callback for curl to write response
static size_t write_callback(void* contents, size_t size, size_t nmemb, std::string* s) {
    size_t new_length = size * nmemb;
    try {
        s->append((char*)contents, new_length);
    } catch (std::bad_alloc& e) {
        return 0;
    }
    return new_length;
}

OllamaTokenizer::OllamaTokenizer(const std::string& model_name, const std::string& ollama_url)
    : model_name_(model_name), ollama_url_(ollama_url) {
    curl_global_init(CURL_GLOBAL_DEFAULT);
}

OllamaTokenizer::~OllamaTokenizer() {
    curl_global_cleanup();
}

std::vector<TokenID> OllamaTokenizer::encode(const std::string& text) {
    CURL* curl = curl_easy_init();
    if (!curl) {
        Logger::instance().error("Failed to initialize CURL");
        return {};
    }
    
    // Build JSON request
    json req = {
        {"model", model_name_},
        {"prompt", text},
        {"stream", false},
        {"raw", true}  // No system prompt
    };
    
    std::string req_str = req.dump();
    std::string response_str;
    
    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    
    curl_easy_setopt(curl, CURLOPT_URL, (ollama_url_ + "/api/generate").c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, req_str.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_str);
    
    CURLcode res = curl_easy_perform(curl);
    
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);
    
    if (res != CURLE_OK) {
        Logger::instance().error("CURL request failed: " + std::string(curl_easy_strerror(res)));
        return {};
    }
    
    // Parse response to get context (token IDs)
    try {
        json resp = json::parse(response_str);
        if (resp.contains("context")) {
            auto context = resp["context"];
            std::vector<TokenID> tokens;
            for (const auto& tok : context) {
                tokens.push_back(tok.get<TokenID>());
            }
            return tokens;
        }
    } catch (const json::exception& e) {
        Logger::instance().error("Failed to parse Ollama response: " + std::string(e.what()));
    }
    
    return {};
}

std::string OllamaTokenizer::decode(const std::vector<TokenID>& tokens) {
    // For decode, we can use our existing GGUF vocab
    // This is a fallback - decode should work from our vocab
    Logger::instance().warning("OllamaTokenizer::decode not implemented - use SimpleSPTokenizer");
    return "";
}

} // namespace ash
