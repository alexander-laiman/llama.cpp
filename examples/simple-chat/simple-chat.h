#pragma once

#include "llama-simple-chat_export.h"  // Changed from llama_simple_chat_export.h to match generated file

#ifdef __cplusplus
extern "C" {
#endif

// Define your API functions with proper export macros
LLAMA_SIMPLE_CHAT_EXPORT void initialize_llama_model(const char* model_path, void (*log_callback)(const char*));
LLAMA_SIMPLE_CHAT_EXPORT char* generate_response(const char* prompt);
LLAMA_SIMPLE_CHAT_EXPORT void free_response(char* response);
LLAMA_SIMPLE_CHAT_EXPORT void shutdown_llama_model();

#ifdef __cplusplus
}
#endif 