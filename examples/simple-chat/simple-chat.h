#pragma once

#include "llama-simple-chat_export.h"  // Changed from llama_simple_chat_export.h to match generated file

#ifdef __cplusplus
extern "C" {
#endif
typedef void (*simple_log_callback)(int level, const char* message);

// Define your API functions with proper export macros
LLAMA_SIMPLE_CHAT_EXPORT void initialize_llama_model(const char* model_path, void (*log_callback)(const char*));
LLAMA_SIMPLE_CHAT_EXPORT char* generate_response(const char* prompt);
LLAMA_SIMPLE_CHAT_EXPORT void free_response(char* response);
LLAMA_SIMPLE_CHAT_EXPORT void shutdown_llama_model();
LLAMA_SIMPLE_CHAT_EXPORT int create_single_embedding(const char* prompt, float** embedding_out, void (*log_callback)(const char*));
LLAMA_SIMPLE_CHAT_EXPORT void free_embedding(float* embedding);
LLAMA_SIMPLE_CHAT_EXPORT void set_simple_log_callback(simple_log_callback callback);

#ifdef __cplusplus
}
#endif 