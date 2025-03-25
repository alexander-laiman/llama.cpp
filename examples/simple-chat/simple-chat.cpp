#include "simple-chat.h"
#include "common.h"
#include "llama.h"
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>
#include <mutex>

// Global state
static llama_model* model = nullptr;
static llama_context* ctx = nullptr;
static llama_sampler* smpl = nullptr;
static std::mutex model_mutex;
static simple_log_callback g_simple_log_callback = nullptr;

static void print_usage(int, char ** argv) {
    printf("\nexample usage:\n");
    printf("\n    %s -m model.gguf [-c context_size] [-ngl n_gpu_layers]\n", argv[0]);
    printf("\n");
}
static void forward_log_callback(ggml_log_level level, const char* text, void* user_data) {
    (void)user_data; // Unused
    
    if (g_simple_log_callback) {
        // Forward the log message to our simple callback
        g_simple_log_callback((int)level, text);
    } else {
        // Fall back to default behavior if no callback is set
        fprintf(stderr, "%s", text);
        fflush(stderr);
    }
}

// Function to set our simple callback
LLAMA_SIMPLE_CHAT_EXPORT void set_simple_log_callback(simple_log_callback callback) {
    g_simple_log_callback = callback;
    
    // Set our forwarding callback as the ggml log callback
    llama_log_set(forward_log_callback, nullptr);
}

// Implementation of exported functions
LLAMA_SIMPLE_CHAT_EXPORT void initialize_llama_model(const char* model_path, void (*log_callback)(const char*)) {
    std::lock_guard<std::mutex> lock(model_mutex);
    if (log_callback) {
        log_callback("Starting model initialization");
    }
    if (model != nullptr) {
        shutdown_llama_model();
        if (log_callback) {
            log_callback("Shutdown existing model");
        }
    }
    
    // Initialize backends
    ggml_backend_load_all();
    if (log_callback) {
        log_callback("Backends initialized");
    }
    // Initialize model with appropriate parameters
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 99; // Use GPU acceleration if available
    model = llama_model_load_from_file(model_path, model_params);
    
    if (model == nullptr) {
        if (log_callback) {
            log_callback("Failed to load model");
        }
        return;
    }
    
    // Initialize context
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 2048;
    ctx_params.n_batch = 2048;
    
    ctx = llama_init_from_model(model, ctx_params);
    if (ctx == nullptr) {
        if (log_callback) {
            log_callback("Failed to initialize context");
        }
        return;
    }
    
    if (log_callback) {
        log_callback("Context initialized successfully");
    }
}

LLAMA_SIMPLE_CHAT_EXPORT char* generate_response(const char* prompt) {
    std::lock_guard<std::mutex> lock(model_mutex);

    if (ctx == nullptr || model == nullptr) {
        return _strdup("Model not initialized");  // Using _strdup instead of strdup for MSVC
    }
    
    // Implement the generation logic here
    // This would be adapted from the main() function's generate lambda
    
    // For now, just a placeholder
    //std::string result = "Response from LLaMA model";  // Replace with actual generation
    
    //return _strdup(result.c_str());  // Using _strdup instead of strdup for MSVC

    
    // initialize the sampler
    smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_min_p(0.05f, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.8f));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    const llama_vocab * vocab = llama_model_get_vocab(model);

    // helper function to evaluate a prompt and generate a response
    auto generate = [&](const std::string & sprompt) {
        std::string response;

        const bool is_first = llama_kv_self_used_cells(ctx) == 0;

        // tokenize the prompt
        const int n_prompt_tokens = -llama_tokenize(vocab, sprompt.c_str(), sprompt.size(), NULL, 0, is_first, true);
        std::vector<llama_token> prompt_tokens(n_prompt_tokens);
        if (llama_tokenize(vocab, sprompt.c_str(), sprompt.size(), prompt_tokens.data(), prompt_tokens.size(), is_first, true) < 0) {
            GGML_ABORT("failed to tokenize the prompt\n");
        }

        // prepare a batch for the prompt
        llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());
        llama_token new_token_id;
        while (true) {
            // check if we have enough space in the context to evaluate this batch
            int n_ctx = llama_n_ctx(ctx);
            int n_ctx_used = llama_kv_self_used_cells(ctx);
            if (n_ctx_used + batch.n_tokens > n_ctx) {
                printf("\033[0m\n");
                fprintf(stderr, "context size exceeded\n");
                exit(0);
            }

            if (llama_decode(ctx, batch)) {
                GGML_ABORT("failed to decode\n");
            }

            // sample the next token
            new_token_id = llama_sampler_sample(smpl, ctx, -1);

            // is it an end of generation?
            if (llama_vocab_is_eog(vocab, new_token_id)) {
                break;
            }

            // convert the token to a string, print it and add it to the response
            char buf[256];
            int n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
            if (n < 0) {
                GGML_ABORT("failed to convert token to piece\n");
            }
            std::string piece(buf, n);
            printf("%s", piece.c_str());
            fflush(stdout);
            response += piece;

            // prepare the next batch with the sampled token
            batch = llama_batch_get_one(&new_token_id, 1);
        }

        return response;
    };

    std::vector<llama_chat_message> messages;
    std::string result;
    std::vector<char> formatted(llama_n_ctx(ctx));
    int prev_len = 0;
        // get user input
        std::string user;
        user = prompt ? std::string(prompt) : "";

        if (user=="") {
            result = "No prompt passed...\n";  
            return _strdup(result.c_str());
        }

        const char * tmpl = llama_model_chat_template(model, /* name */ nullptr);

        // add the user input to the message list and format it
        messages.push_back({"user", _strdup(user.c_str())});
        int new_len = llama_chat_apply_template(tmpl, messages.data(), messages.size(), true, formatted.data(), formatted.size());
        if (new_len > (int)formatted.size()) {
            formatted.resize(new_len);
            new_len = llama_chat_apply_template(tmpl, messages.data(), messages.size(), true, formatted.data(), formatted.size());
        }
        if (new_len < 0) {
            fprintf(stderr, "failed to apply the chat template\n");
            result = "failed to apply the chat template\n";  
            return _strdup(result.c_str());
        }

        // remove previous messages to obtain the prompt to generate the response
        std::string sprompt(formatted.begin() + prev_len, formatted.begin() + new_len);

        // generate a response
        printf("\033[33m");
        std::string response = generate(sprompt);
        printf("\n\033[0m");

        // add the response to the messages
        messages.push_back({"assistant", _strdup(response.c_str())});
        prev_len = llama_chat_apply_template(tmpl, messages.data(), messages.size(), false, nullptr, 0);
        if (prev_len < 0) {
            fprintf(stderr, "failed to apply the chat template\n");
            result = "failed to apply the chat template\n";  
            return _strdup(result.c_str());
        }
        return _strdup(response.c_str());
}

LLAMA_SIMPLE_CHAT_EXPORT void free_response(char* response) {
    if (response) {
        free(response);
    }
}

LLAMA_SIMPLE_CHAT_EXPORT void shutdown_llama_model() {
    std::lock_guard<std::mutex> lock(model_mutex);
    
    if (ctx) {
        llama_free(ctx);
        ctx = nullptr;
    }
    
    if (model) {
        llama_model_free(model);
        model = nullptr;
    }
    if(smpl) {
        llama_sampler_free(smpl);
    }
    
    llama_backend_free();
}
// Creates a single embedding at the given address, returns the number of dimensions
LLAMA_SIMPLE_CHAT_EXPORT int create_single_embedding(const char* prompt, float** embedding_out, void (*log_callback)(const char*)) {
    std::lock_guard<std::mutex> lock(model_mutex);

    if (ctx == nullptr || model == nullptr) {
        if (log_callback) {
            log_callback("Model not initialized. Call initialize_llama_model first.");
        }
        return 0;
    }
    
    if (log_callback) {
        log_callback("Creating embedding for prompt...");
    }
    llama_set_embeddings(ctx, true);

    // Use the global model and vocab
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const enum llama_pooling_type pooling_type = llama_pooling_type(ctx);
    const int n_embd = llama_model_n_embd(model);
    
    // Tokenize the prompt
    if (log_callback) {
        log_callback("Tokenizing input...");
    }
    
    auto inp = common_tokenize(ctx, prompt, true, true);
    
    if (inp.empty()) {
        if (log_callback) {
            log_callback("Error: Empty tokenization result");
        }
        return 0;
    }
    
    // Check if the last token is SEP
    if (inp.back() != llama_vocab_sep(vocab)) {
        if (log_callback) {
            log_callback("Warning: Last token in prompt is not SEP. 'tokenizer.ggml.add_eos_token' should be set to 'true' in the GGUF header");
        }
    }
    
    if (log_callback) {
        char msg[256];
        snprintf(msg, sizeof(msg), "Tokenized %zu tokens", inp.size());
        log_callback(msg);
    }
    
    // Initialize batch
    const int n_batch = inp.size(); // Use input size as batch size
    struct llama_batch batch = llama_batch_init(n_batch, 0, 1);
    
    // Add tokens to batch
    for (size_t i = 0; i < inp.size(); i++) {
        common_batch_add(batch, inp[i], i, {0}, true);
    }
    
    // Determine number of embeddings to generate
    int n_embd_count = (pooling_type == LLAMA_POOLING_TYPE_NONE) ? inp.size() : 1;
    
    if (log_callback) {
        char msg[256];
        snprintf(msg, sizeof(msg), "Will generate %d embeddings with %d dimensions each", n_embd_count, n_embd);
        log_callback(msg);
    }
    
    // Allocate output - we'll use a dynamically allocated array that the caller will need to free
    float* embeddings = new float[n_embd_count * n_embd];
    memset(embeddings, 0, n_embd_count * n_embd * sizeof(float));
    
    // Clear previous kv_cache values (irrelevant for embeddings)
    llama_kv_self_clear(ctx);
    
    if (log_callback) {
        log_callback("Running model inference...");
    }
    
    // Run model
    if (llama_model_has_encoder(model) && !llama_model_has_decoder(model)) {
        // encoder-only model
        if (log_callback) {
            log_callback("Using encoder-only model");
        }
        if (llama_encode(ctx, batch) < 0) {
            if (log_callback) {
                log_callback("Error: Failed to encode");
            }
            llama_batch_free(batch);
            delete[] embeddings;
            return 0;
        }
    } else if (!llama_model_has_encoder(model) && llama_model_has_decoder(model)) {
        // decoder-only model
        if (log_callback) {
            log_callback("Using decoder-only model");
        }
        if (llama_decode(ctx, batch) < 0) {
            if (log_callback) {
                log_callback("Error: Failed to decode");
            }
            llama_batch_free(batch);
            delete[] embeddings;
            return 0;
        }
    } else {
        // encoder-decoder models not supported
        if (log_callback) {
            log_callback("Error: Encoder-decoder models are not supported for embeddings");
        }
        llama_batch_free(batch);
        delete[] embeddings;
        return 0;
    }
    
    if (log_callback) {
        log_callback("Extracting embeddings...");
    }
    
    // Extract embeddings
    int extracted_count = 0;
    for (int i = 0; i < batch.n_tokens; i++) {
        if (!batch.logits[i]) {
            continue;
        }
        
        const float* embd = nullptr;
        int embd_pos = 0;
        
        if (pooling_type == LLAMA_POOLING_TYPE_NONE) {
            // Get token embeddings
            embd = llama_get_embeddings_ith(ctx, i);
            embd_pos = i;
        } else {
            // Get sequence embeddings
            embd = llama_get_embeddings_seq(ctx, batch.seq_id[i][0]);
            embd_pos = batch.seq_id[i][0];
        }
        
        if (embd == nullptr) {
            if (log_callback) {
                char msg[256];
                snprintf(msg, sizeof(msg), "Warning: Failed to get embedding for token %d", i);
                log_callback(msg);
            }
            continue;
        }
        
        float* out = embeddings + embd_pos * n_embd;
        common_embd_normalize(embd, out, n_embd, 1); // Always normalize embeddings
        extracted_count++;
    }
    
    if (log_callback) {
        char msg[256];
        snprintf(msg, sizeof(msg), "Successfully extracted %d embeddings", extracted_count);
        log_callback(msg);
    }
    
    // Clean up
    llama_batch_free(batch);
    
    // Set the output pointer
    *embedding_out = embeddings;
    
    if (log_callback) {
        log_callback("Embedding creation complete");
    }
    
    // Return the number of dimensions
    return n_embd;
}

// Function to free the embedding memory
LLAMA_SIMPLE_CHAT_EXPORT void free_embedding(float* embedding) {
    if (embedding) {
        delete[] embedding;
    }
} 