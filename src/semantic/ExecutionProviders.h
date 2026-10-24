/*
 * ExecutionProviders.h
 *
 * ONNX Runtime execution providers the embedding backend can target.
 * This build creates a CPU session. If a GPU provider is requested and
 * the loaded libonnxruntime reports it, the session tries that provider
 * and falls back to CPU when session creation fails. The names are the
 * ones ORT advertises: CUDA, MIGraphX, OpenVINO, WebGPU.
 *
 * TurboEmbed (https://github.com/ai-pipestream/turboembed) is a later
 * backend behind the same EmbeddingProvider. It is not linked here.
 * A future wrapper would dlopen libturboembed and call, in order,
 * turbo_runtime_create, turbo_context_create, turbo_model_load,
 * turbo_session_create, turbo_embed_write_text, turbo_session_run,
 * and turbo_result_read. make_turboembed_provider() reports that.
 */
#ifndef NPP_EXECUTION_PROVIDERS_H
#define NPP_EXECUTION_PROVIDERS_H

#include "SemanticInterfaces.h"

#include <string>
#include <vector>

struct ExecProviderInfo {
    std::string id;       /* cpu, cuda, migraphx, openvino, webgpu */
    std::string ort_name; /* CPUExecutionProvider, ... */
    bool runtime_present = false;
};

/* CPU is always listed. Other rows are listed only when the linked
 * ONNX Runtime reports them. With ONNX Runtime not linked, CPU is
 * listed and runtime_present is false. */
std::vector<ExecProviderInfo> execution_provider_catalog();

struct OnnxProviderConfig {
    std::string model_path;
    std::string vocab_path;
    int dimension = 0;
    int max_tokens = 256;
    std::string pooling;            /* mean */
    std::string execution_provider; /* cpu, cuda, migraphx, openvino, webgpu */
};

std::unique_ptr<EmbeddingProvider> make_onnx_provider(const OnnxProviderConfig &config,
                                                     std::string &error);

std::unique_ptr<EmbeddingProvider> make_turboembed_provider(std::string &error);

#endif
