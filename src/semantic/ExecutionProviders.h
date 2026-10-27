/*
 * ExecutionProviders.h
 *
 * ONNX Runtime execution providers the embedding backend can target.
 * Automatic selection tries GPU providers advertised by the loaded runtime,
 * then CPU. OpenVINO explicitly targets GPU with FP32 precision. Initialization
 * includes real inference; failed or non-finite GPU inference retries on CPU.
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
    std::string id;       /* auto, cpu, cuda, openvino */
    std::string ort_name; /* CPUExecutionProvider, ... */
    bool runtime_present = false;
};

/* Automatic and CPU are always listed. Other rows are listed only when the
 * loaded ONNX Runtime reports them. When the runtime library is missing,
 * CPU is listed and runtime_present is false. */
std::vector<ExecProviderInfo> execution_provider_catalog();

struct OnnxProviderConfig {
    std::string model_path;
    std::string vocab_path;
    int dimension = 0;
    int max_tokens = 256;
    std::string pooling;            /* mean */
    std::string execution_provider = "auto"; /* auto, cpu, cuda, openvino */
};

std::unique_ptr<EmbeddingProvider> make_onnx_provider(const OnnxProviderConfig &config,
                                                     std::string &error);

std::unique_ptr<EmbeddingProvider> make_turboembed_provider(std::string &error);

#endif
