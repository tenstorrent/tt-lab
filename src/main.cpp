// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "gguf.h"
#include "model.h"
#include "requant.h"
#include "tokenizer.h"
#include "tt_backend.h"

#include <array>
#include <chrono>
#include <cstdio>
#include <exception>
#include <iostream>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace {

void usage(const char *program) {
    std::cerr << "usage: " << program << " run -m MODEL -p PROMPT [-n N|--n-predict N] [--dump-tensors DIR]\n"
              << "       " << program << " run -m MODEL --sim LIBTTSIM --ttq MODEL.ttq "
              << "-p PROMPT [-n OUTPUT_TOKENS] [--profile]\n"
              << "       " << program << " run -m MODEL --sim LIBTTSIM --check --ttq MODEL.ttq "
              << "-p PROMPT [-n OUTPUT_TOKENS] [--debug-layers N] [--profile]\n"
              << "       " << program << " run -m MODEL -p PROMPT --device-proxy --ttq MODEL.ttq [-n N]\n"
              << "       " << program << " run -m MODEL -p PROMPT --device-proxy --check --ttq MODEL.ttq "
              << "[--debug-layers N]\n"
              << "       " << program << " run -m MODEL -p PROMPT --device --ttq MODEL.ttq "
              << "[-n OUTPUT_TOKENS] [--profile]\n"
              << "       " << program << " run -m MODEL -p PROMPT --device --check --ttq MODEL.ttq "
              << "[-n OUTPUT_TOKENS] [--debug-layers N] [--tiles N] [--profile]\n"
              << "       " << program << " inspect -m MODEL\n"
              << "       " << program << " requant -m MODEL -o MODEL.ttq\n"
              << "       " << program << " tokenize -m MODEL -p PROMPT\n"
              << "--tiles 1|8|32: 32 selects four chips; required for 120b, optional for 20b.\n"
              << "--profile: report device stage cycles and host timings with --sim or --device.\n";
}

void print_value(const GgufFile &file, std::string_view key, const GgufValue &value) {
    if (value.type == GgufType::string) {
        constexpr size_t limit = 100;
        std::cout << std::quoted(std::string(value.text.substr(0, limit)));
        if (value.text.size() > limit) {
            std::cout << "...";
        }
    } else if (value.type == GgufType::array) {
        std::cout << gguf_type_name(value.element_type) << '[' << value.count << ']';
    } else if (value.type == GgufType::boolean) {
        std::cout << (file.boolean(key) ? "true" : "false");
    } else if (value.type == GgufType::f32 || value.type == GgufType::f64) {
        std::cout << file.number(key);
    } else {
        std::cout << file.integer(key);
    }
}

void print_info(const GgufFile &file) {
    std::cout << "GGUF version " << file.version() << ", " << file.keys().size() << " metadata keys, "
              << file.tensors().size() << " tensors\n";
    for (const std::string_view key : file.keys()) {
        std::cout << "  " << key << " = ";
        print_value(file, key, *file.find(key));
        std::cout << '\n';
    }
    std::cout << "tensor data offset: " << file.data_offset() << "\n";
    for (const GgufTensor &tensor : file.tensors()) {
        std::cout << "  " << tensor.name << " [";
        for (size_t d = 0; d < tensor.shape.size(); ++d) {
            if (d) {
                std::cout << " x ";
            }
            std::cout << tensor.shape[d];
        }
        std::cout << "] " << tensor_type_name(tensor.type) << " offset " << tensor.offset << '\n';
    }
}

double elapsed_seconds(std::chrono::steady_clock::time_point begin, std::chrono::steady_clock::time_point end) {
    return std::chrono::duration<double>(end - begin).count();
}

bool parse_size(const std::string &text, size_t *output) {
    try {
        size_t used = 0;
        const unsigned long long parsed = std::stoull(text, &used);
        if (used != text.size() || parsed > std::numeric_limits<size_t>::max()) {
            return false;
        }
        *output = size_t(parsed);
        return true;
    } catch (const std::exception &) {
        return false;
    }
}

void print_tokens(const std::string &prompt, const std::vector<int32_t> &tokens) {
    std::cout << "prompt: " << prompt << "\ntokens (" << tokens.size() << "): [";
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (i) {
            std::cout << ", ";
        }
        std::cout << tokens[i];
    }
    std::cout << "]\n";
}

} // namespace

int main(int argc, char **argv) {
    const char *model_path = nullptr;
    const char *output_path = nullptr;
    const char *prompt = nullptr;
    bool predict_set = false;
    size_t predict = 0;
    const char *sim_path = nullptr;
    bool device_proxy = false;
    bool check = false;
    bool profile = false;
    const char *ttq_path = nullptr;
    size_t debug_layers = 0;
    size_t shard_count = 1;
    std::string dump_directory;
    bool device = false;

    if (argc >= 2 && (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help")) {
        usage(argv[0]);
        return 0;
    }
    if (argc < 2) {
        usage(argv[0]);
        return 2;
    }
    const std::string command = argv[1];
    if (command != "run" && command != "inspect" && command != "requant" && command != "tokenize") {
        std::cerr << "unknown command: " << command << '\n';
        usage(argv[0]);
        return 2;
    }

    for (int i = 2; i < argc; ++i) {
        const std::string argument = argv[i];
        if ((argument == "-m" || argument == "--model") && i + 1 < argc) {
            model_path = argv[++i];
        } else if ((argument == "-o" || argument == "--output") && i + 1 < argc) {
            if (command != "requant") {
                std::cerr << argument << " is only valid for 'requant'\n";
                usage(argv[0]);
                return 2;
            }
            output_path = argv[++i];
        } else if ((argument == "-p" || argument == "--prompt") && i + 1 < argc) {
            prompt = argv[++i];
        } else if ((argument == "-n" || argument == "--predict" || argument == "--n-predict") && i + 1 < argc) {
            if (command != "run") {
                std::cerr << argument << " is only valid for 'run'\n";
                usage(argv[0]);
                return 2;
            }
            const std::string value = argv[++i];
            if (!parse_size(value, &predict)) {
                std::cerr << "invalid token count: " << value << '\n';
                usage(argv[0]);
                return 2;
            }
            predict_set = true;
        } else if (argument == "--dump-tensors" && i + 1 < argc) {
            if (command != "run") {
                std::cerr << "--dump-tensors is only valid for 'run'\n";
                usage(argv[0]);
                return 2;
            }
            dump_directory = argv[++i];
        } else if (argument == "--sim" && i + 1 < argc) {
            if (command != "run") {
                std::cerr << "--sim is only valid for 'run'\n";
                usage(argv[0]);
                return 2;
            }
            sim_path = argv[++i];
        } else if (argument == "--device-proxy") {
            if (command != "run") {
                std::cerr << "--device-proxy is only valid for 'run'\n";
                usage(argv[0]);
                return 2;
            }
            device_proxy = true;
        } else if (argument == "--check") {
            if (command != "run") {
                std::cerr << "--check is only valid for 'run'\n";
                usage(argv[0]);
                return 2;
            }
            check = true;
        } else if (argument == "--device") {
            if (command != "run") {
                std::cerr << "--device is only valid for 'run'\n";
                usage(argv[0]);
                return 2;
            }
            device = true;
        } else if (argument == "--ttq" && i + 1 < argc) {
            if (command != "run") {
                std::cerr << "--ttq is only valid for 'run'\n";
                usage(argv[0]);
                return 2;
            }
            ttq_path = argv[++i];
        } else if (argument == "--debug-layers" && i + 1 < argc) {
            if (command != "run") {
                std::cerr << "--debug-layers is only valid for 'run'\n";
                usage(argv[0]);
                return 2;
            }
            const std::string value = argv[++i];
            if (!parse_size(value, &debug_layers)) {
                std::cerr << "invalid debug layer count: " << value << '\n';
                usage(argv[0]);
                return 2;
            }
        } else if (argument == "--profile") {
            if (command != "run") {
                std::cerr << "--profile is only valid for 'run'\n";
                usage(argv[0]);
                return 2;
            }
            profile = true;
        } else if (argument == "--tiles" && i + 1 < argc) {
            if (command != "run") {
                std::cerr << "--tiles is only valid for 'run'\n";
                usage(argv[0]);
                return 2;
            }
            const std::string value = argv[++i];
            if (!parse_size(value, &shard_count)) {
                std::cerr << "invalid tile count: " << value << '\n';
                usage(argv[0]);
                return 2;
            }
        } else if (argument == "-h" || argument == "--help") {
            usage(argv[0]);
            return 0;
        } else {
            std::cerr << "unknown or incomplete option: " << argument << '\n';
            usage(argv[0]);
            return 2;
        }
    }
    if (!model_path) {
        usage(argv[0]);
        return 2;
    }
    if (command != "inspect" && command != "requant" && !prompt) {
        usage(argv[0]);
        return 2;
    }
    if ((command == "inspect" || command == "requant") && prompt) {
        std::cerr << "-p/--prompt is not valid for '" << command << "'\n";
        usage(argv[0]);
        return 2;
    }
    if (command != "requant" && output_path) {
        std::cerr << "-o/--output is only valid for 'requant'\n";
        usage(argv[0]);
        return 2;
    }
    if (command == "requant" && !output_path) {
        usage(argv[0]);
        return 2;
    }

    try {
        GgufFile file(model_path);
        if (command == "inspect") {
            GptOssModel validated_model(file);
            (void)validated_model;
            print_info(file);
            return 0;
        }
        if (command == "requant") {
            write_requant(output_path, file);
            return 0;
        }
        if (command == "tokenize") {
            Tokenizer tokenizer(file);
            print_tokens(prompt, tokenizer.encode(prompt));
            return 0;
        }
        if (command == "run") {
            Tokenizer tokenizer(file);
            const std::string formatted_prompt = harmony_prompt(prompt);
            const std::vector<int32_t> prompt_tokens = tokenizer.encode(formatted_prompt);
            if (prompt_tokens.size() > GptOssModel::context_length) {
                throw std::runtime_error("prompt exceeds the model context");
            }
            const size_t generation_limit = predict_set ? predict : GptOssModel::context_length - prompt_tokens.size();
            if (generation_limit > GptOssModel::context_length - prompt_tokens.size()) {
                throw std::runtime_error("prompt and generation exceed the model context");
            }
            if (debug_layers && !check) {
                throw std::runtime_error("--debug-layers is only valid with --check");
            }
            if (sim_path && device) {
                throw std::runtime_error("--sim and --device are mutually exclusive");
            }
            if (check && !dump_directory.empty()) {
                throw std::runtime_error("--check cannot be combined with tensor dumps");
            }

            std::optional<TtFirmwareBackend> firmware_backend;
            if (sim_path) {
                firmware_backend = TtFirmwareBackend::sim;
            } else if (device) {
                firmware_backend = TtFirmwareBackend::silicon;
            }
            if (device_proxy && firmware_backend) {
                throw std::runtime_error("--device-proxy cannot be combined with --sim or --device");
            }
            if (shard_count != 1 && !firmware_backend) {
                throw std::runtime_error("--tiles is only valid with --sim or --device");
            }
            const TtMeshConfig mesh = {.shard_count = shard_count};
            if (profile) {
                if (!firmware_backend) {
                    throw std::runtime_error("--profile requires --sim or --device");
                }
                tt_enable_profiling();
            }
            if (check) {
                if (!device_proxy && !firmware_backend) {
                    throw std::runtime_error("--check requires --device-proxy, --sim, or --device");
                }
                if (!ttq_path) {
                    throw std::runtime_error("--check requires --ttq");
                }
                if (firmware_backend) {
                    if (prompt_tokens.empty()) {
                        throw std::runtime_error("TT firmware check needs at least one prompt token");
                    }
                    run_tt_firmware_proxy_check(*firmware_backend, sim_path, ttq_path, file, prompt_tokens[0],
                                                generation_limit, debug_layers, mesh);
                    return 0;
                }
                if (device_proxy) {
                    run_device_proxy_check(ttq_path, file, prompt_tokens.data(), prompt_tokens.size(), debug_layers);
                    return 0;
                }
            }
            if (firmware_backend) {
                if (!dump_directory.empty()) {
                    throw std::runtime_error("TT firmware inference cannot be combined with tensor dumps");
                }
                if (!ttq_path) {
                    throw std::runtime_error("TT firmware inference requires --ttq");
                }
                if (prompt_tokens.empty()) {
                    throw std::runtime_error("TT firmware inference needs at least one prompt token");
                }
                run_tt_firmware_inference(*firmware_backend, sim_path, ttq_path, file, tokenizer, formatted_prompt,
                                          prompt_tokens.data(), prompt_tokens.size(), generation_limit, mesh);
                return 0;
            }
            if (device_proxy) {
                if (!ttq_path) {
                    throw std::runtime_error("--device-proxy requires --ttq");
                }
                run_device_proxy_generation(ttq_path, file, tokenizer, formatted_prompt, prompt_tokens.data(),
                                            prompt_tokens.size(), generation_limit);
                return 0;
            }
            const size_t token_capacity = prompt_tokens.size() + generation_limit;
            GptOssModel model(file, token_capacity, dump_directory, prompt_tokens.size());
            const std::chrono::steady_clock::time_point prefill_begin = std::chrono::steady_clock::now();
            const float *logits = model.prefill(prompt_tokens.data(), prompt_tokens.size());
            const std::chrono::steady_clock::time_point prefill_end = std::chrono::steady_clock::now();
            std::cout << formatted_prompt << std::flush;
            const int32_t eos = int32_t(file.integer("tokenizer.ggml.eos_token_id"));
            std::array<char, 8192> decoded;
            size_t decoded_tokens = 0;
            size_t decode_evaluations = 0;
            double decode_seconds = 0.0;
            for (size_t i = 0; i < generation_limit; ++i) {
                const int32_t next = model.greedy(logits);
                if (next == eos) {
                    break;
                }
                ++decoded_tokens;
                const size_t decoded_size = tokenizer.decode_to(next, decoded.data(), decoded.size(), true);
                std::cout.write(decoded.data(), std::streamsize(decoded_size));
                std::cout << std::flush;
                if (i + 1 < generation_limit) {
                    const std::chrono::steady_clock::time_point decode_begin = std::chrono::steady_clock::now();
                    logits = model.forward(next, prompt_tokens.size() + i);
                    const std::chrono::steady_clock::time_point decode_end = std::chrono::steady_clock::now();
                    decode_seconds += elapsed_seconds(decode_begin, decode_end);
                    ++decode_evaluations;
                }
            }
            std::cout << '\n' << std::flush;
            const double prefill_seconds = elapsed_seconds(prefill_begin, prefill_end);
            std::fprintf(stderr, "prefill: %zu tokens, %.3f s, %.3f t/s\n", prompt_tokens.size(), prefill_seconds,
                         double(prompt_tokens.size()) / prefill_seconds);
            std::fprintf(stderr, "generated: %zu tokens\n", decoded_tokens);
            if (decode_evaluations > 0) {
                std::fprintf(stderr, "decode: %zu evals, %.3f s, %.3f t/s\n", decode_evaluations, decode_seconds,
                             double(decode_evaluations) / decode_seconds);
            } else {
                std::fprintf(stderr, "decode: 0 evals, 0.000 s, n/a t/s\n");
            }
            return 0;
        }
    } catch (const std::exception &error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
