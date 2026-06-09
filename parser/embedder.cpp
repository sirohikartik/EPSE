#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdint.h>
#include <torch/script.h>
#include <unordered_map>
#include <vector>
static torch::jit::script::Module model;

static const std::unordered_map<char, int64_t> TOKEN_MAP = {
    {'L', 4},  {'A', 5},  {'G', 6},  {'V', 7},  {'S', 8},  {'E', 9},  {'R', 10},
    {'T', 11}, {'I', 12}, {'D', 13}, {'P', 14}, {'K', 15}, {'Q', 16}, {'N', 17},
    {'F', 18}, {'Y', 19}, {'M', 20}, {'H', 21}, {'W', 22}, {'C', 23}, {'X', 24},
    {'B', 25}, {'U', 26}, {'Z', 27}, {'O', 28}, {'.', 29}, {'-', 30}};

std::vector<int64_t> tokenize(const char *sequence) {

    std::vector<int64_t> tokenized;
    tokenized.reserve(strlen(sequence) + 2);
    size_t i = 0;
    tokenized.push_back(0); // <cls>
    while (sequence[i] != '\0') {
        auto it = TOKEN_MAP.find(sequence[i]);

        if (it != TOKEN_MAP.end())
            tokenized.push_back(it->second);
        else
            tokenized.push_back(3); // <unk>

        i++;
    }
    tokenized.push_back(2); // <eos>
    return tokenized;
}

extern "C" int load_model(const char *model_path) {
    try {
        model = torch::jit::load(model_path);
        model.eval();
        std::cout << "Model loaded successfully\n";

        return 0;
    } catch (const c10::Error &e) {
        std::cerr << "Failed to load model: " << e.what() << '\n';

        return -1;
    }
}

extern "C" void get_embedding(const char *sequence, float *embedding_out) {

    std::vector<int64_t> tokens = tokenize(sequence);

    auto input_ids = torch::tensor(tokens, torch::kInt64).unsqueeze(0);

    auto attention_mask = torch::ones_like(input_ids);

    std::vector<torch::jit::IValue> inputs;

    inputs.push_back(input_ids);
    inputs.push_back(attention_mask);

    torch::NoGradGuard no_grad;

    auto output = model.forward(inputs);

    auto hidden = output.toTensor();

    auto pooled = hidden.mean(1);

    auto flat = pooled.squeeze(0).contiguous();

    auto accessor = flat.accessor<float, 1>();
    int dim = flat.size(0);

    for (int i = 0; i < dim; i++) {
        embedding_out[i] = accessor[i];
    }
}

extern "C" void close_model() { model = torch::jit::script::Module(); }


