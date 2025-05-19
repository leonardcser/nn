#include <emscripten.h>
#include "../core/nn.h"
#include <cstddef>
#include <vector> // For casting and std::vector type

extern "C" {

EMSCRIPTEN_KEEPALIVE
void nn_seed(unsigned int seed_val) {
    nn::seed(seed_val);
}

EMSCRIPTEN_KEEPALIVE
bool nn_create_model(nn::Model** model_ptr, size_t input_dim, size_t hidden_dim, size_t output_dim) {
    bool success = nn::create_model(model_ptr, input_dim, hidden_dim, output_dim);
    if (success && *model_ptr) {
        nn::initialize_model_weights(*model_ptr, -0.5, 0.5);
        return true;
    }
    return false;
}

EMSCRIPTEN_KEEPALIVE
void nn_train(nn::Model* model, const nn::TrainConfig* config, const nn::Dataset* training_data) {
    nn::train(model, config, training_data);
}

EMSCRIPTEN_KEEPALIVE
float nn_step(nn::Model* model, const nn::TrainConfig* config, const nn::Dataset* training_data, size_t batch_idx) {
    return nn::step(model, config, training_data, batch_idx);
}

EMSCRIPTEN_KEEPALIVE
void nn_infer(nn::Model* model, const float* input, float* output) {
    nn::infer(model, input, output);
}

EMSCRIPTEN_KEEPALIVE
void nn_free_model(nn::Model* model) {
    nn::free_model(model);
}

EMSCRIPTEN_KEEPALIVE
void nn_free_dataset(nn::Dataset* dataset) {
    nn::free_dataset(dataset);
}

} // extern "C"
