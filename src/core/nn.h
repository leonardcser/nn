#pragma once

#include <cstddef>
#include <vector>

namespace nn {

struct __attribute__((packed)) Model {
    size_t input_dim;
    size_t hidden_dim;
    size_t output_dim;

    float* weights_input_hidden;
    float* biases_hidden;
    float* weights_hidden_output;
    float* biases_output;

    float* hidden_layer_output;
    float* final_output;

    float* losses_history; // Pointer to _losses_history.data()
    void* _losses_history; // Pointer to std::vector<float>
};

struct __attribute__((packed)) TrainConfig {
    float learning_rate;
    size_t epochs;
    size_t batch_size;
};

struct __attribute__((packed)) Dataset {
    size_t num_samples;
    size_t input_dim_per_sample;
    size_t output_dim_per_sample;

    float* X_samples;
    float* Y_targets;
};

void seed(unsigned int seed_val);
bool create_model(Model** model_ptr, size_t input_dim, size_t hidden_dim, size_t output_dim);
void initialize_model_weights(Model* model, float min_val, float max_val);
void train(Model* model, const TrainConfig* config, const Dataset* training_data);
float step(Model* model, const TrainConfig* config, const Dataset* training_data, size_t batch_idx);
void infer(Model* model, const float* input, float* output);
void free_model(Model* model);
void free_dataset(Dataset* dataset);

} // namespace nn 