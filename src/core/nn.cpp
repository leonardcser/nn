#include "nn.h"
#include <cstddef>
#include <cstdlib>
#include <vector> 
#include <numeric>
#include <algorithm> 
#include <new> // Required for std::nothrow

namespace nn {

void seed(unsigned int seed_val) {
    srand(seed_val);
}


bool create_model(Model** model_ptr, size_t input_dim, size_t hidden_dim, size_t output_dim) {
    if (!model_ptr) return false;

    *model_ptr = (Model*)malloc(sizeof(Model));
    if (!*model_ptr) return false;

    (*model_ptr)->input_dim = input_dim;
    (*model_ptr)->hidden_dim = hidden_dim;
    (*model_ptr)->output_dim = output_dim;

    // Allocate std::vector<float> on the heap and store its pointer in _losses_history
    (*model_ptr)->_losses_history = new (std::nothrow) std::vector<float>();
    if (!(*model_ptr)->_losses_history) { // Check if allocation failed
        free(*model_ptr);
        *model_ptr = nullptr;
        return false;
    }
    // Initialize losses_history pointer to nullptr, it will point to _losses_history.data() later
    (*model_ptr)->losses_history = nullptr; 

    (*model_ptr)->weights_input_hidden = (float*)malloc(input_dim * hidden_dim * sizeof(float));
    (*model_ptr)->biases_hidden = (float*)malloc(hidden_dim * sizeof(float));
    (*model_ptr)->weights_hidden_output = (float*)malloc(hidden_dim * output_dim * sizeof(float));
    (*model_ptr)->biases_output = (float*)malloc(output_dim * sizeof(float));
    (*model_ptr)->hidden_layer_output = (float*)malloc(hidden_dim * sizeof(float));
    (*model_ptr)->final_output = (float*)malloc(output_dim * sizeof(float));

    if (!(*model_ptr)->weights_input_hidden || !(*model_ptr)->biases_hidden || 
        !(*model_ptr)->weights_hidden_output || !(*model_ptr)->biases_output || 
        !(*model_ptr)->hidden_layer_output || !(*model_ptr)->final_output) {
        
        free((*model_ptr)->weights_input_hidden);
        free((*model_ptr)->biases_hidden);
        free((*model_ptr)->weights_hidden_output);
        free((*model_ptr)->biases_output);
        free((*model_ptr)->hidden_layer_output);
        free((*model_ptr)->final_output);
        // Delete the heap-allocated std::vector<float>
        delete static_cast<std::vector<float>*>((*model_ptr)->_losses_history);
        free(*model_ptr);
        *model_ptr = nullptr;
        return false;
    }
    return true;
}

void initialize_model_weights(Model* model, float min_val, float max_val) {
    if (!model) return;

    size_t input_dim = model->input_dim;
    size_t hidden_dim = model->hidden_dim;
    size_t output_dim = model->output_dim;

    float range = max_val - min_val;

    // Initialize weights_input_hidden
    for (size_t i = 0; i < input_dim * hidden_dim; ++i) {
        model->weights_input_hidden[i] = min_val + (range * rand()) / (float)RAND_MAX;
    }
    // Initialize biases_hidden
    for (size_t i = 0; i < hidden_dim; ++i) {
        model->biases_hidden[i] = min_val + (range * rand()) / (float)RAND_MAX;
    }
    // Initialize weights_hidden_output
    for (size_t i = 0; i < hidden_dim * output_dim; ++i) {
        model->weights_hidden_output[i] = min_val + (range * rand()) / (float)RAND_MAX;
    }
    // Initialize biases_output
    for (size_t i = 0; i < output_dim; ++i) {
        model->biases_output[i] = min_val + (range * rand()) / (float)RAND_MAX;
    }
}

float step(Model* model, const TrainConfig* config, const Dataset* training_data, size_t batch_idx) {
    if (!model || !config || !training_data || training_data->num_samples == 0 || config->batch_size == 0) {
        return 0.0f; // Or some indicator of error/no-op
    }

    // Gradient accumulators
    std::vector<float> grad_weights_input_hidden(model->input_dim * model->hidden_dim);
    std::vector<float> grad_biases_hidden(model->hidden_dim);
    std::vector<float> grad_weights_hidden_output(model->hidden_dim * model->output_dim);
    std::vector<float> grad_biases_output(model->output_dim);

    // Temporary storage for error signals (deltas)
    std::vector<float> delta_output_layer(model->output_dim);
    std::vector<float> delta_hidden_layer(model->hidden_dim);

    // Initialize batch gradients to zero
    std::fill(grad_weights_input_hidden.begin(), grad_weights_input_hidden.end(), 0.0f);
    std::fill(grad_biases_hidden.begin(), grad_biases_hidden.end(), 0.0f);
    std::fill(grad_weights_hidden_output.begin(), grad_weights_hidden_output.end(), 0.0f);
    std::fill(grad_biases_output.begin(), grad_biases_output.end(), 0.0f);

    size_t start_sample_idx = batch_idx * config->batch_size;
    size_t current_batch_size = 0;
    float total_batch_squared_error = 0.0f;

    for (size_t sample_in_batch_idx = 0; sample_in_batch_idx < config->batch_size; ++sample_in_batch_idx) {
        size_t actual_sample_idx = start_sample_idx + sample_in_batch_idx;
        if (actual_sample_idx >= training_data->num_samples) {
            break; // Reached end of dataset
        }
        current_batch_size++;

        const float* current_input = &training_data->X_samples[actual_sample_idx * model->input_dim];
        const float* current_target = &training_data->Y_targets[actual_sample_idx * model->output_dim];

        // 1. Forward Pass
        // Hidden layer calculation (linear activation)
        for (size_t h_node = 0; h_node < model->hidden_dim; ++h_node) {
            model->hidden_layer_output[h_node] = model->biases_hidden[h_node];
            for (size_t i_node = 0; i_node < model->input_dim; ++i_node) {
                model->hidden_layer_output[h_node] += current_input[i_node] * model->weights_input_hidden[i_node * model->hidden_dim + h_node];
            }
        }

        // Output layer calculation (linear activation)
        for (size_t o_node = 0; o_node < model->output_dim; ++o_node) {
            model->final_output[o_node] = model->biases_output[o_node];
            for (size_t h_node = 0; h_node < model->hidden_dim; ++h_node) {
                model->final_output[o_node] += model->hidden_layer_output[h_node] * model->weights_hidden_output[h_node * model->output_dim + o_node];
            }
        }

        // 2. Backward Pass (Calculate error signals and accumulate gradients)
        // Calculate delta for output layer (prediction - target for MSE with linear activation)
        float sample_squared_error = 0.0f;
        for (size_t o_node = 0; o_node < model->output_dim; ++o_node) {
            float error = model->final_output[o_node] - current_target[o_node];
            delta_output_layer[o_node] = error;
            sample_squared_error += error * error;
        }
        total_batch_squared_error += sample_squared_error / model->output_dim; // Averaging over output dimensions for this sample

        // Calculate delta for hidden layer
        for (size_t h_node = 0; h_node < model->hidden_dim; ++h_node) {
            delta_hidden_layer[h_node] = 0.0f;
            for (size_t o_node = 0; o_node < model->output_dim; ++o_node) {
                delta_hidden_layer[h_node] += delta_output_layer[o_node] * model->weights_hidden_output[h_node * model->output_dim + o_node];
            }
            // Assuming linear activation for hidden layer, derivative is 1.
        }

        // Accumulate gradients for weights_hidden_output and biases_output
        for (size_t h_node = 0; h_node < model->hidden_dim; ++h_node) {
            for (size_t o_node = 0; o_node < model->output_dim; ++o_node) {
                grad_weights_hidden_output[h_node * model->output_dim + o_node] += delta_output_layer[o_node] * model->hidden_layer_output[h_node];
            }
        }
        for (size_t o_node = 0; o_node < model->output_dim; ++o_node) {
            grad_biases_output[o_node] += delta_output_layer[o_node];
        }

        // Accumulate gradients for weights_input_hidden and biases_hidden
        for (size_t i_node = 0; i_node < model->input_dim; ++i_node) {
            for (size_t h_node = 0; h_node < model->hidden_dim; ++h_node) {
                grad_weights_input_hidden[i_node * model->hidden_dim + h_node] += delta_hidden_layer[h_node] * current_input[i_node];
            }
        }
        for (size_t h_node = 0; h_node < model->hidden_dim; ++h_node) {
            grad_biases_hidden[h_node] += delta_hidden_layer[h_node];
        }
    } // End loop over samples in batch

    if (current_batch_size == 0) return 0.0f; // Skip if batch was empty, return 0 loss

    // 3. Update Weights (after processing all samples in the batch)
    float inv_batch_size = 1.0f / current_batch_size;

    // Update weights_hidden_output and biases_output
    for (size_t h_node = 0; h_node < model->hidden_dim; ++h_node) {
        for (size_t o_node = 0; o_node < model->output_dim; ++o_node) {
            model->weights_hidden_output[h_node * model->output_dim + o_node] -= config->learning_rate * grad_weights_hidden_output[h_node * model->output_dim + o_node] * inv_batch_size;
        }
    }
    for (size_t o_node = 0; o_node < model->output_dim; ++o_node) {
        model->biases_output[o_node] -= config->learning_rate * grad_biases_output[o_node] * inv_batch_size;
    }

    // Update weights_input_hidden and biases_hidden
    for (size_t i_node = 0; i_node < model->input_dim; ++i_node) {
        for (size_t h_node = 0; h_node < model->hidden_dim; ++h_node) {
            model->weights_input_hidden[i_node * model->hidden_dim + h_node] -= config->learning_rate * grad_weights_input_hidden[i_node * model->hidden_dim + h_node] * inv_batch_size;
        }
    }
    for (size_t h_node = 0; h_node < model->hidden_dim; ++h_node) {
        model->biases_hidden[h_node] -= config->learning_rate * grad_biases_hidden[h_node] * inv_batch_size;
    }

    float batch_mse_loss = total_batch_squared_error / current_batch_size;

    // Cast _losses_history to std::vector<float>* and store loss
    std::vector<float>* losses_vec = static_cast<std::vector<float>*>(model->_losses_history);
    losses_vec->push_back(batch_mse_loss);
    // Update the public-facing pointer to the vector's data
    model->losses_history = losses_vec->data();

    return batch_mse_loss;
}

void train(Model* model, const TrainConfig* config, const Dataset* training_data) {
    if (!model || !config || !training_data || training_data->num_samples == 0 || config->batch_size == 0 || config->epochs == 0) {
        return;
    }

    for (size_t epoch = 0; epoch < config->epochs; ++epoch) {
        size_t num_batches = (training_data->num_samples + config->batch_size - 1) / config->batch_size;
        for (size_t batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
            step(model, config, training_data, batch_idx);
        }
    }
}

void infer(Model* model, const float* input, float* output) {
    if (!model || !input || !output) return;
    // Placeholder for inference logic
    // 1. Calculate hidden layer output: hidden = activation(input * W_ih + b_h)
    //    (Store in model->hidden_layer_output)
    // 2. Calculate final output: final = activation(hidden * W_ho + b_o)
    //    (Store in model->final_output)
    // 3. Copy model->final_output to output buffer
}

void free_model(Model* model) {
    if (!model) return;
    free(model->weights_input_hidden);
    free(model->biases_hidden);
    free(model->weights_hidden_output);
    free(model->biases_output);
    free(model->hidden_layer_output);
    free(model->final_output);
    
    // Delete the heap-allocated std::vector<float> pointed to by _losses_history
    delete static_cast<std::vector<float>*>(model->_losses_history);
    // model->losses_history (the float*) does not need to be freed as it points to _losses_history data()

    free(model);
    model = nullptr;
}

void free_dataset(Dataset* dataset) {
    if (!dataset) return;
    free(dataset->X_samples);
    free(dataset->Y_targets);
    dataset = nullptr;
}

} // namespace nn 