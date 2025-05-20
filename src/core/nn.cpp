#include "nn.h"
#include "../wasm/utility.cpp"

#include <vector>
#include <numeric>      // For std::iota
#include <algorithm>    // For std::shuffle, std::fill, std::transform
#include <random>       // For std::default_random_engine, std::uniform_real_distribution, std::normal_distribution
#include <cstdio>       // For printf, fprintf
#include <stdexcept>    // For std::runtime_error
#include <cmath>        // For std::sqrt, std::exp, std::tanh, std::max, std::fabs

// Helper for random number generation
static std::default_random_engine random_generator;

namespace nn {

void seed_random_generator(unsigned int seed_val) {
    random_generator.seed(seed_val);
}

//------------------------------------------------------------------------------
// Activation Functions & Derivatives
//------------------------------------------------------------------------------

void activation_sigmoid(const float* z, float* a, int size) {
    for (int i = 0; i < size; ++i) {
        a[i] = 1.0f / (1.0f + std::exp(-z[i]));
    }
}

void derivative_sigmoid(const float* z __attribute__((unused)), const float* a, float* d, int size) {
    // d_sigmoid = a * (1 - a)
    for (int i = 0; i < size; ++i) {
        d[i] = a[i] * (1.0f - a[i]);
    }
}

void activation_relu(const float* z, float* a, int size) {
    for (int i = 0; i < size; ++i) {
        a[i] = std::max(0.0f, z[i]);
    }
}

void derivative_relu(const float* z, const float* a __attribute__((unused)), float* d, int size) {
    // d_relu = 1 if z > 0, 0 otherwise
    for (int i = 0; i < size; ++i) {
        d[i] = (z[i] > 0.0f) ? 1.0f : 0.0f;
    }
}

void activation_tanh(const float* z, float* a, int size) {
    for (int i = 0; i < size; ++i) {
        a[i] = std::tanh(z[i]);
    }
}

void derivative_tanh(const float* z __attribute__((unused)), const float* a, float* d, int size) {
    // d_tanh = 1 - a^2
    for (int i = 0; i < size; ++i) {
        d[i] = 1.0f - (a[i] * a[i]);
    }
}

//------------------------------------------------------------------------------
// Loss Functions & Derivatives
//------------------------------------------------------------------------------

float loss_mean_squared_error(const float* predictions, const float* targets, int size) {
    float sum_squared_error = 0.0f;
    for (int i = 0; i < size; ++i) {
        float error = predictions[i] - targets[i];
        sum_squared_error += error * error;
    }
    return sum_squared_error / size; // Mean
}

void derivative_mean_squared_error(const float* predictions, const float* targets, float* out_delta, int size) {
    // dL/da = 2 * (predictions - targets) / size for MSE
    // However, often the 2/size is combined with learning rate or handled elsewhere.
    // Here, we'll compute (predictions - targets), as this is commonly dL/da_last_layer for MSE backprop.
    // The factor of 2 can be absorbed into the learning rate.
    // The division by N (batch size) is typically done when updating weights.
    // For simplicity, let's compute prediction - target. If a different scaling is needed, it can be adjusted.
    for (int i = 0; i < size; ++i) {
        out_delta[i] = predictions[i] - targets[i];
    }
}

//------------------------------------------------------------------------------
// Layer-specific operations (Implementations)
//------------------------------------------------------------------------------

// --- Dense Layer --- 
void dense_forward_pass(Layer* self, const float* input_data, float* output_data) {
    // output = weights * input + biases
    // Weights: output_size x input_size
    // Input: input_size x 1
    // Output: output_size x 1
    // Biases: output_size x 1

    for (int i = 0; i < self->output_size; ++i) {
        output_data[i] = self->biases[i]; // Initialize with bias
        for (int j = 0; j < self->input_size; ++j) {
            output_data[i] += self->weights[i * self->input_size + j] * input_data[j];
        }
    }
    // Store for backward pass: The input data is already pointed to by self->last_input_data
    // The output of this dense layer (z_values for a following activation) could be stored if needed,
    // but often the activation layer's forward pass immediately uses it. 
    // Here we assume self->last_output_activations (or a similar field like last_z_values if we add it)
    // will be set by the code managing layer-to-layer data flow IF this dense layer is immediately followed by an activation.
    // For now, this function just calculates output_data.
}

void dense_backward_pass(Layer* self, const float* upstream_delta, const float* last_input_data_for_this_layer, float* computed_delta_for_prev_layer) {
    // upstream_delta (dL/dZ_current or dL/dA_current if no activation immediately after this dense layer) has shape (output_size, 1)
    // last_input_data_for_this_layer (A_prev) has shape (input_size, 1)
    // weights (W) has shape (output_size, input_size)

    // 1. Calculate dL/dW = upstream_delta * last_input_data_for_this_layer^T
    // dW[i][j] = upstream_delta[i] * last_input_data_for_this_layer[j]
    for (int i = 0; i < self->output_size; ++i) {
        for (int j = 0; j < self->input_size; ++j) {
            self->_dW[i * self->input_size + j] += upstream_delta[i] * last_input_data_for_this_layer[j];
        }
    }

    // 2. Calculate dL/db = upstream_delta
    for (int i = 0; i < self->output_size; ++i) {
        self->_db[i] += upstream_delta[i];
    }

    // 3. Calculate dL/dA_prev = W^T * upstream_delta
    // computed_delta_for_prev_layer has shape (input_size, 1)
    if (computed_delta_for_prev_layer) { // Null if this is the first layer
        for (int j = 0; j < self->input_size; ++j) {
            float sum = 0.0f;
            for (int i = 0; i < self->output_size; ++i) {
                sum += self->weights[i * self->input_size + j] * upstream_delta[i]; // W_ij transposed is W_ji
            }
            computed_delta_for_prev_layer[j] = sum;
        }
    }
}

void dense_update_weights(Layer* self, float learning_rate, int batch_size) {
    // Update weights: W = W - learning_rate * (dL/dW / batch_size)
    float scale = learning_rate / static_cast<float>(batch_size);
    for (int i = 0; i < self->output_size * self->input_size; ++i) {
        self->weights[i] -= scale * self->_dW[i];
    }

    // Update biases: b = b - learning_rate * (dL/db / batch_size)
    for (int i = 0; i < self->output_size; ++i) {
        self->biases[i] -= scale * self->_db[i];
    }
}

void dense_zero_gradients(Layer* self) {
    std::fill(self->_dW, self->_dW + self->output_size * self->input_size, 0.0f);
    std::fill(self->_db, self->_db + self->output_size, 0.0f);
}

// --- Activation Layer ---
void activation_forward_pass(Layer* self, const float* input_data, float* output_data) {
    // input_data here are the Z values from the previous (typically Dense) layer.
    // output_data are the activations A.
    self->_activate_func(input_data, output_data, self->input_size); // input_size == output_size for activation
    
    // Store necessary values for backward pass
    // self->_last_input_data is already set to point to the Z values by the model orchestration.
    // self->_last_output_activations will be set to point to output_data by model orchestration.
    // Activation layers often need both Z (input to activation) and A (output of activation) for derivative calculation.
    // We assume self->_last_input_data (which is Z) and self->_last_output_activations (which is A) are correctly managed outside.
}

void activation_backward_pass(Layer* self, const float* upstream_delta, const float* last_input_data_for_this_layer, float* computed_delta_for_prev_layer) {
    // upstream_delta (dL/dA_current) has shape (output_size, 1)
    // last_input_data_for_this_layer (Z_current) has shape (input_size, 1) which is same as output_size
    // We also need A_current (_last_output_activations) for some derivatives like sigmoid, tanh.
    // self->_last_output_activations should point to A_current.

    // 1. Calculate dAdZ = _activate_derivative_func(Z_current, A_current)
    // This temporary buffer would ideally be part of the layer's or model's pre-allocated buffers.
    // For now, let's allocate it on the stack if small, or heap if large, or better, use a model-level buffer.
    // Let's assume computed_delta_for_prev_layer can be re-used or is large enough temporarily if not first layer.
    // The output `computed_delta_for_prev_layer` will be dL/dZ_current.

    if (computed_delta_for_prev_layer) { // Should always be true unless it's an output activation directly connected to loss, which is rare.
        // We need A_current (this layer's output from forward pass) for some derivatives.
        // The `self->_last_output_activations` should point to it.
        if (!self->_last_output_activations) {
             // This is an issue. For now, let's recompute if not available, but ideally it's stored.
             // This implies a design where last_output_activations MUST be set correctly.
             // For this simplified version, we'll proceed assuming the derivative function can handle it
             // or that `last_input_data_for_this_layer` (Z) is enough (e.g. for ReLU)
             // For Sigmoid/Tanh: `self->_activate_derivative_func` needs `a` (activations)
             // This highlights the need for careful buffer management in `model_forward_pass`.
        }

        // Let's use a temporary buffer for dAdZ for clarity, assuming its size is self->output_size.
        // In a real scenario, this would be a pre-allocated buffer.
        float* dAdZ = new float[self->output_size]; 
        self->_activate_derivative_func(last_input_data_for_this_layer /* z */, 
                                       self->_last_output_activations /* a */, 
                                       dAdZ, 
                                       self->output_size);

        // 2. Calculate dL/dZ_current = upstream_delta * dAdZ (element-wise)
        for (int i = 0; i < self->output_size; ++i) {
            computed_delta_for_prev_layer[i] = upstream_delta[i] * dAdZ[i];
        }
        delete[] dAdZ;
    }
}
// Activation layers don't have weights to update or gradients to zero in the same way dense layers do.
// Their function pointers for update_weights and zero_gradients will typically be NULL.
// So, no dense_update_weights or dense_zero_gradients equivalent here.

//------------------------------------------------------------------------------
// Model Creation and Memory Management
//------------------------------------------------------------------------------
Model* create_model(int model_input_dim, int num_model_layers, const LayerType* layer_def_types, const int* layer_def_params) {
    if (num_model_layers <= 0) {
        throw std::runtime_error("Error: Number of layers must be positive.");
    }
    if (!layer_def_types || !layer_def_params) {
        throw std::runtime_error("Error: Layer definitions arrays must not be null.");
    }

    Model* model = new Model();
    model->num_layers = num_model_layers;
    model->_layers = new Layer[num_model_layers];
    model->model_input_dim = model_input_dim;
    model->total_trainable_params = 0;

    int current_input_size = model_input_dim;
    model->_max_intermediate_io_size = model_input_dim; 

    for (int i = 0; i < num_model_layers; ++i) {
        Layer* current_layer = &model->_layers[i];
        LayerType type = layer_def_types[i];

        current_layer->type = type;
        current_layer->input_size = current_input_size;
        
        current_layer->weights = nullptr;
        current_layer->biases = nullptr;
        current_layer->_dW = nullptr;
        current_layer->_db = nullptr;
        current_layer->_activate_func = nullptr;
        current_layer->_activate_derivative_func = nullptr;
        current_layer->_forward_pass = nullptr;
        current_layer->_backward_pass = nullptr;
        current_layer->_update_weights = nullptr;
        current_layer->_zero_gradients = nullptr;
        current_layer->_last_input_data = nullptr;
        current_layer->_last_output_activations = nullptr;

        if (type == LAYER_TYPE_DENSE) {
            current_layer->output_size = layer_def_params[i];
            if (current_layer->output_size <= 0) {
                for(int k_cleanup = 0; k_cleanup <= i; ++k_cleanup) {
                    Layer* l_to_clean = &model->_layers[k_cleanup];
                    delete[] l_to_clean->weights; 
                    delete[] l_to_clean->biases;
                    delete[] l_to_clean->_dW;     
                    delete[] l_to_clean->_db;
                }
                delete[] model->_layers;
                delete model;
                throw std::runtime_error("Error: Dense layer output size must be positive.");
            }

            int weight_count = current_layer->input_size * current_layer->output_size;
            current_layer->weights = new float[weight_count];
            current_layer->biases = new float[current_layer->output_size];
            current_layer->_dW = new float[weight_count];
            current_layer->_db = new float[current_layer->output_size];
            
            model->total_trainable_params += weight_count + current_layer->output_size;

            current_layer->_forward_pass = dense_forward_pass;
            current_layer->_backward_pass = dense_backward_pass;
            current_layer->_update_weights = dense_update_weights;
            current_layer->_zero_gradients = dense_zero_gradients;

        } else if (type == LAYER_TYPE_ACTIVATION_SIGMOID) {
            current_layer->output_size = current_input_size;
            current_layer->_activate_func = activation_sigmoid;
            current_layer->_activate_derivative_func = derivative_sigmoid;
            current_layer->_forward_pass = activation_forward_pass;
            current_layer->_backward_pass = activation_backward_pass;
        } else if (type == LAYER_TYPE_ACTIVATION_RELU) {
            current_layer->output_size = current_input_size;
            current_layer->_activate_func = activation_relu;
            current_layer->_activate_derivative_func = derivative_relu;
            current_layer->_forward_pass = activation_forward_pass;
            current_layer->_backward_pass = activation_backward_pass;
        } else if (type == LAYER_TYPE_ACTIVATION_TANH) {
            current_layer->output_size = current_input_size;
            current_layer->_activate_func = activation_tanh;
            current_layer->_activate_derivative_func = derivative_tanh;
            current_layer->_forward_pass = activation_forward_pass;
            current_layer->_backward_pass = activation_backward_pass;
        } else {
            for(int k_cleanup = 0; k_cleanup <= i; ++k_cleanup) {
                Layer* l_to_clean = &model->_layers[k_cleanup];
                delete[] l_to_clean->weights; delete[] l_to_clean->biases;
                delete[] l_to_clean->_dW; delete[] l_to_clean->_db;
            }
            delete[] model->_layers;
            delete model;
            throw std::runtime_error("Error: Unknown layer type encountered in create_model.");
        }

        current_input_size = current_layer->output_size;
        if (current_layer->output_size > model->_max_intermediate_io_size) {
            model->_max_intermediate_io_size = current_layer->output_size;
        }
    }

    model->model_output_dim = current_input_size; // Output dim of the last layer is the model's output dim

    // Allocate ping-pong buffers based on _max_intermediate_io_size
    // Ensure _max_intermediate_io_size is at least model_input_dim and model_output_dim for consistency, though
    // model input and output might be handled with separate user-provided buffers during forward/predict.
    // The internal buffers are for layer-to-layer data transfer.
    if (model->model_input_dim > model->_max_intermediate_io_size) model->_max_intermediate_io_size = model->model_input_dim;
    if (model->model_output_dim > model->_max_intermediate_io_size) model->_max_intermediate_io_size = model->model_output_dim;
    
    // Add a check for zero size allocation
    if (model->_max_intermediate_io_size == 0 && num_model_layers > 0) {
        // This case should ideally not happen if layers are configured correctly
        // fprintf(stderr, "Warning: _max_intermediate_io_size is 0. Buffers will not be allocated.\n");
        // Fallback or error handling, for now, let them be nullptr
        model->_forward_buffer1 = nullptr;
        model->_forward_buffer2 = nullptr;
        model->_backward_buffer1 = nullptr;
        model->_backward_buffer2 = nullptr;
        model->_final_output_buffer = nullptr;
    } else if (model->_max_intermediate_io_size > 0) {
        model->_forward_buffer1 = new float[model->_max_intermediate_io_size];
        model->_forward_buffer2 = new float[model->_max_intermediate_io_size];
        model->_backward_buffer1 = new float[model->_max_intermediate_io_size];
        model->_backward_buffer2 = new float[model->_max_intermediate_io_size];
        // _final_output_buffer should be model_output_dim
        if (model->model_output_dim > 0) {
             model->_final_output_buffer = new float[model->model_output_dim];
        } else {
             model->_final_output_buffer = nullptr; // Or handle error if output_dim is 0 for a valid model
        }
    } else { // _max_intermediate_io_size is 0, likely no layers or error
        model->_forward_buffer1 = nullptr;
        model->_forward_buffer2 = nullptr;
        model->_backward_buffer1 = nullptr;
        model->_backward_buffer2 = nullptr;
        model->_final_output_buffer = nullptr;
    }

    // Initialize buffer pointers (can be refined later in forward/backward passes)
    model->_current_layer_input_ptr = nullptr;
    model->_current_layer_output_ptr = nullptr;
    model->_current_delta_input_ptr = nullptr;
    model->_current_delta_output_ptr = nullptr;

    return model;
}

void free_model(Model* model) {
    if (!model) return;

    for (int i = 0; i < model->num_layers; ++i) {
        Layer* layer = &model->_layers[i];
        delete[] layer->weights; // Ok if null
        delete[] layer->biases;  // Ok if null
        delete[] layer->_dW;      // Ok if null
        delete[] layer->_db;      // Ok if null
    }
    delete[] model->_layers;

    delete[] model->_forward_buffer1;
    delete[] model->_forward_buffer2;
    delete[] model->_backward_buffer1;
    delete[] model->_backward_buffer2;
    delete[] model->_final_output_buffer;

    delete model;
}

//------------------------------------------------------------------------------
// Weight Initialization
//------------------------------------------------------------------------------
void initialize_weights_xavier_uniform(Model* model) {
    if (!model) return;
    for (int i = 0; i < model->num_layers; ++i) {
        Layer* layer = &model->_layers[i];
        if (layer->type == LAYER_TYPE_DENSE) {
            float limit = std::sqrt(6.0f / (layer->input_size + layer->output_size));
            std::uniform_real_distribution<float> dist(-limit, limit);
            for (int j = 0; j < layer->input_size * layer->output_size; ++j) {
                layer->weights[j] = dist(random_generator);
            }
        }
    }
}

void initialize_weights_he_uniform(Model* model) {
    if (!model) return;
    for (int i = 0; i < model->num_layers; ++i) {
        Layer* layer = &model->_layers[i];
        if (layer->type == LAYER_TYPE_DENSE) {
            float limit = std::sqrt(6.0f / layer->input_size);
            std::uniform_real_distribution<float> dist(-limit, limit);
            for (int j = 0; j < layer->input_size * layer->output_size; ++j) {
                layer->weights[j] = dist(random_generator);
            }
        }
    }
}

void initialize_weights_uniform_range(Model* model, float min_val, float max_val) {
    if (!model) return;
    std::uniform_real_distribution<float> dist(min_val, max_val);
    for (int i = 0; i < model->num_layers; ++i) {
        Layer* layer = &model->_layers[i];
        if (layer->type == LAYER_TYPE_DENSE) {
            for (int j = 0; j < layer->input_size * layer->output_size; ++j) {
                layer->weights[j] = dist(random_generator);
            }
        }
    }
}

void initialize_biases_zero(Model* model) {
    if (!model) return;
    for (int i = 0; i < model->num_layers; ++i) {
        Layer* layer = &model->_layers[i];
        if (layer->type == LAYER_TYPE_DENSE && layer->biases) {
            std::fill(layer->biases, layer->biases + layer->output_size, 0.0f);
        }
    }
}

//------------------------------------------------------------------------------
// Dataset Management
//------------------------------------------------------------------------------
Dataset* create_dataset(
    int num_samples, int input_dim, int output_dim,
    const float* inputs_flat_data, // Model will copy this data
    const float* targets_flat_data // Model will copy this data
) {
    if (num_samples <= 0 || input_dim <= 0 || output_dim <= 0 || !inputs_flat_data || !targets_flat_data) {
        throw std::runtime_error("Error: Invalid arguments for create_dataset. All dimensions must be positive and data pointers non-null.");
    }

    Dataset* dataset = new Dataset();
    dataset->num_samples = num_samples;
    dataset->input_dim = input_dim;
    dataset->output_dim = output_dim;

    float* copied_inputs = new float[num_samples * input_dim];
    std::copy(inputs_flat_data, inputs_flat_data + num_samples * input_dim, copied_inputs);
    dataset->_inputs_flat = copied_inputs;

    float* copied_targets = new float[num_samples * output_dim];
    std::copy(targets_flat_data, targets_flat_data + num_samples * output_dim, copied_targets);
    dataset->_targets_flat = copied_targets;

    return dataset;
}

void free_dataset(Dataset* dataset) {
    if (!dataset) return;

    delete[] dataset->_inputs_flat;
    delete[] dataset->_targets_flat;
    delete dataset;
}

//------------------------------------------------------------------------------
// Core Model Operations
//------------------------------------------------------------------------------

void model_forward_pass(Model* model, const float* input_sample) {
    if (!model || !input_sample) return;
    if (model->num_layers == 0) return; // No layers to pass through

    // Use ping-pong buffers for inter-layer data transfer
    float* current_input = model->_forward_buffer1;
    float* current_output = model->_forward_buffer2;

    // Copy initial input_sample to the first buffer
    std::copy(input_sample, input_sample + model->model_input_dim, current_input);

    for (int i = 0; i < model->num_layers; ++i) {
        Layer* layer = &model->_layers[i];
        
        // Set _last_input_data for the layer (points to the data it's about to process)
        layer->_last_input_data = current_input; 

        layer->_forward_pass(layer, current_input, current_output);
        
        // Store the output of this layer for potential use in backpropagation (e.g., A for activation derivative)
        layer->_last_output_activations = current_output;

        // Swap buffers for the next iteration
        // The output of the current layer becomes the input for the next
        std::swap(current_input, current_output);
    }

    // After the loop, current_input holds the final output of the network
    // Copy it to model->_final_output_buffer
    if (model->_final_output_buffer) {
        std::copy(current_input, current_input + model->model_output_dim, model->_final_output_buffer);
    }
    // Update model's direct access pointers (though _final_output_buffer is the main one for external access)
    model->_current_layer_input_ptr = nullptr; // Reset, as pass is complete
    model->_current_layer_output_ptr = model->_final_output_buffer; // Points to the final result
}

void model_backward_pass(Model* model, const float* target_sample, const TrainConfig* config) {
    if (!model || !target_sample || !config || model->num_layers == 0) return;

    // 0. Ensure the forward pass for this input_sample_for_forward has been done
    // so that layer->_last_input_data and layer->_last_output_activations are populated.
    // (This function is typically called right after a forward pass with the same sample during training)

    // 1. Calculate initial delta (dL/dA_last_layer or dL/dZ_last_layer for combined loss/activation)
    // This delta is calculated w.r.t. the *output of the model*, which is model->_final_output_buffer.
    // The target_sample is the ground truth.
    // The size is model->model_output_dim.

    float* current_delta = model->_backward_buffer1; // Or directly use one of the buffers
    float* prev_layer_delta_output_buffer = model->_backward_buffer2;

    // Calculate dL/d(predictions) using the model's final output from the forward pass.
    // model->_final_output_buffer should contain the predictions for input_sample_for_forward.
    config->_loss_derivative_func(model->_final_output_buffer, target_sample, current_delta, model->model_output_dim);

    // 2. Propagate delta backwards through layers
    for (int i = model->num_layers - 1; i >= 0; --i) {
        Layer* layer = &model->_layers[i];
        
        // The `upstream_delta` for this layer is `current_delta`.
        // The `last_input_data_for_this_layer` is `layer->_last_input_data` (input it received during forward).
        // For activation layers, `layer->_last_z_values` (if used and different) or `layer->_last_input_data` (if Z) 
        // and `layer->_last_output_activations` (A) are needed.
        
        const float* last_input_for_this_layer_backprop; 
        if (layer->type == LAYER_TYPE_DENSE) {
            // Dense layer backward pass needs A_prev (which is layer->_last_input_data of *this* dense layer)
            last_input_for_this_layer_backprop = layer->_last_input_data;
        } else { // Activation Layer
            // Activation layer backward pass needs Z_current (which is layer->_last_input_data of *this* activation layer)
            // and A_current (which is layer->_last_output_activations of *this* activation layer)
            last_input_for_this_layer_backprop = layer->_last_input_data; // This is Z for activation layer
            // layer->_last_output_activations is A, used internally by activation_backward_pass
        }

        float* computed_delta_for_prev_layer_ptr = (i == 0) ? nullptr : prev_layer_delta_output_buffer;
        
        layer->_backward_pass(layer, current_delta, last_input_for_this_layer_backprop, computed_delta_for_prev_layer_ptr);

        // Swap delta buffers for next iteration (i.e., for the previous layer)
        std::swap(current_delta, prev_layer_delta_output_buffer);
    }
    // Gradients dW and db are now accumulated in each dense layer.
}

void model_update_weights(Model* model, float learning_rate, int batch_size) {
    if (!model || batch_size <= 0) return;
    for (int i = 0; i < model->num_layers; ++i) {
        Layer* layer = &model->_layers[i];
        if (layer->_update_weights) {
            layer->_update_weights(layer, learning_rate, batch_size);
        }
    }
}

void model_zero_gradients(Model* model) {
    if (!model) return;
    for (int i = 0; i < model->num_layers; ++i) {
        Layer* layer = &model->_layers[i];
        if (layer->_zero_gradients) {
            layer->_zero_gradients(layer);
        }
    }
}

//------------------------------------------------------------------------------
// Training
//------------------------------------------------------------------------------

// Performs a single training step on a given batch of data.
float step_model(
    Model* model,
    const float* batch_inputs,
    const float* batch_targets,
    int num_samples_in_batch,
    const TrainConfig* config
) {
    if (!model || !batch_inputs || !batch_targets || !config || num_samples_in_batch <= 0) {
        // Consider throwing an error or returning a specific value like -1.0f or NaN
        fprintf(stderr, "Error: Invalid arguments for step_model.\n");
        return -1.0f; // Or handle error appropriately
    }

    model_zero_gradients(model); // Zero gradients at the start of each batch step

    float batch_loss_sum = 0.0f;

    for (int i = 0; i < num_samples_in_batch; ++i) {
        const float* input_sample = batch_inputs + (i * model->model_input_dim);
        const float* target_sample = batch_targets + (i * model->model_output_dim);

        // Forward pass
        model_forward_pass(model, input_sample);
        // model->_final_output_buffer now contains the predictions

        // Calculate loss for this sample
        batch_loss_sum += config->_compute_loss_func(model->_final_output_buffer, target_sample, model->model_output_dim);

        // Backward pass (accumulates gradients)
        model_backward_pass(model, target_sample, config);
    }

    // Update weights based on accumulated gradients for the batch
    model_update_weights(model, config->learning_rate, num_samples_in_batch);
    
    return batch_loss_sum / static_cast<float>(num_samples_in_batch); // Average loss for the batch
}

void train_model(
    Model* model,
    const Dataset* train_data,
    TrainConfig* config,
    const Dataset* validation_data // Optional, can be NULL
) {
    if (!model || !train_data || !config) {
        throw std::runtime_error("Error: Invalid arguments for train_model. Model, train_data, and config must not be null.");
    }
    if (train_data->num_samples == 0) {
        throw std::runtime_error("Error: Training data has no samples.");
    }
    if (config->batch_size <= 0 ) { 
        throw std::runtime_error("Error: Batch size must be positive.");
    }

    // Ensure loss functions are set if not provided explicitly
    if (config->_compute_loss_func == nullptr) {
        if (config->loss_type == LOSS_MEAN_SQUARED_ERROR) {
            config->_compute_loss_func = loss_mean_squared_error;
        } else {
            // If other loss types are added, they should be handled here
            throw std::runtime_error("Error: _compute_loss_func is null and no default assignment for the given loss_type.");
        }
    }

    if (config->_loss_derivative_func == nullptr) {
        if (config->loss_type == LOSS_MEAN_SQUARED_ERROR) {
            config->_loss_derivative_func = derivative_mean_squared_error;
        } else {
            // If other loss types are added, they should be handled here
            throw std::runtime_error("Error: _loss_derivative_func is null and no default assignment for the given loss_type.");
        }
    }

    // Seed random generator for shuffling (if not already seeded externally for weights)
    // It's good practice to allow external seeding for reproducibility of the whole process.
    // If config->random_seed is 0, it might mean "don't re-seed here" or "use a time-based seed".
    // For now, let's assume if seed is non-zero, we use it for shuffling.
    if (config->random_seed != 0) { // Allow 0 to mean "use current random state"
        random_generator.seed(config->random_seed);
    }

    int num_batches = (train_data->num_samples + config->batch_size - 1) / config->batch_size; // Ceiling division

    std::vector<int> sample_indices(train_data->num_samples);
    std::iota(sample_indices.begin(), sample_indices.end(), 0); // Fill with 0, 1, ..., n-1

    // Temporary buffers for batch data if shuffling
    float* shuffled_batch_inputs = nullptr;
    float* shuffled_batch_targets = nullptr;
    if (config->shuffle_each_epoch) {
        // Allocate only once if batch size is consistent, or handle dynamic allocation if batch size can vary (though it's fixed in TrainConfig)
        // Ensure these buffers are large enough for the maximum batch size (config->batch_size)
        shuffled_batch_inputs = new float[config->batch_size * train_data->input_dim];
        shuffled_batch_targets = new float[config->batch_size * train_data->output_dim];
    }
    
    for (int epoch = 0; epoch < config->epochs; ++epoch) {

        float epoch_loss_sum = 0.0f;

        if (config->shuffle_each_epoch) { 
           std::shuffle(sample_indices.begin(), sample_indices.end(), random_generator);
        }

        for (int batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
            int start_sample_idx_in_dataset = batch_idx * config->batch_size;
            int num_samples_in_current_batch = 0;

            // Determine the actual number of samples in this batch (can be smaller for the last batch)
            if (start_sample_idx_in_dataset + config->batch_size <= train_data->num_samples) {
                num_samples_in_current_batch = config->batch_size;
            } else {
                num_samples_in_current_batch = train_data->num_samples - start_sample_idx_in_dataset;
            }

            if (num_samples_in_current_batch <= 0) {
                continue; // Should not happen if num_batches is calculated correctly
            }

            // Get pointers to the data for the current batch
            // If shuffling is implemented, use sample_indices[start_sample_idx_in_dataset + i] to get actual data
            const float* current_batch_inputs_ptr;
            const float* current_batch_targets_ptr;

            if (config->shuffle_each_epoch) {
                // Copy shuffled data to temporary batch buffers
                for (int k = 0; k < num_samples_in_current_batch; ++k) {
                    int original_sample_idx = sample_indices[start_sample_idx_in_dataset + k];
                    // Copy input features
                    std::copy(
                        train_data->_inputs_flat + (original_sample_idx * train_data->input_dim),
                        train_data->_inputs_flat + (original_sample_idx * train_data->input_dim) + train_data->input_dim,
                        shuffled_batch_inputs + (k * train_data->input_dim)
                    );
                    // Copy target features
                    std::copy(
                        train_data->_targets_flat + (original_sample_idx * train_data->output_dim),
                        train_data->_targets_flat + (original_sample_idx * train_data->output_dim) + train_data->output_dim,
                        shuffled_batch_targets + (k * train_data->output_dim)
                    );
                }
                current_batch_inputs_ptr = shuffled_batch_inputs;
                current_batch_targets_ptr = shuffled_batch_targets;
            } else {
                current_batch_inputs_ptr = train_data->_inputs_flat + (start_sample_idx_in_dataset * train_data->input_dim);
                current_batch_targets_ptr = train_data->_targets_flat + (start_sample_idx_in_dataset * train_data->output_dim);
            }

            float average_batch_loss = step_model(model, current_batch_inputs_ptr, current_batch_targets_ptr, num_samples_in_current_batch, config);

            epoch_loss_sum += average_batch_loss * num_samples_in_current_batch; // Accumulate total loss for the epoch

            if (config->print_progress_every_n_batches > 0 && (batch_idx + 1) % config->print_progress_every_n_batches == 0) {
                printf("Epoch [%d/%d], Batch [%d/%d], Avg Batch Loss: %f\n",
                          epoch + 1, config->epochs,
                          batch_idx + 1, num_batches,
                          average_batch_loss);
            }
        }

        float average_training_loss_for_epoch = (train_data->num_samples > 0) ? (epoch_loss_sum / train_data->num_samples) : 0.0f;
        
        float calculated_validation_loss = -1.0f; // Signify not computed
        if (validation_data && validation_data->num_samples > 0) {
            float val_loss_sum = 0.0f;
            for (int k = 0; k < validation_data->num_samples; ++k) {
                const float* val_input = validation_data->_inputs_flat + (k * validation_data->input_dim);
                const float* val_target = validation_data->_targets_flat + (k * validation_data->output_dim);
                // predict() internally calls model_forward_pass
                const float* val_prediction = predict(model, val_input);
                val_loss_sum += config->_compute_loss_func(val_prediction, val_target, model->model_output_dim);
            }
            calculated_validation_loss = (validation_data->num_samples > 0) ? (val_loss_sum / validation_data->num_samples) : -1.0f;
        }


        // Call the epoch completed callback if provided
        if (config->epoch_callback_func_id >= 0) {
            _callback(config->epoch_callback_func_id, (void*)&average_training_loss_for_epoch, sizeof(average_training_loss_for_epoch));
        }
    }

    // Clean up allocated batch buffers
    if (config->shuffle_each_epoch) {
        delete[] shuffled_batch_inputs;
        delete[] shuffled_batch_targets;
    }
}

//------------------------------------------------------------------------------
// Prediction
//------------------------------------------------------------------------------

const float* get_model_output(const Model* model) {
    if (!model) return nullptr;
    return model->_final_output_buffer;
}

const float* predict(Model* model, const float* input_sample) {
    if (!model || !input_sample) return nullptr;
    model_forward_pass(model, input_sample);
    return model->_final_output_buffer;
}

} // namespace nn
