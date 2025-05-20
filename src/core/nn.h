#pragma once

#include <cstddef>

// Forward declare linalg types if they were to be used in these headers directly
// For now, we assume linalg.h is included by .cpp files for math,
// and our ABI uses float* for dynamic data.
// #include "linalg.h"

namespace nn {

//------------------------------------------------------------------------------
// Enums
//------------------------------------------------------------------------------

enum LayerType {
    LAYER_TYPE_DENSE,
    LAYER_TYPE_ACTIVATION_SIGMOID,
    LAYER_TYPE_ACTIVATION_RELU,
    LAYER_TYPE_ACTIVATION_TANH,
    // Add more layer types as needed (e.g., Convolution, Pooling, Dropout)
};

enum LossFunctionType {
    LOSS_MEAN_SQUARED_ERROR,
    // Add more loss types as needed
};

//------------------------------------------------------------------------------
// Function Pointer Typedefs
//------------------------------------------------------------------------------

// Activation Functions:
// Input: z_values (vector before activation), Output: activations (vector after activation)
// Both vectors are of 'size'.
typedef void (*ActivationFunctionPtr)(const float* z_values, float* activations, int size);

// Activation Function Derivatives:
// Input: z_values (vector before activation), activations (vector after activation)
// Output: derivatives (d(activation)/dz)
// All vectors are of 'size'.
typedef void (*ActivationDerivativePtr)(const float* z_values, const float* activations, float* derivatives, int size);

// Loss Functions:
// Input: predictions (model output), targets (ground truth)
// Both vectors are of 'size' (model output dimension).
// Returns the computed loss value.
typedef float (*ComputeLossPtr)(const float* predictions, const float* targets, int size);

// Loss Function Derivatives:
// Input: predictions (model output), targets (ground truth)
// Output: out_delta (gradient of loss w.r.t. predictions, dL/da_last_layer)
// All vectors are of 'size' (model output dimension).
// For CrossEntropyWithSoftmax, this will be dL/dz_last_layer (predictions - targets).
typedef void (*LossDerivativePtr)(const float* predictions, const float* targets, float* out_delta, int size);

// Epoch Completed Callback:
// Called after each epoch during training.
// - epoch: The completed epoch number (1-indexed).
// - training_loss: Average training loss for the completed epoch.
// - validation_loss: Average validation loss for the completed epoch (-1.0f if not applicable).
// - user_data: Arbitrary user data pointer passed from TrainConfig.
typedef void (*EpochCompletedCallback)(int epoch, float training_loss, float validation_loss, void* user_data);

// Layer-specific operations (assigned during layer creation)
// `self` is a pointer to the Layer struct itself.
// `input_data` is the output from the previous layer (or model input).
// `output_data` is the buffer where this layer writes its output.
typedef void (*LayerForwardPassFunc)(struct Layer* self, const float* input_data, float* output_data);

// `upstream_delta` is the error gradient from the next layer (dL/da_current_layer).
// `computed_delta_for_prev_layer` is the error gradient this layer computes for the previous layer (dL/da_prev_layer).
// `last_input_data_for_this_layer` is the `input_data` that was fed to this layer during the corresponding forward pass.
typedef void (*LayerBackwardPassFunc)(struct Layer* self, const float* upstream_delta, const float* last_input_data_for_this_layer, float* computed_delta_for_prev_layer);

// `learning_rate` for updating weights.
// `batch_size` for averaging/scaling gradients if accumulated.
typedef void (*LayerUpdateWeightsFunc)(struct Layer* self, float learning_rate, int batch_size);

// Function to zero out accumulated gradients (dW, db).
typedef void (*LayerZeroGradientsFunc)(struct Layer* self);


//------------------------------------------------------------------------------
// Core Structs
//------------------------------------------------------------------------------

// --- Layer Configuration Structs ---


struct __attribute__((packed)) Layer {
    LayerType type;
    int input_size;
    int output_size; // For Dense: output features. For Activation: input_size == output_size.

    // --- Fields for DENSE layer ---
    // Weights: Stored row-major (output_size rows, input_size columns)
    // Access: weights[output_idx * input_size + input_idx]
    float* weights; // Retained as public for now
    float* biases;  // Retained as public for now

    // Gradients (accumulated over a batch)
    float* _dW; // Gradients for weights (same shape as weights)
    float* _db; // Gradients for biases (same shape as biases)

    // --- Fields for ACTIVATION layer ---
    ActivationFunctionPtr _activate_func;        // e.g., activation_sigmoid
    ActivationDerivativePtr _activate_derivative_func; // e.g., derivative_sigmoid

    // --- Internal state pointers for backward pass ---
    // These point to memory locations (likely in Model's buffers) that stored
    // the relevant data during the forward pass for THIS layer.
    const float* _last_input_data;  // Input this layer received (size: input_size)
    const float* _last_output_activations; // Output this layer produced (size: output_size)


    // --- Layer-specific function pointers ---
    LayerForwardPassFunc _forward_pass;
    LayerBackwardPassFunc _backward_pass;
    LayerUpdateWeightsFunc _update_weights; // NULL if layer has no weights (e.g., activation layers)
    LayerZeroGradientsFunc _zero_gradients; // NULL if layer has no gradients to zero

    // Could add a void* user_data for extensibility if needed
};

struct __attribute__((packed)) Model {
    int num_layers;
    Layer* _layers; // Array of Layer structs

    // Overall model input and output dimensions
    int model_input_dim;
    int model_output_dim;

    // --- Internal Buffers for efficient processing ---
    // These buffers are used to pass data between layers during forward and backward passes
    // to avoid repeated allocations/deallocations. Their size is determined by the
    // maximum I/O size required by any layer.
    int _max_intermediate_io_size; // Max of all layer input/output sizes (excluding model input/output buffers if separate)
    
    float* _forward_buffer1;      // Ping-pong buffers for activations/outputs of layers
    float* _forward_buffer2;      // One buffer acts as input, the other as output, then they swap roles.
    
    float* _backward_buffer1;     // Ping-pong buffers for delta values flowing backward
    float* _backward_buffer2;

    // Pointers to track which buffer is current input/output for forward/backward pass
    // These will point to forward_buffer1/2 or backward_buffer1/2.
    float* _current_layer_input_ptr;
    float* _current_layer_output_ptr;
    float* _current_delta_input_ptr;  // Delta coming from next layer
    float* _current_delta_output_ptr; // Delta to be passed to previous layer

    // Buffer to store the final output of the model after a forward pass.
    // This is what get_model_output() will point to.
    float* _final_output_buffer; // Size: model_output_dim


    // --- Optional: Information about the model ---
    size_t total_trainable_params; // Sum of all weights and biases elements
};

struct __attribute__((packed)) TrainConfig {
    float learning_rate;
    int epochs;
    int batch_size;
    unsigned int random_seed; // For reproducible weight initialization and data shuffling
    int shuffle_each_epoch; // Changed from bool to int (0 for false, 1 for true)

    LossFunctionType loss_type;
    ComputeLossPtr _compute_loss_func;         // e.g., loss_mean_squared_error
    LossDerivativePtr _loss_derivative_func; // e.g., derivative_mean_squared_error

    // Optional: Callback after each epoch
    EpochCompletedCallback epoch_callback_func;
    void* epoch_callback_user_data;

    int print_progress_every_n_batches; // 0 to disable
};

struct __attribute__((packed)) Dataset {
    int num_samples;
    int input_dim;
    int output_dim;

    // Data is expected to be in row-major order.
    // inputs_flat: [sample1_feature1, s1_f2, ..., s1_f_input_dim, s2_f1, ...]
    // targets_flat: [sample1_target1, s1_t2, ..., s1_t_output_dim, s2_t1, ...]
    // The Dataset struct itself does not own this memory by default;
    // it points to user-provided data. create_dataset might copy it.
    float* _inputs_flat;  // Changed to float* as create_dataset copies and Dataset owns this memory
    float* _targets_flat; // Changed to float* as create_dataset copies and Dataset owns this memory
};


//------------------------------------------------------------------------------
// Function Declarations
//------------------------------------------------------------------------------

// --- Initialization and Memory Management ---
void seed_random_generator(unsigned int seed_val);

// Creates a model based on an array of layer configurations.
// - model_input_dim: The number of input features for the entire model.
// - num_model_layers: The total number of layers in the model.
// - layer_def_types: An array of LayerType for each layer
// - layer_def_params: An array of parameters (e.g., num_output_neurons for Dense, ignored for Activation)
// The function will chain the input/output dimensions of the layers.
Model* create_model(
    int model_input_dim,
    int num_model_layers,
    const LayerType* layer_def_types, // Array of LayerType for each layer
    const int* layer_def_params     // Array of parameters (e.g., num_output_neurons for Dense, ignored for Activation)
);

void free_model(Model* model);

// Weight Initialization Strategies (call after create_model)
void initialize_weights_xavier_uniform(Model* model); // Good general purpose for symmetric activations (sigmoid, tanh)
void initialize_weights_he_uniform(Model* model);     // Good for ReLU family activations
void initialize_weights_uniform_range(Model* model, float min_val, float max_val);
void initialize_biases_zero(Model* model); // Common practice


// Dataset Management
// create_dataset can choose to copy the input data or just point to it.
// If it copies, free_dataset must free that copied data.
// For wasm, copying might be safer if JS memory might be reclaimed.
Dataset* create_dataset(
    int num_samples, int input_dim, int output_dim,
    const float* inputs_flat_data, // Model will copy this data
    const float* targets_flat_data // Model will copy this data
);
void free_dataset(Dataset* dataset); // Frees the Dataset struct and its internally copied data.


// --- Training ---
void train_model(
    Model* model,
    const Dataset* train_data,
    TrainConfig* config,
    const Dataset* validation_data // Optional, can be NULL
);

// Performs a single training step on a given batch of data.
// - model: The model to train.
// - batch_inputs: Pointer to the input data for the current batch.
// - batch_targets: Pointer to the target data for the current batch.
// - num_samples_in_batch: The number of samples in the current batch.
// - config: Training configuration (learning rate, loss functions, etc.).
// Returns the average loss for the processed batch.
float step_model(
    Model* model,
    const float* batch_inputs,
    const float* batch_targets,
    int num_samples_in_batch,
    const TrainConfig* config
);

// --- Core Operations (might be internal, or exposed for advanced use) ---

// Performs a full forward pass through the model.
// - input_sample: A single input sample (size: model->model_input_dim).
// The result is stored in model->final_output_buffer.
void model_forward_pass(Model* model, const float* input_sample);

// Performs a full backward pass and accumulates gradients in each layer's dW and db.
// - input_sample_for_forward: The input sample that was used for the corresponding forward pass.
// - target_sample: The ground truth for that input sample.
// - config: To access loss function derivatives.
void model_backward_pass(Model* model, const float* target_sample, const TrainConfig* config);

// Updates weights for all layers that have them, using accumulated gradients and learning rate.
// Typically called after processing a batch.
void model_update_weights(Model* model, float learning_rate, int batch_size);

// Zeros out gradients for all layers that have them.
// Typically called before processing a new batch.
void model_zero_gradients(Model* model);


// --- Prediction ---
// Returns a pointer to the model's internal final_output_buffer.
// The content of this buffer is valid until the next call to model_forward_pass on the same model.
// The caller should NOT free this pointer.
const float* get_model_output(const Model* model);

// Convenience function: performs forward pass and returns output pointer.
const float* predict(Model* model, const float* input_sample);


//------------------------------------------------------------------------------
// Pre-defined Activation Functions & Derivatives (Implementations in .cpp)
//------------------------------------------------------------------------------
// Sigmoid
void activation_sigmoid(const float* z, float* a, int size);
void derivative_sigmoid(const float* z __attribute__((unused)), const float* a, float* d, int size);

// ReLU
void activation_relu(const float* z, float* a, int size);
void derivative_relu(const float* z, const float* a __attribute__((unused)), float* d, int size);

// Tanh
void activation_tanh(const float* z, float* a, int size);
void derivative_tanh(const float* z __attribute__((unused)), const float* a, float* d, int size);

//------------------------------------------------------------------------------
// Pre-defined Loss Functions & Derivatives (Implementations in .cpp)
//------------------------------------------------------------------------------
// Mean Squared Error
float loss_mean_squared_error(const float* predictions, const float* targets, int size);
void derivative_mean_squared_error(const float* predictions, const float* targets, float* out_delta, int size);


} // namespace nn
