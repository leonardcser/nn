export interface WasmExports {
  memory: WebAssembly.Memory;
  allocate_memory: (size: number) => number; // returns void*
  free_memory: (ptr: number) => void;
  nn_seed_random_generator: (seed_val: number) => void;
  nn_create_model: (
    model_input_dim: number,
    num_model_layers: number,
    layer_types_arr: number, // int* (representing LayerType[])
    layer_params_arr: number // int* (parameters for each layer, e.g., num_output_neurons for Dense)
  ) => number; // returns Model*
  nn_free_model: (model_ptr: number) => void;
  nn_initialize_weights_xavier_uniform: (model_ptr: number) => void;
  nn_initialize_weights_he_uniform: (model_ptr: number) => void;
  nn_initialize_weights_uniform_range: (
    model_ptr: number,
    min_val: number,
    max_val: number
  ) => void;
  nn_initialize_biases_zero: (model_ptr: number) => void;
  nn_create_dataset: (
    num_samples: number,
    input_dim: number,
    output_dim: number,
    inputs_flat_data: number, // float*
    targets_flat_data: number // float*
  ) => number; // returns Dataset*
  nn_free_dataset: (dataset_ptr: number) => void;
  nn_train_model: (
    model_ptr: number, // Model*
    train_data_ptr: number, // Dataset*
    config_ptr: number, // TrainConfig*
    validation_data_ptr: number | null // Dataset* (optional, can be 0 for null)
  ) => void;
  nn_step_model: (
    model_ptr: number, // Model*
    batch_inputs_ptr: number, // float*
    batch_targets_ptr: number, // float*
    num_samples_in_batch: number, // int
    config_ptr: number // TrainConfig*
  ) => number; // returns float (average batch loss)
  nn_get_model_output: (model_ptr: number) => number; // returns float*
  nn_predict: (model_ptr: number, input_sample_ptr: number) => number; // returns float*
  nn_get_layer_output_activations: (
    model_ptr: number,
    layer_index: number,
    out_activation_size_ptr: number // int*
  ) => number; // returns float*

  // JS defined functions
  register_callback: (func: (data?: DataView) => void) => number;
  _callback: (id: number, data: number, data_size: number) => void;
}

export type WasmSymbol = keyof WasmExports;

export const WASM_SYMBOLS: WasmSymbol[] = [
  'memory',
  'allocate_memory',
  'free_memory',
  'nn_seed_random_generator',
  'nn_create_model',
  'nn_free_model',
  'nn_initialize_weights_xavier_uniform',
  'nn_initialize_weights_he_uniform',
  'nn_initialize_weights_uniform_range',
  'nn_initialize_biases_zero',
  'nn_create_dataset',
  'nn_free_dataset',
  'nn_train_model',
  'nn_step_model',
  'nn_get_model_output',
  'nn_predict',
  'nn_get_layer_output_activations',
] as const;

export enum FieldType {
  Int32,
  Float32,
  Uint32, // for size_t types like dimensions, counts, etc.
  Ptr, // Alias for Uint32, representing a pointer
}

export type StructDescriptor<T extends Record<string, any>> = {
  [K in keyof T]: FieldType;
};

export enum LayerType {
  LAYER_TYPE_DENSE,
  LAYER_TYPE_ACTIVATION_SIGMOID,
  LAYER_TYPE_ACTIVATION_RELU,
  LAYER_TYPE_ACTIVATION_TANH,
}

export enum LossFunctionType {
  LOSS_MEAN_SQUARED_ERROR,
}

export interface Model {
  num_layers: number; // int
  _layers: number; // Layer*
  model_input_dim: number; // int
  model_output_dim: number; // int
  _max_intermediate_io_size: number; // int
  _forward_buffer1: number; // float*
  _forward_buffer2: number; // float*
  _backward_buffer1: number; // float*
  _backward_buffer2: number; // float*
  _current_layer_input_ptr: number; // float*
  _current_layer_output_ptr: number; // float*
  _current_delta_input_ptr: number; // float*
  _current_delta_output_ptr: number; // float*
  _final_output_buffer: number; // float*
  total_trainable_params: number; // size_t -> Uint32
}

export interface TrainConfig {
  learning_rate: number; // float
  epochs: number; // int
  batch_size: number; // int
  random_seed: number; // unsigned int -> Uint32
  shuffle_each_epoch: number; // int (0 for false, 1 for true)
  loss_type: LossFunctionType; // enum
  _compute_loss_func: number; // ComputeLossPtr
  _loss_derivative_func: number; // LossDerivativePtr
  epoch_callback_func_id: number; // int (-1 to disable)
  print_progress_every_n_batches: number; // int
}

export interface Dataset {
  num_samples: number; // int
  input_dim: number; // int
  output_dim: number; // int
  _inputs_flat: number; // float*
  _targets_flat: number; // float*
}

export interface Layer {
  type: LayerType; // enum
  input_size: number; // int
  output_size: number; // int
  weights: number; // float*
  biases: number; // float*
  _dW: number; // float*
  _db: number; // float*
  _activate_func: number; // ActivationFunctionPtr
  _activate_derivative_func: number; // ActivationDerivativePtr
  _last_input_data: number; // const float*
  _last_output_activations: number; // const float*
  _forward_pass: number; // LayerForwardPassFunc
  _backward_pass: number; // LayerBackwardPassFunc
  _update_weights: number; // LayerUpdateWeightsFunc (nullable)
  _zero_gradients: number; // LayerZeroGradientsFunc (nullable)
}
