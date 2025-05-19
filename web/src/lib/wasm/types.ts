export interface WasmExports {
  memory: WebAssembly.Memory;
  allocate_memory: (size: number) => number; // returns pointer
  free_memory: (ptr: number) => void;
  nn_seed: (seed_val: number) => void;
  nn_create_model: (
    model_ptr_ptr: number,
    input_dim: number,
    hidden_dim: number,
    output_dim: number
  ) => boolean;
  nn_train: (model_ptr: number, config_ptr: number, training_data_ptr: number) => void;
  nn_step: (
    model_ptr: number,
    config_ptr: number,
    training_data_ptr: number,
    batch_idx: number
  ) => number;
  nn_infer: (model_ptr: number, input_ptr: number, output_ptr: number) => void;
  nn_free_model: (model_ptr: number) => void;
  nn_free_dataset: (dataset_ptr: number) => void;
}

export type WasmSymbol = keyof WasmExports;

export const WASM_SYMBOLS: WasmSymbol[] = [
  'memory',
  'allocate_memory',
  'free_memory',
  'nn_seed',
  'nn_create_model',
  'nn_train',
  'nn_step',
  'nn_infer',
  'nn_free_model',
  'nn_free_dataset',
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

export interface Model {
  input_dim: number; // size_t
  hidden_dim: number; // size_t
  output_dim: number; // size_t
  weights_input_hidden: number; // float* -> pointer
  biases_hidden: number; // float* -> pointer
  weights_hidden_output: number; // float* -> pointer
  biases_output: number; // float* -> pointer
  hidden_layer_output: number; // float* -> pointer
  final_output: number; // float* -> pointer
  losses_history: number; // float* -> pointer
}

export interface TrainConfig {
  learning_rate: number; // float
  epochs: number; // size_t
  batch_size: number; // size_t
}

export interface Dataset {
  num_samples: number; // size_t
  input_dim_per_sample: number; // size_t
  output_dim_per_sample: number; // size_t
  X_samples: number; // float* -> pointer
  Y_targets: number; // float* -> pointer
}
