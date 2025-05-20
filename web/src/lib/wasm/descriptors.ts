import { WStruct } from './struct';
import {
  FieldType,
  type StructDescriptor,
  type WasmExports,
  type Model,
  type TrainConfig,
  type Dataset,
  type Layer,
} from './types';

// Struct descriptors
const modelDescriptor: StructDescriptor<Model> = {
  num_layers: FieldType.Int32,
  _layers: FieldType.Ptr, // Layer*
  model_input_dim: FieldType.Int32,
  model_output_dim: FieldType.Int32,
  _max_intermediate_io_size: FieldType.Int32,
  _forward_buffer1: FieldType.Ptr, // float*
  _forward_buffer2: FieldType.Ptr, // float*
  _backward_buffer1: FieldType.Ptr, // float*
  _backward_buffer2: FieldType.Ptr, // float*
  _current_layer_input_ptr: FieldType.Ptr, // float*
  _current_layer_output_ptr: FieldType.Ptr, // float*
  _current_delta_input_ptr: FieldType.Ptr, // float*
  _current_delta_output_ptr: FieldType.Ptr, // float*
  _final_output_buffer: FieldType.Ptr, // float*
  total_trainable_params: FieldType.Uint32, // size_t
};

const trainConfigDescriptor: StructDescriptor<TrainConfig> = {
  learning_rate: FieldType.Float32,
  epochs: FieldType.Int32,
  batch_size: FieldType.Int32,
  random_seed: FieldType.Uint32, // unsigned int
  loss_type: FieldType.Int32, // enum LossFunctionType
  _compute_loss_func: FieldType.Ptr, // ComputeLossPtr
  _loss_derivative_func: FieldType.Ptr, // LossDerivativePtr
  print_progress_every_n_batches: FieldType.Int32,
};

const datasetDescriptor: StructDescriptor<Dataset> = {
  num_samples: FieldType.Int32,
  input_dim: FieldType.Int32,
  output_dim: FieldType.Int32,
  _inputs_flat: FieldType.Ptr, // float*
  _targets_flat: FieldType.Ptr, // float*
};

const layerDescriptor: StructDescriptor<Layer> = {
  type: FieldType.Int32, // enum LayerType
  input_size: FieldType.Int32,
  output_size: FieldType.Int32,
  weights: FieldType.Ptr, // float*
  biases: FieldType.Ptr, // float*
  _dW: FieldType.Ptr, // float*
  _db: FieldType.Ptr, // float*
  _activate_func: FieldType.Ptr, // ActivationFunctionPtr
  _activate_derivative_func: FieldType.Ptr, // ActivationDerivativePtr
  _last_input_data: FieldType.Ptr, // const float*
  _last_output_activations: FieldType.Ptr, // const float*
  _forward_pass: FieldType.Ptr, // LayerForwardPassFunc
  _backward_pass: FieldType.Ptr, // LayerBackwardPassFunc
  _update_weights: FieldType.Ptr, // LayerUpdateWeightsFunc (nullable)
  _zero_gradients: FieldType.Ptr, // LayerZeroGradientsFunc (nullable)
};

// WStruct subclasses
export class WModel extends WStruct<Model> {
  constructor(exports: WasmExports, source?: number | Model) {
    super(exports, modelDescriptor, source);
  }

  public override free(): void {
    if (this._ptr !== 0) {
      this.exports.nn_free_model(this._ptr);
      this._ptr = 0; // Mark as freed
    }
  }
}

export class WTrainConfig extends WStruct<TrainConfig> {
  constructor(exports: WasmExports, source?: TrainConfig) {
    super(exports, trainConfigDescriptor, source);
  }
  // No specific free function for TrainConfig in C API, assuming it's stack allocated or part of other structs
}

export class WDataset extends WStruct<Dataset> {
  constructor(exports: WasmExports, source?: number | Dataset) {
    super(exports, datasetDescriptor, source);
  }

  public override free(): void {
    if (this._ptr !== 0) {
      this.exports.nn_free_dataset(this._ptr);
      this._ptr = 0;
    }
  }
}

export class WLayer extends WStruct<Layer> {
  constructor(exports: WasmExports, source?: number | Layer) {
    super(exports, layerDescriptor, source);
  }
  // Layers are part of the Model, freed with Model
}
