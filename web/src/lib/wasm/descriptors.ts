import { WStruct } from './struct';
import {
  FieldType,
  type StructDescriptor,
  type WasmExports,
  type Model,
  type TrainConfig,
  type Dataset,
} from './types';

// Struct descriptors
const modelDescriptor: StructDescriptor<Model> = {
  input_dim: FieldType.Uint32,
  hidden_dim: FieldType.Uint32,
  output_dim: FieldType.Uint32,
  weights_input_hidden: FieldType.Ptr,
  biases_hidden: FieldType.Ptr,
  weights_hidden_output: FieldType.Ptr,
  biases_output: FieldType.Ptr,
  hidden_layer_output: FieldType.Ptr,
  final_output: FieldType.Ptr,
  losses_history: FieldType.Ptr,
};

const trainConfigDescriptor: StructDescriptor<TrainConfig> = {
  learning_rate: FieldType.Float32,
  epochs: FieldType.Uint32,
  batch_size: FieldType.Uint32,
};

const datasetDescriptor: StructDescriptor<Dataset> = {
  num_samples: FieldType.Uint32,
  input_dim_per_sample: FieldType.Uint32,
  output_dim_per_sample: FieldType.Uint32,
  X_samples: FieldType.Ptr,
  Y_targets: FieldType.Ptr,
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
}

export class WDataset extends WStruct<Dataset> {
  constructor(exports: WasmExports, source?: Dataset) {
    super(exports, datasetDescriptor, source);
  }
}
