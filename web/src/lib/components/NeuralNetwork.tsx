import useSWR from 'swr';
import { useState } from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import {
  wasmFetcher,
  type WasmExports,
  WModel,
  WTrainConfig,
  WDataset,
  WFloat32Array,
  LayerType,
  LossFunctionType,
} from '../wasm';
import { WInt32Array } from '../wasm/array';
import { cn } from '../utils';

const XOR_XDATA = [0, 0, 0, 1, 1, 0, 1, 1];
const XOR_YDATA = [0, 1, 1, 0];
const INPUT_DIM = 2;
const HIDDEN_DIM = 20;
const OUTPUT_DIM = 1;
const SEED = 0;
const LR = 0.075;
const EPOCHS = 500;
const BATCH_SIZE = 4;

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend);

export default function NeuralNetwork() {
  const { data: wasmInstance } = useSWR<WasmExports>(
    import.meta.env.BASE_URL + 'output.wasm',
    wasmFetcher
  );

  const [isTraining, setIsTraining] = useState(false);

  const startTraining = async () => {
    if (!wasmInstance) {
      console.error('Wasm instance not available');
      return;
    }

    setIsTraining(true);
    console.log('Starting training...');

    let model: WModel | null = null;
    let trainConfig: WTrainConfig | null = null;
    let dataset: WDataset | null = null;
    let xSamplesArray: WFloat32Array | null = null;
    let yTargetsArray: WFloat32Array | null = null;
    let layerTypesArray: WInt32Array | null = null;
    let layerParamsArray: WInt32Array | null = null;
    let singleInputArray: WFloat32Array | null = null;
    // predictionsArray is created and freed within the loop, no need for module-level temporary variable.

    try {
      wasmInstance.nn_seed_random_generator(SEED);

      const inputDim = INPUT_DIM;
      const hiddenDim = HIDDEN_DIM;
      const outputDim = OUTPUT_DIM;

      const layer_types_values: LayerType[] = [
        LayerType.LAYER_TYPE_DENSE,
        LayerType.LAYER_TYPE_ACTIVATION_RELU,
        LayerType.LAYER_TYPE_DENSE,
        LayerType.LAYER_TYPE_ACTIVATION_SIGMOID,
      ];
      const layer_params_values: number[] = [hiddenDim, 0, outputDim, 0];

      layerTypesArray = new WInt32Array(
        wasmInstance,
        layer_types_values.length,
        new Int32Array(layer_types_values)
      );

      layerParamsArray = new WInt32Array(
        wasmInstance,
        layer_params_values.length,
        new Int32Array(layer_params_values)
      );

      const modelPtr = wasmInstance.nn_create_model(
        inputDim,
        layer_types_values.length,
        layerTypesArray.ptr(),
        layerParamsArray.ptr()
      );

      if (modelPtr === 0) {
        console.error('nn_create_model failed: returned a null model pointer.');
        // Early exit, finally block will clean up initialized arrays
        return;
      }

      model = new WModel(wasmInstance, modelPtr);

      wasmInstance.nn_initialize_weights_he_uniform(model.ptr());
      wasmInstance.nn_initialize_biases_zero(model.ptr());

      trainConfig = new WTrainConfig(wasmInstance, {
        learning_rate: LR,
        epochs: EPOCHS,
        batch_size: BATCH_SIZE,
        random_seed: SEED,
        shuffle_each_epoch: 1,
        loss_type: LossFunctionType.LOSS_MEAN_SQUARED_ERROR,
        _compute_loss_func: 0,
        _loss_derivative_func: 0,
        epoch_callback_func_id: wasmInstance.register_callback((data?: DataView) => {
          const loss = data?.getFloat32(0, true);
          console.log(loss);
        }),
        print_progress_every_n_batches: 0,
      });

      const numSamples = XOR_YDATA.length;

      xSamplesArray = new WFloat32Array(wasmInstance, numSamples * inputDim);
      xSamplesArray.set(new Float32Array(XOR_XDATA));

      yTargetsArray = new WFloat32Array(wasmInstance, numSamples * outputDim);
      yTargetsArray.set(new Float32Array(XOR_YDATA));

      dataset = new WDataset(wasmInstance, {
        num_samples: numSamples,
        input_dim: inputDim,
        output_dim: outputDim,
        _inputs_flat: xSamplesArray.ptr(),
        _targets_flat: yTargetsArray.ptr(),
      });

      wasmInstance.nn_train_model(model.ptr(), dataset.ptr(), trainConfig.ptr(), 0);
      console.log('Training complete.');

      if (xSamplesArray && model) {
        console.log('Predictions after training:');
        const xData = XOR_XDATA;
        // inputDim and outputDim are already defined above
        singleInputArray = new WFloat32Array(wasmInstance, inputDim);
        // predictionsArray is created inside the loop

        for (let i = 0; i < xData.length / inputDim; i++) {
          const sampleInput = new Float32Array(xData.slice(i * inputDim, (i + 1) * inputDim));
          singleInputArray.set(sampleInput);

          const outputPtr = wasmInstance.nn_predict(model.ptr(), singleInputArray.ptr());

          // Create and use predictionsArray locally within the loop iteration
          // const predictionsArrayLocal = new WFloat32Array(wasmInstance, outputDim, outputPtr);
          // We don't own outputPtr, so we should copy data if we need it after this scope
          // or ensure WFloat32Array handles this correctly if it's just a view.
          // Assuming WFloat32Array with a ptr argument creates a view and doesn't own the memory.
          // For logging, directly accessing memory.buffer is fine.
          const predictionResult = new Float32Array(
            wasmInstance.memory.buffer,
            outputPtr, // outputPtr from nn_predict is the direct pointer to the result
            outputDim
          );

          console.log(
            `Input: [${sampleInput.join(', ')}], Output: [${predictionResult.join(', ')}] (Expected: ${XOR_YDATA[i]})`
          );
          // predictionsArrayLocal is a view, no need to free if it doesn't own memory.
          // If nn_predict allocates new memory for each prediction, that memory should be freed by a wasm function.
          // Assuming nn_predict returns a pointer to an internal buffer that is reused or managed by the model.
        }
      }
    } catch (error) {
      console.error('Error during training or prediction:', error);
    } finally {
      console.log('Cleaning up WASM objects...');
      model?.free();
      trainConfig?.free();
      dataset?.free();
      xSamplesArray?.free();
      yTargetsArray?.free();
      layerTypesArray?.free();
      layerParamsArray?.free();
      singleInputArray?.free();
      // predictionsArray was scoped locally or handled if it was a view.

      setIsTraining(false);
      console.log('Cleanup complete.');
    }
  };

  return (
    <div
      className={cn('opacity-0 transition-opacity duration-300 p-4', wasmInstance && 'opacity-100')}
    >
      <button
        onClick={startTraining}
        disabled={isTraining || !wasmInstance}
        className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded disabled:opacity-50"
      >
        {isTraining ? 'Training...' : 'Start Training'}
      </button>
      {isTraining && <p>Training in progress... Please check console for logs.</p>}
    </div>
  );
}
