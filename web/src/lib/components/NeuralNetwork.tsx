import useSWR from 'swr';
import { useEffect, useState, useRef } from 'react';
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

  const wasmInstanceRef = useRef(wasmInstance);
  const modelRef = useRef<WModel | null>(null);
  const trainConfigRef = useRef<WTrainConfig | null>(null);
  const datasetRef = useRef<WDataset | null>(null);
  const xSamplesArrayRef = useRef<WFloat32Array | null>(null);
  const yTargetsArrayRef = useRef<WFloat32Array | null>(null);
  const layerTypesArrayRef = useRef<WInt32Array | null>(null);
  const layerParamsArrayRef = useRef<WInt32Array | null>(null);

  useEffect(() => {
    wasmInstanceRef.current = wasmInstance;
  }, [wasmInstance]);

  useEffect(() => {
    if (!wasmInstance) {
      return;
    }

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

    const _layerTypesArray = new WInt32Array(
      wasmInstance,
      layer_types_values.length,
      new Int32Array(layer_types_values)
    );
    layerTypesArrayRef.current = _layerTypesArray;

    const _layerParamsArray = new WInt32Array(
      wasmInstance,
      layer_params_values.length,
      new Int32Array(layer_params_values)
    );
    layerParamsArrayRef.current = _layerParamsArray;

    const modelPtr = wasmInstance.nn_create_model(
      inputDim,
      layer_types_values.length,
      _layerTypesArray.ptr(),
      _layerParamsArray.ptr()
    );

    if (modelPtr === 0) {
      console.error('nn_create_model failed: returned a null model pointer.');
      _layerTypesArray.free();
      _layerParamsArray.free();
      layerTypesArrayRef.current = null;
      layerParamsArrayRef.current = null;
      return;
    }

    const _model = new WModel(wasmInstance, modelPtr);
    modelRef.current = _model;

    wasmInstance.nn_initialize_weights_he_uniform(modelPtr);
    wasmInstance.nn_initialize_biases_zero(modelPtr);

    const _trainConfig = new WTrainConfig(wasmInstance, {
      learning_rate: LR,
      epochs: EPOCHS,
      batch_size: BATCH_SIZE,
      random_seed: SEED,
      loss_type: LossFunctionType.LOSS_MEAN_SQUARED_ERROR,
      _compute_loss_func: 0,
      _loss_derivative_func: 0,
      print_progress_every_n_batches: 1,
    });
    trainConfigRef.current = _trainConfig;

    const numSamples = XOR_YDATA.length;

    const _xSamplesArray = new WFloat32Array(wasmInstance, numSamples * inputDim);
    _xSamplesArray.set(new Float32Array(XOR_XDATA));
    xSamplesArrayRef.current = _xSamplesArray;

    const _yTargetsArray = new WFloat32Array(wasmInstance, numSamples * outputDim);
    _yTargetsArray.set(new Float32Array(XOR_YDATA));
    yTargetsArrayRef.current = _yTargetsArray;

    const _dataset = new WDataset(wasmInstance, {
      num_samples: numSamples,
      input_dim: inputDim,
      output_dim: outputDim,
      _inputs_flat: _xSamplesArray.ptr(),
      _targets_flat: _yTargetsArray.ptr(),
    });
    datasetRef.current = _dataset;

    return () => {
      modelRef.current?.free();
      trainConfigRef.current?.free();
      datasetRef.current?.free();
      xSamplesArrayRef.current?.free();
      yTargetsArrayRef.current?.free();
      layerTypesArrayRef.current?.free();
      layerParamsArrayRef.current?.free();
    };
  }, [wasmInstance]);

  const startTraining = async () => {
    const currentWasmInstance = wasmInstanceRef.current;
    const currentModel = modelRef.current;
    const currentTrainConfig = trainConfigRef.current;
    const currentDataset = datasetRef.current;

    if (!currentWasmInstance || !currentModel || !currentTrainConfig || !currentDataset) {
      console.error('Wasm modules not initialized');
      return;
    }

    setIsTraining(true);

    console.log('Starting training...');
    try {
      currentWasmInstance.nn_train_model(
        currentModel.ptr(),
        currentDataset.ptr(),
        currentTrainConfig.ptr(),
        0
      );
      console.log('Training complete.');

      if (xSamplesArrayRef.current && modelRef.current) {
        console.log('Predictions after training:');
        const xData = XOR_XDATA;
        const inputDim = INPUT_DIM;
        const outputDim = OUTPUT_DIM;
        const singleInputArray = new WFloat32Array(currentWasmInstance, inputDim);
        const predictionsArray = new WFloat32Array(currentWasmInstance, outputDim);

        for (let i = 0; i < xData.length / inputDim; i++) {
          const sampleInput = new Float32Array(xData.slice(i * inputDim, (i + 1) * inputDim));
          singleInputArray.set(sampleInput);

          const outputPtr = currentWasmInstance.nn_predict(
            modelRef.current.ptr(),
            singleInputArray.ptr()
          );

          const predictionResult = new Float32Array(
            currentWasmInstance.memory.buffer,
            outputPtr,
            outputDim
          );

          console.log(
            `Input: [${sampleInput.join(', ')}], Output: [${predictionResult.join(', ')}] (Expected: ${XOR_YDATA[i]})`
          );
        }
        singleInputArray.free();
        predictionsArray.free();
      }
    } catch (error) {
      console.error('Error during training:', error);
    } finally {
      setIsTraining(false);
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
