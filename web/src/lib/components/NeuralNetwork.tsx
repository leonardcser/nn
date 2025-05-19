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
import { Line } from 'react-chartjs-2';
import {
  wasmFetcher,
  type WasmExports,
  WModel,
  WTrainConfig,
  WDataset,
  WFloat32Array,
  WPointer,
} from '../wasm';
import { cn } from '../utils';

const XOR_XDATA = [0, 0, 0, 1, 1, 0, 1, 1];
const XOR_YDATA = [0, 1, 1, 0];
const INPUT_DIM = 2;
const HIDDEN_DIM = 20;
const OUTPUT_DIM = 1;
const SEED = 0;
const LR = 0.05;
const EPOCHS = 100;
const BATCH_SIZE = 4;

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend);

export default function NeuralNetwork() {
  const { data: wasmInstance } = useSWR<WasmExports>(
    import.meta.env.BASE_URL + 'output.wasm',
    wasmFetcher
  );

  const [isTraining, setIsTraining] = useState(false);
  const [lossData, setLossData] = useState<number[]>([]);
  const [currentEpoch, setCurrentEpoch] = useState(0);

  const wasmInstanceRef = useRef(wasmInstance);
  const modelRef = useRef<WModel | null>(null);
  const trainConfigRef = useRef<WTrainConfig | null>(null);
  const datasetRef = useRef<WDataset | null>(null);
  const xSamplesArrayRef = useRef<WFloat32Array | null>(null);
  const yTargetsArrayRef = useRef<WFloat32Array | null>(null);

  useEffect(() => {
    wasmInstanceRef.current = wasmInstance;
  }, [wasmInstance]);

  useEffect(() => {
    if (!wasmInstance) {
      return;
    }

    wasmInstance.nn_seed(SEED);

    const inputDim = INPUT_DIM;
    const hiddenDim = HIDDEN_DIM;
    const outputDim = OUTPUT_DIM;

    // Use WPointer to manage the model pointer holder
    const modelPtrHolder = new WPointer(wasmInstance);

    try {
      const createSuccess = wasmInstance.nn_create_model(
        modelPtrHolder.ptr(),
        inputDim,
        hiddenDim,
        outputDim
      );

      if (!createSuccess) {
        console.error('nn_create_model returned false (C++ model creation failed).');
        return; // Early exit
      }

      const actualModelPtr = modelPtrHolder.get();

      if (actualModelPtr === 0) {
        console.error('nn_create_model succeeded but returned a null model pointer.');
        return; // Early exit
      }

      const _model = new WModel(wasmInstance, actualModelPtr);
      modelRef.current = _model;

      const _trainConfig = new WTrainConfig(wasmInstance, {
        learning_rate: LR,
        epochs: EPOCHS,
        batch_size: BATCH_SIZE,
      });
      trainConfigRef.current = _trainConfig;

      const numSamples = BATCH_SIZE;
      const inputDimPerSample = inputDim;
      const outputDimPerSample = outputDim;

      const _xSamplesArray = new WFloat32Array(wasmInstance, numSamples * inputDimPerSample);
      _xSamplesArray.set(new Float32Array(XOR_XDATA));
      xSamplesArrayRef.current = _xSamplesArray;

      const _yTargetsArray = new WFloat32Array(wasmInstance, numSamples * outputDimPerSample);
      _yTargetsArray.set(new Float32Array(XOR_YDATA));
      yTargetsArrayRef.current = _yTargetsArray;

      const _dataset = new WDataset(wasmInstance, {
        num_samples: numSamples,
        input_dim_per_sample: inputDimPerSample,
        output_dim_per_sample: outputDimPerSample,
        X_samples: _xSamplesArray.ptr(),
        Y_targets: _yTargetsArray.ptr(),
      });
      datasetRef.current = _dataset;
    } finally {
      // Free the memory used for the holder itself, as it's no longer needed
      modelPtrHolder.free();
    }

    return () => {
      modelRef.current?.free();
      trainConfigRef.current?.free();
      datasetRef.current?.free();
      xSamplesArrayRef.current?.free();
      yTargetsArrayRef.current?.free();
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
    setLossData([]);
    setCurrentEpoch(0);

    const trainConfigData = currentTrainConfig.data();
    const datasetData = currentDataset.data();

    const epochs = trainConfigData.epochs;
    const batchSize = trainConfigData.batch_size;
    const numSamples = datasetData.num_samples;
    const numBatches = Math.ceil(numSamples / batchSize);

    for (let epoch = 0; epoch < epochs; epoch++) {
      setCurrentEpoch(epoch + 1);
      for (let batchIdx = 0; batchIdx < numBatches; batchIdx++) {
        const loss = currentWasmInstance.nn_step(
          currentModel.ptr(),
          currentTrainConfig.ptr(),
          currentDataset.ptr(),
          batchIdx
        );
        console.log('loss', loss);
        setLossData((prevLossData) => [...prevLossData, loss]);

        // Wait for 500ms before the next step
        await new Promise((resolve) => setTimeout(resolve, 100));
      }
    }
    setIsTraining(false);
  };

  const chartData = {
    labels: lossData.map((_, i) => `Step ${i + 1}`),
    datasets: [
      {
        label: 'Training Loss',
        data: lossData,
        fill: false,
        borderColor: 'oklch(62.3% 0.214 259.815)',
        tension: 0.1,
      },
    ],
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
      {isTraining && <p>Epoch: {currentEpoch}</p>}
      {lossData.length > 0 && (
        <div className="mt-4">
          <Line data={chartData} />
        </div>
      )}
    </div>
  );
}
