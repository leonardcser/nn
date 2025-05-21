import useSWR from 'swr';
import { useState, useRef } from 'react';
import Plot from 'react-plotly.js';
import type { Layout as PlotlyLayout, Data as PlotlyData } from 'plotly.js';
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
const HIDDEN_DIM = 128;
const OUTPUT_DIM = 1;
const SEED = 0;
const LR = 0.1;
const EPOCHS = 250;
const BATCH_SIZE = 4;
const MIN_DENSE_LAYERS = 1;
const MAX_DENSE_LAYERS = 5;
const DEFAULT_DENSE_LAYERS = 2;

export default function NeuralNetwork() {
  const { data: wasmInstance } = useSWR<WasmExports>(
    import.meta.env.BASE_URL + 'output.wasm',
    wasmFetcher
  );

  const [isTraining, setIsTraining] = useState(false);
  const [lossHistory, setLossHistory] = useState<number[]>([]);
  const [learningRate, setLearningRate] = useState(LR);
  const [epochs, setEpochs] = useState(EPOCHS);
  const [batchSize, setBatchSize] = useState(BATCH_SIZE);
  const [hiddenDim, setHiddenDim] = useState(HIDDEN_DIM);
  const [trainedModel, setTrainedModel] = useState<WModel | null>(null);
  const modelRef = useRef<WModel | null>(null);
  const [numDenseLayers, setNumDenseLayers] = useState(DEFAULT_DENSE_LAYERS);

  const startTraining = async () => {
    if (!wasmInstance) {
      console.error('Wasm instance not available');
      return;
    }

    setIsTraining(true);
    setLossHistory([]); // Reset loss history at the start of training
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
      const outputDim = OUTPUT_DIM;

      // Build layer types and params dynamically
      const layer_types_values: LayerType[] = [];
      const layer_params_values: number[] = [];
      for (let i = 0; i < numDenseLayers; i++) {
        layer_types_values.push(LayerType.LAYER_TYPE_DENSE);
        layer_params_values.push(hiddenDim);
        // Add ReLU after each dense except the last
        if (i < numDenseLayers - 1) {
          layer_types_values.push(LayerType.LAYER_TYPE_ACTIVATION_RELU);
          layer_params_values.push(0);
        }
      }
      // Output layer
      layer_types_values.push(LayerType.LAYER_TYPE_DENSE);
      layer_params_values.push(outputDim);
      layer_types_values.push(LayerType.LAYER_TYPE_ACTIVATION_SIGMOID);
      layer_params_values.push(0);

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
      modelRef.current = model;

      wasmInstance.nn_initialize_weights_he_uniform(model.ptr());
      wasmInstance.nn_initialize_biases_zero(model.ptr());

      trainConfig = new WTrainConfig(wasmInstance, {
        learning_rate: learningRate,
        epochs: epochs,
        batch_size: batchSize,
        random_seed: SEED,
        shuffle_each_epoch: 1,
        loss_type: LossFunctionType.LOSS_MEAN_SQUARED_ERROR,
        _compute_loss_func: 0,
        _loss_derivative_func: 0,
        epoch_callback_func_id: wasmInstance.register_callback((data?: DataView) => {
          const loss = data?.getFloat32(0, true);
          if (loss !== undefined) {
            setLossHistory((prevHistory) => [...prevHistory, loss]);
          }
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
      setTrainedModel(model); // Save model for prediction
      model = null; // Prevent double free in finally
    } catch (error) {
      console.error('Error during training or prediction:', error);
    } finally {
      console.log('Cleaning up WASM objects...');
      if (model) model.free();
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

  const plotlyData: PlotlyData[] = [
    {
      x: lossHistory.map((_, index) => index + 1),
      y: lossHistory,
      type: 'scatter',
      mode: 'lines',
      name: 'Training Loss',
      line: { color: 'rgb(75, 192, 192)' },
    },
  ];

  const plotlyLayout: Partial<PlotlyLayout> = {
    title: { text: 'Training Loss Over Epochs' },
    xaxis: {
      title: { text: 'Epoch' },
    },
    yaxis: {
      title: { text: 'Loss' },
      rangemode: 'tozero',
    },
    autosize: true,
  };

  return (
    <div
      className={cn('opacity-0 transition-opacity duration-300 p-4', wasmInstance && 'opacity-100')}
    >
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-8">
        {/* Left: Sliders and Plot */}
        <div>
          <div className="grid grid-cols-2 gap-4 mb-4">
            <div>
              <label
                htmlFor="lr-slider"
                className="block mb-2 text-sm font-medium text-gray-900 dark:text-white"
              >
                Learning Rate: {learningRate.toExponential(2)}
              </label>
              <input
                id="lr-slider"
                type="range"
                min="0.0001"
                max="1"
                step="0.0001"
                value={learningRate}
                onChange={(e) => setLearningRate(parseFloat(e.target.value))}
                className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-700"
                disabled={isTraining}
              />
            </div>
            <div>
              <label
                htmlFor="epochs-slider"
                className="block mb-2 text-sm font-medium text-gray-900 dark:text-white"
              >
                Epochs: {epochs}
              </label>
              <input
                id="epochs-slider"
                type="range"
                min="100"
                max="10000"
                step="100"
                value={epochs}
                onChange={(e) => setEpochs(parseInt(e.target.value))}
                className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-700"
                disabled={isTraining}
              />
            </div>
            <div>
              <label
                htmlFor="batch-size-slider"
                className="block mb-2 text-sm font-medium text-gray-900 dark:text-white"
              >
                Batch Size: {batchSize}
              </label>
              <input
                id="batch-size-slider"
                type="range"
                min="1"
                max="16"
                step="1"
                value={batchSize}
                onChange={(e) => setBatchSize(parseInt(e.target.value))}
                className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-700"
                disabled={isTraining}
              />
            </div>
            <div>
              <label
                htmlFor="hidden-dim-slider"
                className="block mb-2 text-sm font-medium text-gray-900 dark:text-white"
              >
                Hidden Layer Dims: {hiddenDim}
              </label>
              <input
                id="hidden-dim-slider"
                type="range"
                min="8"
                max="1024"
                step="8"
                value={hiddenDim}
                onChange={(e) => setHiddenDim(parseInt(e.target.value))}
                className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-700"
                disabled={isTraining}
              />
            </div>
            <div>
              <label
                htmlFor="dense-layers-slider"
                className="block mb-2 text-sm font-medium text-gray-900 dark:text-white"
              >
                Dense Layers: {numDenseLayers}
              </label>
              <input
                id="dense-layers-slider"
                type="range"
                min={MIN_DENSE_LAYERS}
                max={MAX_DENSE_LAYERS}
                step={1}
                value={numDenseLayers}
                onChange={(e) => setNumDenseLayers(parseInt(e.target.value))}
                className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-700"
                disabled={isTraining}
              />
            </div>
          </div>
          <button
            onClick={startTraining}
            disabled={isTraining || !wasmInstance}
            className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded disabled:opacity-50"
          >
            {isTraining ? 'Training...' : 'Start Training'}
          </button>
          {isTraining && <p>Training in progress... Please check console for logs.</p>}
        </div>
        {/* Right: Predictions Table */}
        <div className="flex flex-col items-center">
          <div className="w-full max-w-lg">
            <h3 className="text-lg font-semibold mb-2 text-center text-gray-700 dark:text-gray-200">
              XOR Truth Table & Model Predictions
            </h3>
            <table className="min-w-full border border-gray-300 dark:border-gray-700 rounded-lg overflow-hidden">
              <thead className="bg-gray-100 dark:bg-gray-700">
                <tr>
                  <th className="px-4 py-2 border-b border-gray-300 dark:border-gray-600">
                    Input 1
                  </th>
                  <th className="px-4 py-2 border-b border-gray-300 dark:border-gray-600">
                    Input 2
                  </th>
                  <th className="px-4 py-2 border-b border-gray-300 dark:border-gray-600">
                    Expected
                  </th>
                  <th className="px-4 py-2 border-b border-gray-300 dark:border-gray-600">
                    Predicted
                  </th>
                </tr>
              </thead>
              <tbody>
                {[0, 1].flatMap((i1) =>
                  [0, 1].map((i2) => {
                    const expected = i1 ^ i2;
                    let predicted: string | null = null;
                    if (trainedModel && wasmInstance) {
                      // Predict for this input
                      const inputArray = new WFloat32Array(wasmInstance, 2);
                      inputArray.set(new Float32Array([i1, i2]));
                      const outputPtr = wasmInstance.nn_predict(
                        trainedModel.ptr(),
                        inputArray.ptr()
                      );
                      const prediction = new Float32Array(
                        wasmInstance.memory.buffer,
                        outputPtr,
                        1
                      )[0];
                      predicted = prediction.toFixed(4) + ` (→ ${prediction >= 0.5 ? '1' : '0'})`;
                      inputArray.free();
                    }
                    return (
                      <tr
                        key={`${i1}-${i2}`}
                        className="text-center border-b border-gray-200 dark:border-gray-700"
                      >
                        <td className="px-4 py-2">{i1}</td>
                        <td className="px-4 py-2">{i2}</td>
                        <td className="px-4 py-2 font-semibold">{expected}</td>
                        <td className="px-4 py-2">
                          {trainedModel && wasmInstance ? (
                            predicted
                          ) : (
                            <span className="text-gray-400">-</span>
                          )}
                        </td>
                      </tr>
                    );
                  })
                )}
              </tbody>
            </table>
            {!trainedModel && (
              <div className="mt-4 text-sm text-yellow-600 dark:text-yellow-300 text-center">
                Train the model to enable prediction.
              </div>
            )}
          </div>
        </div>
      </div>
      {lossHistory.length > 0 && (
        <div className="mt-4">
          <Plot
            data={plotlyData}
            layout={plotlyLayout}
            style={{ width: '100%', height: '100%' }}
            useResizeHandler={true}
          />
        </div>
      )}
    </div>
  );
}
