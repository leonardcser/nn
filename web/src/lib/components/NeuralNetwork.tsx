import useSWR from 'swr';
import { useState } from 'react';
import {
  wasmFetcher,
  type WasmExports,
  type Point,
  serializeStruct,
  deserializeStruct,
  pointDescriptor,
  calculateStructSize,
} from '../wasm';
import { cn } from '../utils';

export default function NeuralNetwork() {
  const { data: wasmInstance } = useSWR<WasmExports>(
    import.meta.env.BASE_URL + 'output.wasm',
    wasmFetcher
  );

  const [point1X, setPoint1X] = useState(0);
  const [point1Y, setPoint1Y] = useState(0);
  const [point2X, setPoint2X] = useState(0);
  const [point2Y, setPoint2Y] = useState(0);
  const [resultPoint, setResultPoint] = useState<Point | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleAddPoints = () => {
    if (!wasmInstance) {
      setError('WASM module not loaded yet.');
      return;
    }
    setError(null);
    setResultPoint(null);

    let p1Ptr = 0;
    let p2Ptr = 0;
    let resultPtr = 0;

    try {
      const point1: Point = { x: point1X, y: point1Y };
      const point2: Point = { x: point2X, y: point2Y };

      p1Ptr = serializeStruct(wasmInstance, point1, pointDescriptor);
      p2Ptr = serializeStruct(wasmInstance, point2, pointDescriptor);
      resultPtr = wasmInstance.allocate_memory(calculateStructSize(pointDescriptor));

      if (p1Ptr === 0 || p2Ptr === 0 || resultPtr === 0) {
        throw new Error('Memory allocation failed for one of the points.');
      }

      wasmInstance.add_points(p1Ptr, p2Ptr, resultPtr);
      const result = deserializeStruct(wasmInstance, resultPtr, pointDescriptor);
      setResultPoint(result);
    } catch (e: any) {
      console.error('Error in handleAddPoints:', e);
      setError(`Error: ${e.message || 'An unknown error occurred.'}`);
      setResultPoint(null);
    } finally {
      if (wasmInstance) {
        if (p1Ptr !== 0) wasmInstance.free_memory(p1Ptr);
        if (p2Ptr !== 0) wasmInstance.free_memory(p2Ptr);
        if (resultPtr !== 0) wasmInstance.free_memory(resultPtr);
      }
    }
  };

  return (
    <div
      className={cn('opacity-0 transition-opacity duration-300 p-4', wasmInstance && 'opacity-100')}
    >
      <h1 className="text-2xl font-bold mb-4">WASM Point Adder</h1>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
        <div className="border p-3 rounded">
          <h2 className="text-lg font-semibold mb-2">Point 1</h2>
          <div className="flex space-x-2">
            <input
              type="number"
              value={point1X}
              onChange={(e) => setPoint1X(Number(e.target.value))}
              className="border p-2 rounded w-1/2"
              placeholder="X1"
            />
            <input
              type="number"
              value={point1Y}
              onChange={(e) => setPoint1Y(Number(e.target.value))}
              className="border p-2 rounded w-1/2"
              placeholder="Y1"
            />
          </div>
        </div>
        <div className="border p-3 rounded">
          <h2 className="text-lg font-semibold mb-2">Point 2</h2>
          <div className="flex space-x-2">
            <input
              type="number"
              value={point2X}
              onChange={(e) => setPoint2X(Number(e.target.value))}
              className="border p-2 rounded w-1/2"
              placeholder="X2"
            />
            <input
              type="number"
              value={point2Y}
              onChange={(e) => setPoint2Y(Number(e.target.value))}
              className="border p-2 rounded w-1/2"
              placeholder="Y2"
            />
          </div>
        </div>
      </div>
      <button
        onClick={handleAddPoints}
        disabled={!wasmInstance}
        className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded w-full md:w-auto mb-4"
      >
        Add Points
      </button>
      {error && (
        <div className="mt-4 p-3 bg-red-100 text-red-700 rounded">
          <p>{error}</p>
        </div>
      )}
      {resultPoint && (
        <div className="mt-4 p-4 bg-gray-100 rounded">
          <h2 className="text-xl">
            Result Point: ({resultPoint.x}, {resultPoint.y})
          </h2>
        </div>
      )}
    </div>
  );
}
