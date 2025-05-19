import useSWR from 'swr';
import { useState } from 'react';
import { wasmFetcher, type WasmExports, type Point, WPoint } from '../wasm';
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

    let p1: WPoint | null = null;
    let p2: WPoint | null = null;
    let result: WPoint | null = null;

    try {
      p1 = new WPoint(wasmInstance, { x: point1X, y: point1Y });
      p2 = new WPoint(wasmInstance, { x: point2X, y: point2Y });
      result = new WPoint(wasmInstance, { x: 0, y: 0 });

      wasmInstance.add_points(p1.ptr(), p2.ptr(), result.ptr());

      setResultPoint(result.data());
    } catch (e: any) {
      console.error('Error in handleAddPoints:', e);
      setError(`Error: ${e.message || 'An unknown error occurred.'}`);
      setResultPoint(null);
    } finally {
      p1?.free();
      p2?.free();
      result?.free();
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
