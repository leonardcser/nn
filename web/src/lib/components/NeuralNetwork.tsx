import useSWR from 'swr';
import { useState } from 'react';
import { wasmFetcher } from '../wasm';
import { cn } from '../utils';

export default function NeuralNetwork() {
  const { data: wasmInstance } = useSWR(import.meta.env.BASE_URL + 'output.wasm', wasmFetcher);

  const [num1, setNum1] = useState(0);
  const [num2, setNum2] = useState(0);
  const [result, setResult] = useState<number | null>(null);

  const handleAdd = () => {
    if (wasmInstance && typeof (wasmInstance as any).add === 'function') {
      const sum = (wasmInstance as any).add(num1, num2);
      setResult(sum);
    } else {
      console.error('wasmInstance.add function not found or not a function');
      setResult(null);
    }
  };

  return (
    <div className={cn('opacity-0 transition-opacity duration-300', wasmInstance && 'opacity-100')}>
      <h1 className="text-2xl font-bold mb-4">WASM Adder</h1>
      <div className="flex flex-col space-y-4 md:flex-row md:space-y-0 md:space-x-4 mb-4">
        <input
          type="number"
          value={num1}
          onChange={(e) => setNum1(Number(e.target.value))}
          className="border p-2 rounded w-full md:w-1/3"
          placeholder="Enter first number"
        />
        <input
          type="number"
          value={num2}
          onChange={(e) => setNum2(Number(e.target.value))}
          className="border p-2 rounded w-full md:w-1/3"
          placeholder="Enter second number"
        />
        <button
          onClick={handleAdd}
          disabled={!wasmInstance}
          className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded w-full md:w-auto"
        >
          Add
        </button>
      </div>
      {result !== null && (
        <div className="mt-4 p-4 bg-gray-100 rounded">
          <h2 className="text-xl">Result: {result}</h2>
        </div>
      )}
    </div>
  );
}
