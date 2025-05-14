import type { Route } from "./+types/home";
import { wasmFetcher } from "../utils/fetchers";
import { useState } from "react";

export function meta({}: Route.MetaArgs) {
  return [
    { title: "New React Router App" },
    { name: "description", content: "Welcome to React Router!" },
  ];
}

export async function clientLoader() {
  return {
    wasmInstance: await wasmFetcher("/output.wasm"),
  };
}

export function HydrateFallback() {
  return <div>Loading...</div>;
}

export default function Home({ loaderData }: Route.ComponentProps) {
  const { wasmInstance } = loaderData;

  const [num1, setNum1] = useState<number>(0);
  const [num2, setNum2] = useState<number>(0);
  const [sum, setSum] = useState<number | null>(null);

  const handleAdd = () => {
    const result = wasmInstance.add(num1, num2);
    setSum(result);
  };

  return (
    <div className="p-4">
      <h1 className="text-2xl font-bold mb-4">Wasm Calculator</h1>
      <p className="mb-4">Wasm module loaded successfully!</p>
      <div className="flex gap-2 mb-4">
        <input
          type="number"
          value={num1}
          onChange={(e) => setNum1(parseInt(e.target.value, 10) || 0)}
          placeholder="Enter first number"
          className="border p-2 flex-1"
        />
        <input
          type="number"
          value={num2}
          onChange={(e) => setNum2(parseInt(e.target.value, 10) || 0)}
          placeholder="Enter second number"
          className="border p-2 flex-1"
        />
      </div>
      <button
        onClick={handleAdd}
        className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded"
      >
        Add
      </button>
      {sum !== null && (
        <p className="mt-4 text-lg">
          Result of {num1} + {num2}:{" "}
          <span className="font-semibold">{sum}</span>
        </p>
      )}
    </div>
  );
}
