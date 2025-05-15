export interface WasmInstance {
  add: (a: number, b: number) => number;
}
export async function wasmFetcher(url: string): Promise<WasmInstance> {
  const response = await fetch(url);
  const buffer = await response.arrayBuffer();
  const module = await WebAssembly.compile(buffer);
  const instance = await WebAssembly.instantiate(module);
  const exports = instance.exports;
  const functions: { [key: string]: CallableFunction } = {};
  for (const key in exports) {
    if (typeof exports[key] === 'function') {
      functions[key] = exports[key] as CallableFunction;
    }
  }
  return functions as unknown as WasmInstance;
}
