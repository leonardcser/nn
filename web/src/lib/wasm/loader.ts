import type { WasmExports } from './types';

export async function wasmFetcher(url: string): Promise<WasmExports> {
  try {
    const response = await fetch(url);
    const buffer = await response.arrayBuffer();
    const module = await WebAssembly.compile(buffer);
    const importObject = {
      env: {
        emscripten_notify_memory_growth: (memoryIndex: number) => {
          console.debug(`Wasm memory grown (index: ${memoryIndex})`);
        },
        abort: (reasonCodeOrPtr: number) => {
          console.error(`Wasm execution aborted. Reason: ${reasonCodeOrPtr}`);
          throw new Error(`Wasm execution aborted. Reason: ${reasonCodeOrPtr}`);
        },
      },
    };
    const instance = await WebAssembly.instantiate(module, importObject);
    const exports = instance.exports as Record<string, WebAssembly.ExportValue>;
    const functions: { [key: string]: CallableFunction } = {};
    let memory: WebAssembly.Memory | undefined = undefined;

    for (const key in exports) {
      if (typeof exports[key] === 'function') {
        functions[key] = exports[key] as CallableFunction;
      } else if (exports[key] instanceof WebAssembly.Memory) {
        memory = exports[key] as WebAssembly.Memory;
      }
    }

    if (!memory) {
      throw new Error("WASM module did not export 'memory'");
    }

    return { ...functions, memory } as unknown as WasmExports;
  } catch (error) {
    console.error('Error loading WASM module:', error);
    throw error; // Re-throw to allow caller to handle the error
  }
}
