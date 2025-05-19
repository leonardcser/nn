import { assert } from '../utils';
import { type WasmExports, type WasmSymbol, WASM_SYMBOLS } from './types';

export async function wasmFetcher(url: string): Promise<WasmExports> {
  try {
    const response = await fetch(url);
    const buffer = await response.arrayBuffer();
    const module = await WebAssembly.compile(buffer);

    let instance: WebAssembly.Instance;

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
      wasi_snapshot_preview1: {
        fd_write: (
          fd: number,
          iovs_ptr: number,
          iovs_len: number,
          nwritten_ptr: number
        ): number => {
          let nwritten = 0;
          if (fd === 1 || fd === 2) {
            // stdout or stderr
            const exports = instance.exports as unknown as WasmExports;
            const memory = exports.memory;
            if (!memory) {
              console.error('fd_write: memory is not available');
              return 1; // Indicate an error (e.g., EPERM or another WASI errno)
            }
            const memoryView = new DataView(memory.buffer);
            const iovecs = new Uint32Array(memory.buffer, iovs_ptr, iovs_len * 2); // Each iovec is [ptr, len]

            let output = '';
            for (let i = 0; i < iovs_len; i++) {
              const ptr = iovecs[i * 2];
              const len = iovecs[i * 2 + 1];
              const bytes = new Uint8Array(memory.buffer, ptr, len);
              output += new TextDecoder().decode(bytes);
              nwritten += len;
            }

            if (fd === 1) console.log(output);
            if (fd === 2) console.error(output);

            memoryView.setUint32(nwritten_ptr, nwritten, true);
            return 0; // Success
          }
          console.warn(`fd_write called for unhandled fd: ${fd}`);
          return 1; // Indicate an error for unhandled file descriptors
        },
      },
    };
    instance = await WebAssembly.instantiate(module, importObject);
    const exports = instance.exports as Record<string, WebAssembly.ExportValue>;
    const functions: { [key: string]: CallableFunction } = {};
    let memory: WebAssembly.Memory | undefined = undefined;

    const missingSymbols = WASM_SYMBOLS.filter((symbol) => !(symbol in exports));
    if (missingSymbols.length > 0) {
      throw new Error(
        `Wasm module did not export the following symbols: ${missingSymbols.join(', ')}`
      );
    }

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
