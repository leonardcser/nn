import { type WasmExports, WASM_SYMBOLS } from './types';

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
        fd_seek: (
          fd: number,
          offset: bigint | number, // filedelta
          whence: number, // whence
          newoffset_ptr: number // filesize*
        ): number => {
          // whence: 0 (SEEK_SET), 1 (SEEK_CUR), 2 (SEEK_END)
          const exports = instance.exports as unknown as WasmExports;
          const memory = exports.memory;
          if (!memory) {
            console.error('fd_seek: memory is not available');
            return 1; // WASI errno for EPERM or similar
          }
          const memoryView = new DataView(memory.buffer);

          // For a stub, we'll just pretend the seek was successful and the new offset is 0.
          // A real implementation would need to handle different file descriptors and whence values.
          // For stdout/stderr, seeking doesn't make much sense, but other FDs might.
          if (fd === 1 || fd === 2) {
            // Typically, seek is not supported on stdout/stderr
            // memoryView.setBigUint64(newoffset_ptr, BigInt(0), true); // or some current position if tracked
            // return 29; // ESPIPE - Illegal seek
          }

          // For now, let's assume success and write 0 as the new offset.
          // WASI expects a 64-bit integer for file sizes/offsets.
          memoryView.setBigUint64(newoffset_ptr, BigInt(0), true);
          return 0; // WASI errno for success
        },
        fd_read: (
          fd: number,
          iovs_ptr: number, // iovec_array*
          iovs_len: number, // iovec_array_len
          nread_ptr: number // size*
        ): number => {
          const exports = instance.exports as unknown as WasmExports;
          const memory = exports.memory;
          if (!memory) {
            console.error('fd_read: memory is not available');
            return 1; // WASI errno for EPERM or similar
          }
          const memoryView = new DataView(memory.buffer);

          // For a stub, we'll pretend we read 0 bytes.
          // A real implementation would read from the file descriptor into the iovs buffers.
          // For stdin (fd=0), this would involve providing actual input data.
          // For now, indicate 0 bytes read.
          memoryView.setUint32(nread_ptr, 0, true);
          return 0; // WASI errno for success
        },
        fd_close: (fd: number): number => {
          // In a real implementation, you would clean up resources associated with the fd.
          // For stdout/stderr, closing might be a no-op or an error depending on the environment.
          if (fd === 0 || fd === 1 || fd === 2) {
            // console.warn(`Attempted to close standard fd: ${fd}`);
            // Depending on strictness, could return an error or just succeed.
            // For now, let's say it's successful for standard streams as well.
          }
          return 0; // WASI errno for success
        },
        environ_sizes_get: (
          environ_count_ptr: number, // size*
          environ_buf_size_ptr: number // size*
        ): number => {
          const exports = instance.exports as unknown as WasmExports;
          const memory = exports.memory;
          if (!memory) {
            console.error('environ_sizes_get: memory is not available');
            return 1; // WASI errno for EPERM or similar
          }
          const memoryView = new DataView(memory.buffer);

          // For a stub, we'll report 0 environment variables and 0 buffer size.
          memoryView.setUint32(environ_count_ptr, 0, true);
          memoryView.setUint32(environ_buf_size_ptr, 0, true);
          return 0; // WASI errno for success
        },
        environ_get: (
          environ_ptr: number, // u8**
          environ_buf_ptr: number // u8*
        ): number => {
          // Since environ_sizes_get returns 0, this function should not be called
          // with non-zero buffer sizes, or if it is, it doesn't need to write anything.
          const exports = instance.exports as unknown as WasmExports;
          const memory = exports.memory;
          if (!memory) {
            console.error('environ_get: memory is not available');
            return 1; // WASI errno for EPERM or similar
          }
          // const memoryView = new DataView(memory.buffer);
          // No data to write since we report 0 env vars.
          return 0; // WASI errno for success
        },
        clock_time_get: (
          clock_id: number, // clockid
          precision: bigint, // timestamp
          time_ptr: number // timestamp*
        ): number => {
          const exports = instance.exports as unknown as WasmExports;
          const memory = exports.memory;
          if (!memory) {
            console.error('clock_time_get: memory is not available');
            return 1; // Or an appropriate WASI errno like EACCES or EPERM
          }
          const memoryView = new DataView(memory.buffer);

          if (clock_id === 0) {
            // CLOCK_REALTIME
            // performance.now() gives milliseconds with microsecond precision.
            // Date.now() gives milliseconds.
            // We need nanoseconds for WASI.
            const now_ns = BigInt(Date.now()) * BigInt(1000000);
            memoryView.setBigUint64(time_ptr, now_ns, true);
            return 0; // Success
          }
          console.warn(`clock_time_get called for unhandled clock_id: ${clock_id}`);
          return 28; // WASI errno for EINVAL (Invalid argument)
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

    console.log('instance', instance);

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
