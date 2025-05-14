export interface WasmExports {
  add: (a: number, b: number) => number;
}

export async function wasmFetcher(url: string): Promise<WasmExports> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch Wasm module: ${response.statusText} (URL: ${response.url})`);
  }

  const wasmBuffer = await response.arrayBuffer();

  const importObject = {}; // Empty import object for now
  const { instance } = await WebAssembly.instantiate(wasmBuffer, importObject);

  const exportsObject: { [key: string]: any } = {};
  for (const exportName in instance.exports) {
    if (Object.prototype.hasOwnProperty.call(instance.exports, exportName)) {
      exportsObject[exportName] = instance.exports[exportName];
    }
  }
  return exportsObject as WasmExports;
}
