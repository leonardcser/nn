export interface WasmExports {
  memory: WebAssembly.Memory;
  allocate_memory: (size: number) => number; // returns pointer
  free_memory: (ptr: number) => void;
  add_points: (ptr1: number, ptr2: number, resultPtr: number) => void;
}

export enum FieldType {
  Int32,
  // Add other types like Float32, Int8, etc. as needed
}

export type StructDescriptor<T extends Record<string, any>> = {
  [K in keyof T]: FieldType;
};

export interface Point {
  x: number;
  y: number;
}
