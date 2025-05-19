import { FieldType, type StructDescriptor, type WasmExports } from './types';

export const fieldTypeToSize: Record<FieldType, number> = {
  [FieldType.Int32]: 4,
};

export const fieldTypeToDataViewSetter: Record<FieldType, keyof DataView> = {
  [FieldType.Int32]: 'setInt32',
};

export const fieldTypeToDataViewGetter: Record<FieldType, keyof DataView> = {
  [FieldType.Int32]: 'getInt32',
};

export function calculateStructSize<T extends Record<string, any>>(
  descriptor: StructDescriptor<T>
): number {
  let size = 0;
  for (const key in descriptor) {
    size += fieldTypeToSize[descriptor[key]];
  }
  return size;
}

export function serializeStruct<T extends Record<string, any>>(
  exports: WasmExports,
  data: T,
  descriptor: StructDescriptor<T>
): number {
  const size = calculateStructSize(descriptor);
  const ptr = exports.allocate_memory(size);
  if (ptr === 0) {
    throw new Error('WASM memory allocation failed');
  }
  const memoryView = new DataView(exports.memory.buffer, ptr, size);
  let offset = 0;
  for (const key in descriptor) {
    if (Object.prototype.hasOwnProperty.call(data, key)) {
      const fieldType = descriptor[key];
      const value = data[key];
      const setterName = fieldTypeToDataViewSetter[fieldType];
      if (typeof memoryView[setterName] === 'function') {
        (
          memoryView[setterName] as (
            byteOffset: number,
            value: number,
            littleEndian?: boolean
          ) => void
        )(offset, value as number, true); // Assuming little endian
      } else {
        throw new Error(`Unsupported setter for field type: ${FieldType[fieldType]}`);
      }
      offset += fieldTypeToSize[fieldType];
    }
  }
  return ptr;
}

export function deserializeStruct<T extends Record<string, any>>(
  exports: WasmExports,
  ptr: number,
  descriptor: StructDescriptor<T>
): T {
  if (ptr === 0) {
    throw new Error('Cannot deserialize from null pointer');
  }
  const size = calculateStructSize(descriptor);
  const memoryView = new DataView(exports.memory.buffer, ptr, size);
  const result: Partial<T> = {};
  let offset = 0;
  for (const key in descriptor) {
    const fieldType = descriptor[key];
    const getterName = fieldTypeToDataViewGetter[fieldType];
    if (typeof memoryView[getterName] === 'function') {
      result[key] = (
        memoryView[getterName] as (byteOffset: number, littleEndian?: boolean) => number
      )(offset, true) as T[Extract<keyof T, string>]; // Assuming little endian
    } else {
      throw new Error(`Unsupported getter for field type: ${FieldType[fieldType]}`);
    }
    offset += fieldTypeToSize[fieldType];
  }
  return result as T;
}
