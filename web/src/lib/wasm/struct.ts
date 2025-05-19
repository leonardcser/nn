import { type WasmExports, type StructDescriptor } from './types';
import {
  calculateStructSize,
  serializeStruct,
  deserializeStruct,
  writeStructToMemory,
} from './memory';

export class WStruct<T extends Record<string, any>> {
  private _ptr: number;
  private structSize: number;

  constructor(
    private exports: WasmExports,
    private descriptor: StructDescriptor<T>,
    initialData: T
  ) {
    this.structSize = calculateStructSize(this.descriptor);
    this._ptr = serializeStruct(this.exports, initialData, this.descriptor);
    if (this._ptr === 0) {
      throw new Error('Failed to allocate Wasm memory for ManagedWasmStruct');
    }
  }

  public ptr(): number {
    if (this._ptr === 0) {
      throw new Error('Pointer is null, struct might have been freed.');
    }
    return this._ptr;
  }

  public data(): T {
    if (this._ptr === 0) {
      throw new Error('Cannot read from freed struct.');
    }
    return deserializeStruct(this.exports, this._ptr, this.descriptor);
  }

  public write(data: T): void {
    if (this._ptr === 0) {
      throw new Error('Cannot update freed struct.');
    }
    writeStructToMemory(this.exports, this._ptr, data, this.descriptor, this.structSize);
  }

  public free(): void {
    if (this._ptr !== 0) {
      this.exports.free_memory(this._ptr);
      this._ptr = 0; // Mark as freed
    }
  }

  public isFreed(): boolean {
    return this._ptr === 0;
  }
}
