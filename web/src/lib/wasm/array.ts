import type { WasmExports } from './types';

export type TypedArray = Int32Array | Uint32Array | Float32Array | Float64Array;

export type TypedArrayConstructor<T extends TypedArray = TypedArray> = {
  new (length: number): T;
  new (array: ArrayLike<number> | ArrayBufferLike): T;
  new (buffer: ArrayBufferLike, byteOffset: number, length?: number): T;
  BYTES_PER_ELEMENT: number;
  name: string; // For error messages
};

export class WArray<T extends TypedArray> {
  private _ptr: number;
  private _size: number; // Number of elements
  private _byteSize: number;
  protected exports: WasmExports;
  protected arrayConstructor: TypedArrayConstructor<T>;

  constructor(
    exports: WasmExports,
    arrayConstructor: TypedArrayConstructor<T>,
    size: number,
    initialData?: T
  ) {
    this.exports = exports;
    this.arrayConstructor = arrayConstructor;
    this._size = size;
    this._byteSize = size * this.arrayConstructor.BYTES_PER_ELEMENT;
    this._ptr = this.exports.allocate_memory(this._byteSize);

    if (initialData) {
      if (initialData.length !== size) {
        throw new Error(
          `Initial data length (${initialData.length}) does not match array size (${size}) for ${this.arrayConstructor.name}`
        );
      }
      this.set(initialData);
    }
  }

  public ptr(): number {
    return this._ptr;
  }

  public size(): number {
    return this._size;
  }

  public byteSize(): number {
    return this._byteSize;
  }

  public set(data: T, offsetElements: number = 0): void {
    if (offsetElements < 0 || offsetElements + data.length > this._size) {
      throw new Error(
        `Data (length ${data.length} at offset ${offsetElements}) exceeds array bounds (size ${this._size}) for ${this.arrayConstructor.name}`
      );
    }
    const byteOffset = this._ptr + offsetElements * this.arrayConstructor.BYTES_PER_ELEMENT;
    const bufferInWasm = new this.arrayConstructor(
      this.exports.memory.buffer,
      byteOffset,
      data.length
    );
    bufferInWasm.set(data);
  }

  public get(offsetElements: number = 0, lengthElements?: number): T {
    const actualLength = lengthElements ?? this._size - offsetElements;
    if (offsetElements < 0 || offsetElements + actualLength > this._size) {
      throw new Error(
        `Read (length ${actualLength} at offset ${offsetElements}) exceeds array bounds (size ${this._size}) for ${this.arrayConstructor.name}`
      );
    }
    const byteOffset = this._ptr + offsetElements * this.arrayConstructor.BYTES_PER_ELEMENT;
    // Create a copy
    const data = new this.arrayConstructor(
      this.exports.memory.buffer.slice(
        byteOffset,
        byteOffset + actualLength * this.arrayConstructor.BYTES_PER_ELEMENT
      )
    );
    return data;
  }

  public free(): void {
    if (this._ptr !== 0) {
      this.exports.free_memory(this._ptr);
      this._ptr = 0; // Mark as freed
    }
  }
}

export class WFloat32Array extends WArray<Float32Array> {
  constructor(exports: WasmExports, size: number, initialData?: Float32Array) {
    super(exports, Float32Array, size, initialData);
  }
}

export class WInt32Array extends WArray<Int32Array> {
  constructor(exports: WasmExports, size: number, initialData?: Int32Array) {
    super(exports, Int32Array, size, initialData);
  }
}

export class WUint32Array extends WArray<Uint32Array> {
  constructor(exports: WasmExports, size: number, initialData?: Uint32Array) {
    super(exports, Uint32Array, size, initialData);
  }
}

export class WFloat64Array extends WArray<Float64Array> {
  constructor(exports: WasmExports, size: number, initialData?: Float64Array) {
    super(exports, Float64Array, size, initialData);
  }
}
