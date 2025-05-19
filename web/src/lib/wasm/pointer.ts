import type { WasmExports } from './types';

// We assume pointers are 32-bit (4 bytes) for now.
// This should match the size of a pointer in your WASM target.
const POINTER_SIZE_BYTES = 4;

/**
 * WPointer manages a memory location in WASM that holds a pointer value.
 * This is useful for scenarios like passing a pointer-to-a-pointer (e.g., Model**)
 * to a C function that allocates memory and writes the address into the provided location.
 *
 * T represents the type of the value that the pointer *points to* (e.g., number for an address).
 */
export class WPointer<T extends number = number> {
  protected exports: WasmExports;
  private _address: number; // The address of this pointer variable itself in WASM memory.
  // This is the value you pass to functions expecting a pointer-to-a-pointer.
  private memoryView: DataView;

  /**
   * @param exports The WASM exports object.
   * @param initialValue If provided, this value (e.g., an address) will be written into the allocated pointer memory.
   */
  constructor(exports: WasmExports, initialValue?: T) {
    this.exports = exports;
    this._address = this.exports.allocate_memory(POINTER_SIZE_BYTES);
    if (this._address === 0) {
      throw new Error('WPointer: Failed to allocate memory for the pointer holder.');
    }
    // It's generally safer to create the DataView on demand or ensure it's
    // always valid, especially if memory can grow. For simplicity here,
    // we create it once. If memory growth invalidates this, it needs to be refreshed.
    this.memoryView = new DataView(this.exports.memory.buffer);

    if (initialValue !== undefined) {
      this.set(initialValue);
    }
  }

  /**
   * Gets the value currently stored at this pointer's memory location.
   * For a Model**, this would be the Model* address.
   * @returns The pointer value (address) stored.
   */
  public get(): T {
    if (this.isFreed()) {
      throw new Error('WPointer.get(): Pointer has been freed.');
    }
    // Ensure DataView is accessing the potentially grown memory buffer
    if (this.memoryView.buffer !== this.exports.memory.buffer) {
      this.memoryView = new DataView(this.exports.memory.buffer);
    }
    return this.memoryView.getUint32(this._address, true) as T; // true for little-endian
  }

  /**
   * Sets the value stored at this pointer's memory location.
   * @param value The pointer value (address) to store.
   */
  public set(value: T): void {
    if (this.isFreed()) {
      throw new Error('WPointer.set(): Pointer has been freed.');
    }
    if (this.memoryView.buffer !== this.exports.memory.buffer) {
      this.memoryView = new DataView(this.exports.memory.buffer);
    }
    this.memoryView.setUint32(this._address, value, true); // true for little-endian
  }

  /**
   * Returns the address of this WPointer's memory location itself.
   * This is the address to pass to C functions expecting a pointer-to-a-pointer (e.g., `Model**`).
   * @returns The address of the pointer holder in WASM memory.
   */
  public ptr(): number {
    if (this.isFreed()) {
      throw new Error('WPointer.ptr(): Pointer has been freed.');
    }
    return this._address;
  }

  /**
   * Frees the WASM memory allocated for this pointer holder.
   * Note: This does NOT free the memory that the stored pointer value might point to.
   */
  public free(): void {
    if (!this.isFreed()) {
      this.exports.free_memory(this._address);
      this._address = 0;
    }
  }

  /**
   * Checks if the WASM memory for this pointer holder has been freed.
   * @returns True if freed, false otherwise.
   */
  public isFreed(): boolean {
    return this._address === 0;
  }
}
