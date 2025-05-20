import type { WasmExports } from './types';

let iota = 0;
const callback_table: ((data?: DataView) => void)[] = [];

export function register_callback(func: (data?: DataView) => void): number {
  const id = iota++;
  callback_table[id] = func;
  return id;
}

export function _callback(instance: WasmExports, id: number, data: number, data_size: number) {
  const view = new DataView(instance.memory.buffer, data, data_size);
  return callback_table[id](view);
}
