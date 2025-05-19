import { WStruct } from './struct';
import { type Point, FieldType, type StructDescriptor, type WasmExports } from './types';

// Example usage for Point:
const pointDescriptor: StructDescriptor<Point> = {
  x: FieldType.Int32,
  y: FieldType.Int32,
};

export class WPoint extends WStruct<Point> {
  constructor(exports: WasmExports, initialData: Point) {
    super(exports, pointDescriptor, initialData);
  }
}
