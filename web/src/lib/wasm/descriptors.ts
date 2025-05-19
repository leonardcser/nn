import { type Point, FieldType, type StructDescriptor } from './types';

// Example usage for Point:
export const pointDescriptor: StructDescriptor<Point> = {
  x: FieldType.Int32,
  y: FieldType.Int32,
};
