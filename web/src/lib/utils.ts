import { type ClassValue, clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function assert<T>(
  value: T | null | undefined,
  message: string
): asserts value is NonNullable<T> {
  if (!value) {
    throw new Error(`Assertion failed: ${message}`);
  }
}
