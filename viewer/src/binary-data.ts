import type {RGBA} from './plot-style';

export type BinaryAttributeValue = Float32Array | Uint8Array;

export interface BinaryAttribute {
  value: BinaryAttributeValue;
  size: number;
}

export interface BinaryLayerData {
  length: number;
  attributes: Record<string, BinaryAttribute>;
}

export function packPositions(
  x: readonly number[],
  y: readonly number[],
  offsetX = 0,
  offsetY = 0,
): Float32Array {
  if (x.length !== y.length) throw new Error('x/y lengths differ');
  const result = new Float32Array(x.length * 2);
  for (let index = 0; index < x.length; index += 1) {
    result[index * 2] = (x[index] ?? 0) + offsetX;
    result[index * 2 + 1] = (y[index] ?? 0) + offsetY;
  }
  return result;
}

export function packColors(
  length: number,
  colorAt: (index: number) => RGBA,
): Uint8Array {
  const result = new Uint8Array(length * 4);
  for (let index = 0; index < length; index += 1) {
    const color = colorAt(index);
    result.set(color, index * 4);
  }
  return result;
}

export function packFloat32(
  length: number,
  valueAt: (index: number) => number,
): Float32Array {
  const result = new Float32Array(length);
  for (let index = 0; index < length; index += 1) {
    result[index] = valueAt(index);
  }
  return result;
}
