export type NumericRange = [number, number];

export function finiteRangeBy<T>(
  values: readonly T[],
  valueOf: (value: T) => number | null | undefined,
  fallback: NumericRange = [0, 1],
): NumericRange {
  let minimum = Infinity;
  let maximum = -Infinity;
  let found = false;
  for (const item of values) {
    const value = valueOf(item);
    if (value === null || value === undefined || !Number.isFinite(value)) continue;
    minimum = Math.min(minimum, value);
    maximum = Math.max(maximum, value);
    found = true;
  }
  return found ? [minimum, maximum] : fallback;
}
