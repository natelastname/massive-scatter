from pathlib import Path

path = Path("viewer/src/main.ts")
text = path.read_text()

text = text.replace(
    "import {aggregateCellCorner} from './lod-cell';\n",
    "import {aggregateCellCorner} from './lod-cell';\nimport {finiteRangeBy, type NumericRange} from './range';\n",
)

old = '''function numericValues(encoding: EncodingManifest, data: PlotDatum[], aggregate: boolean): number[] {
  const result: number[] = [];
  for (const datum of data) {
    const value = encodingValue(encoding, datum, aggregate);
    if (typeof value === 'number' && Number.isFinite(value)) result.push(value);
  }
  return result;
}

function finiteRange(values: number[], fallback: [number, number] = [0, 1]): [number, number] {
  if (values.length === 0) return fallback;
  let minimum = Infinity;
  let maximum = -Infinity;
  for (const value of values) {
    minimum = Math.min(minimum, value);
    maximum = Math.max(maximum, value);
  }
  return [minimum, maximum];
}

'''
if old not in text:
    raise SystemExit("numericValues/finiteRange block not found")
text = text.replace(old, "")

old = '''function encodingRange(
  encoding: EncodingManifest,
  data: PlotDatum[],
  aggregate: boolean,
): [number, number] {
  if (!manifest.plot) return finiteRange(data.map(datum => datum.legacyValue));
  if (encoding.kind === 'constant') return [0, 1];
  if (encoding.kind === 'count') return finiteRange(data.map(datum => datum.count));
  const sourceRange = encoding.source ? manifest.plot.numeric_ranges[encoding.source] : undefined;
  const reducer = aggregateDefinition(encoding.aggregate)?.reducer;
  if (sourceRange && (!aggregate || reducer === 'mean' || reducer === 'min' || reducer === 'max')) {
    return sourceRange;
  }
  return finiteRange(numericValues(encoding, data, aggregate), sourceRange ?? [0, 1]);
}
'''
new = '''function encodingRange(
  encoding: EncodingManifest,
  data: PlotDatum[],
  aggregate: boolean,
): NumericRange {
  if (!manifest.plot) return finiteRangeBy(data, datum => datum.legacyValue);
  if (encoding.kind === 'constant') return [0, 1];
  if (encoding.kind === 'count') return finiteRangeBy(data, datum => datum.count);
  const sourceRange = encoding.source ? manifest.plot.numeric_ranges[encoding.source] : undefined;
  const reducer = aggregateDefinition(encoding.aggregate)?.reducer;
  if (sourceRange && (!aggregate || reducer === 'mean' || reducer === 'min' || reducer === 'max')) {
    return sourceRange;
  }
  return finiteRangeBy(
    data,
    datum => {
      const value = encodingValue(encoding, datum, aggregate);
      return typeof value === 'number' && Number.isFinite(value) ? value : null;
    },
    sourceRange ?? [0, 1],
  );
}
'''
if old not in text:
    raise SystemExit("encodingRange block not found")
text = text.replace(old, new)

old = '''function datumColor(datum: PlotDatum, data: PlotDatum[], aggregate: boolean): RGBA {
  if (!manifest.plot) {
    const range = finiteRange(data.map(item => item.legacyValue));
    return colorMap(datum.legacyValue, range[0], range[1]);
  }
  const scatter = manifest.plot.scatter;
  let color: RGBA;
  if (scatter.color.kind === 'constant') {
    color = parseColor(String(scatter.color.value ?? 'black'));
    currentColorRange = null;
  } else {
    const range = encodingRange(scatter.color, data, aggregate);
    currentColorRange = range;
    const raw = encodingValue(scatter.color, datum, aggregate);
    const value = typeof raw === 'number' ? raw : range[0];
    color = colorMap(value, range[0], range[1], scatter.cmap);
  }

  const alphaEncoding = scatter.alpha;
  if (alphaEncoding.kind === 'constant') {
    const alpha = Number(alphaEncoding.value ?? 1);
    return withAlpha(color, Number.isFinite(alpha) ? alpha : 1);
  }
  const rawAlpha = encodingValue(alphaEncoding, datum, aggregate);
  if (typeof rawAlpha !== 'number') return color;
  const alphaRange = encodingRange(alphaEncoding, data, aggregate);
  return withAlpha(color, normalized(rawAlpha, alphaRange, 0.15, 1));
}
'''
new = '''function datumColor(
  datum: PlotDatum,
  aggregate: boolean,
  colorRange: NumericRange | null,
  alphaRange: NumericRange | null,
): RGBA {
  if (!manifest.plot) {
    const range = colorRange ?? [0, 1];
    return colorMap(datum.legacyValue, range[0], range[1]);
  }
  const scatter = manifest.plot.scatter;
  let color: RGBA;
  if (scatter.color.kind === 'constant') {
    color = parseColor(String(scatter.color.value ?? 'black'));
  } else {
    const range = colorRange ?? [0, 1];
    const raw = encodingValue(scatter.color, datum, aggregate);
    const value = typeof raw === 'number' ? raw : range[0];
    color = colorMap(value, range[0], range[1], scatter.cmap);
  }

  const alphaEncoding = scatter.alpha;
  if (alphaEncoding.kind === 'constant') {
    const alpha = Number(alphaEncoding.value ?? 1);
    return withAlpha(color, Number.isFinite(alpha) ? alpha : 1);
  }
  const rawAlpha = encodingValue(alphaEncoding, datum, aggregate);
  if (typeof rawAlpha !== 'number') return color;
  return withAlpha(color, normalized(rawAlpha, alphaRange ?? [0, 1], 0.15, 1));
}
'''
if old not in text:
    raise SystemExit("datumColor block not found")
text = text.replace(old, new)

old = '''function renderLayer(response: ViewResponse) {
  const data = responseData(response);
  const aggregate = response.mode === 'aggregate';
  if (manifest.plot?.scatter.color.kind === 'constant') {
    currentColorRange = null;
  } else if (manifest.plot) {
    currentColorRange = encodingRange(manifest.plot.scatter.color, data, aggregate);
  }
  renderOrigin = response.origin;
'''
new = '''function renderLayer(response: ViewResponse) {
  const data = responseData(response);
  const aggregate = response.mode === 'aggregate';
  let colorRange: NumericRange | null;
  let alphaRange: NumericRange | null = null;
  if (!manifest.plot) {
    colorRange = finiteRangeBy(data, datum => datum.legacyValue);
  } else {
    const scatter = manifest.plot.scatter;
    colorRange = scatter.color.kind === 'constant' ? null : encodingRange(scatter.color, data, aggregate);
    if (scatter.alpha.kind !== 'constant') {
      alphaRange = encodingRange(scatter.alpha, data, aggregate);
    }
  }
  currentColorRange = colorRange;
  renderOrigin = response.origin;
'''
if old not in text:
    raise SystemExit("renderLayer prefix not found")
text = text.replace(old, new)

text = text.replace(
    "getFillColor: datum => datumColor(datum, data, true),",
    "getFillColor: datum => datumColor(datum, true, colorRange, alphaRange),",
)
text = text.replace(
    "getColor: datum => datumColor(datum, data, false),",
    "getColor: datum => datumColor(datum, false, colorRange, alphaRange),",
)

if "datumColor(datum, data" in text:
    raise SystemExit("per-datum full-data color call remains")
if "finiteRange(data.map" in text:
    raise SystemExit("temporary full-range map remains")

path.write_text(text)

Path("viewer/src/range.ts").write_text(
    '''export type NumericRange = [number, number];

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
'''
)

Path("viewer/test/range.test.ts").write_text(
    '''import assert from 'node:assert/strict';
import test from 'node:test';

import {finiteRangeBy} from '../src/range.ts';

test('finiteRangeBy scans each datum once', () => {
  const values = Array.from({length: 10_000}, (_, index) => ({value: index - 5000}));
  let visits = 0;
  const range = finiteRangeBy(values, item => {
    visits += 1;
    return item.value;
  });
  assert.deepEqual(range, [-5000, 4999]);
  assert.equal(visits, values.length);
});

test('finiteRangeBy ignores non-finite values and uses fallback for empty data', () => {
  assert.deepEqual(finiteRangeBy([1, Number.NaN, 7, Number.POSITIVE_INFINITY], value => value), [1, 7]);
  assert.deepEqual(finiteRangeBy([], value => Number(value), [4, 9]), [4, 9]);
});
'''
)
