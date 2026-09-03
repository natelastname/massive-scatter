from pathlib import Path
import re

Path('viewer/src/binary-data.ts').write_text('''import type {RGBA} from './plot-style';

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
''')

Path('viewer/test/binary-data.test.ts').write_text('''import assert from 'node:assert/strict';
import test from 'node:test';

import {packColors, packFloat32, packPositions} from '../src/binary-data.ts';

test('packPositions creates interleaved viewport-local GPU coordinates', () => {
  assert.deepEqual(
    Array.from(packPositions([10, 20], [30, 40], -2, -3)),
    [8, 27, 18, 37],
  );
});

test('packColors and packFloat32 create one attribute value per primitive', () => {
  assert.deepEqual(
    Array.from(packColors(2, index => (index ? [5, 6, 7, 8] : [1, 2, 3, 4]))),
    [1, 2, 3, 4, 5, 6, 7, 8],
  );
  assert.deepEqual(Array.from(packFloat32(3, index => index + 0.5)), [0.5, 1.5, 2.5]);
});
''')

path = Path('viewer/src/main.ts')
text = path.read_text()
text = text.replace(
    "import {aggregateCellCorner} from './lod-cell';\n",
    "import {packColors, packFloat32, packPositions, type BinaryLayerData} from './binary-data';\n",
)
text = text.replace("import {finiteRangeBy, type NumericRange} from './range';", "import {type NumericRange} from './range';")
text = re.sub(
    r"interface PlotDatum \{.*?\n\}\n\nconst AXIS_LEFT",
    '''type PrimitiveKind = 'point' | 'cell';

interface PickingBatch {
  layerId: string;
  kind: PrimitiveKind;
  response: PrimitiveViewResponse;
  level?: number;
}

const AXIS_LEFT''',
    text,
    flags=re.S,
)
text = text.replace(
    "const currentColorRanges = new Map<string, NumericRange | null>();",
    "const currentColorRanges = new Map<string, NumericRange | null>();\nconst pickingBatches = new Map<string, PickingBatch>();",
)
text = text.replace(
    "  getTooltip: (info: PickingInfo<PlotDatum>) => tooltip(info),",
    "  getTooltip: (info: PickingInfo) => tooltip(info),",
)

start = text.index('function responseData(')
end = text.index('function makeDeckLayers(')
replacement = r'''function responseCount(response: PrimitiveViewResponse, index: number): number {
  return response.count?.[index] ?? 1;
}

function legacyValueAt(response: PrimitiveViewResponse, index: number): number {
  return response.color?.[index] ?? response.count?.[index] ?? 1;
}

function encodingValueAt(
  encoding: EncodingManifest,
  response: PrimitiveViewResponse,
  index: number,
  aggregate: boolean,
): Scalar {
  if (encoding.kind === 'constant') return encoding.value ?? null;
  if (encoding.kind === 'count') return responseCount(response, index);
  if (aggregate) {
    return encoding.aggregate
      ? response.aggregates?.[encoding.aggregate]?.[index] ?? null
      : null;
  }
  return encoding.source ? response.fields?.[encoding.source]?.[index] ?? null : null;
}

function aggregateDefinition(
  layer: LayerManifest,
  key: string | undefined,
): AggregateDefinition | undefined {
  if (!key) return undefined;
  return layer.aggregates?.find(item => item.key === key);
}

function finiteResponseRange(
  response: PrimitiveViewResponse,
  valueAt: (index: number) => number | null | undefined,
  fallback: NumericRange = [0, 1],
): NumericRange {
  let minimum = Infinity;
  let maximum = -Infinity;
  let found = false;
  for (let index = 0; index < response.x.length; index += 1) {
    const value = valueAt(index);
    if (value === null || value === undefined || !Number.isFinite(value)) continue;
    minimum = Math.min(minimum, value);
    maximum = Math.max(maximum, value);
    found = true;
  }
  return found ? [minimum, maximum] : fallback;
}

function encodingRange(
  layer: LayerManifest,
  encoding: EncodingManifest,
  response: PrimitiveViewResponse,
  aggregate: boolean,
): NumericRange {
  if (!layer.plot) {
    return finiteResponseRange(response, index => legacyValueAt(response, index));
  }
  if (encoding.kind === 'constant') return [0, 1];
  if (encoding.kind === 'count') {
    return finiteResponseRange(response, index => responseCount(response, index));
  }
  const sourceRange = encoding.source
    ? layer.plot.numeric_ranges[encoding.source]
    : undefined;
  const reducer = aggregateDefinition(layer, encoding.aggregate)?.reducer;
  if (
    sourceRange &&
    (!aggregate || reducer === 'mean' || reducer === 'min' || reducer === 'max')
  ) {
    return sourceRange;
  }
  return finiteResponseRange(
    response,
    index => {
      const value = encodingValueAt(encoding, response, index, aggregate);
      return typeof value === 'number' && Number.isFinite(value) ? value : null;
    },
    sourceRange ?? [0, 1],
  );
}

function mergeNumericRange(
  left: NumericRange | null,
  right: NumericRange,
): NumericRange {
  return left
    ? [Math.min(left[0], right[0]), Math.max(left[1], right[1])]
    : right;
}

interface RangeBatch {
  response: PrimitiveViewResponse;
  aggregate: boolean;
}

function combinedLegacyRange(batches: RangeBatch[]): NumericRange {
  let result: NumericRange | null = null;
  for (const batch of batches) {
    if (batch.response.x.length === 0) continue;
    result = mergeNumericRange(
      result,
      finiteResponseRange(batch.response, index => legacyValueAt(batch.response, index)),
    );
  }
  return result ?? [0, 1];
}

function combinedEncodingRange(
  layer: LayerManifest,
  encoding: EncodingManifest,
  batches: RangeBatch[],
): NumericRange {
  let result: NumericRange | null = null;
  for (const batch of batches) {
    if (batch.response.x.length === 0) continue;
    result = mergeNumericRange(
      result,
      encodingRange(layer, encoding, batch.response, batch.aggregate),
    );
  }
  return result ?? [0, 1];
}

function normalized(value: number, range: NumericRange, low = 0, high = 1): number {
  if (range[0] === range[1]) return high;
  const t = Math.max(0, Math.min(1, (value - range[0]) / (range[1] - range[0])));
  return low + (high - low) * t;
}

function colorAt(
  layer: LayerManifest,
  response: PrimitiveViewResponse,
  index: number,
  aggregate: boolean,
  colorRange: NumericRange | null,
  alphaRange: NumericRange | null,
): RGBA {
  if (!layer.plot) {
    const range = colorRange ?? [0, 1];
    return colorMap(legacyValueAt(response, index), range[0], range[1]);
  }
  const scatter = layer.plot.scatter;
  let color: RGBA;
  if (scatter.color.kind === 'constant') {
    color = parseColor(String(scatter.color.value ?? 'black'));
  } else {
    const range = colorRange ?? [0, 1];
    const raw = encodingValueAt(scatter.color, response, index, aggregate);
    const value = typeof raw === 'number' ? raw : range[0];
    color = colorMap(value, range[0], range[1], scatter.cmap);
  }

  const alphaEncoding = scatter.alpha;
  if (alphaEncoding.kind === 'constant') {
    const alpha = Number(alphaEncoding.value ?? 1);
    return withAlpha(color, Number.isFinite(alpha) ? alpha : 1);
  }
  const rawAlpha = encodingValueAt(alphaEncoding, response, index, aggregate);
  if (typeof rawAlpha !== 'number') return color;
  return withAlpha(color, normalized(rawAlpha, alphaRange ?? [0, 1], 0.15, 1));
}

function markerAt(
  layer: LayerManifest,
  response: PrimitiveViewResponse,
  index: number,
): string {
  if (!layer.plot) return 'circle';
  const encoding = layer.plot.scatter.marker;
  if (encoding.kind === 'constant') return String(encoding.value ?? 'circle');
  if (encoding.kind !== 'field' || !encoding.source) return 'circle';
  const value = String(response.fields?.[encoding.source]?.[index] ?? '');
  const categories = layer.plot.categorical_fields[encoding.source] ?? [];
  const categoryIndex = Math.max(0, categories.indexOf(value));
  return markerForCategory(categoryIndex);
}

function sizeAt(
  layer: LayerManifest,
  response: PrimitiveViewResponse,
  index: number,
): number {
  if (!layer.plot) return 3;
  const encoding = layer.plot.scatter.size;
  if (encoding.kind === 'constant') {
    const value = Number(encoding.value ?? 3);
    return Number.isFinite(value) ? Math.max(2, value) : 3;
  }
  if (encoding.kind !== 'field' || !encoding.source) return 3;
  const raw = response.fields?.[encoding.source]?.[index];
  if (typeof raw !== 'number') return 3;
  const range = layer.plot.numeric_ranges[encoding.source] ?? [raw, raw];
  return normalized(raw, range, 3, 14);
}

function positionAttribute(
  response: PrimitiveViewResponse,
  offsetX = 0,
  offsetY = 0,
) {
  return {value: packPositions(response.x, response.y, offsetX, offsetY), size: 2};
}

function colorAttribute(
  layer: LayerManifest,
  response: PrimitiveViewResponse,
  aggregate: boolean,
  colorRange: NumericRange | null,
  alphaRange: NumericRange | null,
) {
  return {
    value: packColors(response.x.length, index =>
      colorAt(layer, response, index, aggregate, colorRange, alphaRange),
    ),
    size: 4,
  };
}

function registerPicking(
  id: string,
  layerId: string,
  kind: PrimitiveKind,
  response: PrimitiveViewResponse,
  level?: number,
) {
  pickingBatches.set(id, {layerId, kind, response, level});
}

'''
text = text[:start] + replacement + text[end:]

start = text.index('function makeDeckLayers(')
end = text.index('function renderLayers(')
replacement = r'''function makeDeckLayers(response: LayerViewResponse): Layer[] {
  const layer = layerManifest(response.id);
  const rangeBatches: RangeBatch[] = [
    {response: response.points, aggregate: false},
    ...response.cells.map(batch => ({response: batch, aggregate: true})),
  ];

  let colorRange: NumericRange | null;
  let alphaRange: NumericRange | null = null;
  if (!layer.plot) {
    colorRange = combinedLegacyRange(rangeBatches);
  } else {
    const scatter = layer.plot.scatter;
    colorRange =
      scatter.color.kind === 'constant'
        ? null
        : combinedEncodingRange(layer, scatter.color, rangeBatches);
    if (scatter.alpha.kind !== 'constant') {
      alphaRange = combinedEncodingRange(layer, scatter.alpha, rangeBatches);
    }
  }
  currentColorRanges.set(layer.id, colorRange);

  const result: Layer[] = [];
  for (const batch of response.cells) {
    if (batch.x.length === 0) continue;
    const id = `cells-${layer.id}-lod-${batch.level}`;
    const half = batch.cell_size / 2;
    const data: BinaryLayerData = {
      length: batch.x.length,
      attributes: {
        getPosition: positionAttribute(batch, -half, -half),
        getFillColor: colorAttribute(
          layer,
          batch,
          true,
          colorRange,
          alphaRange,
        ),
      },
    };
    registerPicking(id, layer.id, 'cell', batch, batch.level);
    result.push(
      new GridCellLayer({
        id,
        data,
        coordinateSystem: COORDINATE_SYSTEM.CARTESIAN,
        cellSize: batch.cell_size,
        coverage: 1,
        extruded: false,
        opacity: 1,
        pickable: true,
      }),
    );
  }

  const points = response.points;
  if (points.x.length === 0) return result;
  if (layer.plot) {
    const id = `points-styled-${layer.id}`;
    const data: BinaryLayerData = {
      length: points.x.length,
      attributes: {
        getPosition: positionAttribute(points),
        getColor: colorAttribute(
          layer,
          points,
          false,
          colorRange,
          alphaRange,
        ),
        getSize: {
          value: packFloat32(points.x.length, index => sizeAt(layer, points, index)),
          size: 1,
        },
      },
    };
    registerPicking(id, layer.id, 'point', points);
    result.push(
      new IconLayer({
        id,
        data,
        coordinateSystem: COORDINATE_SYSTEM.CARTESIAN,
        getIcon: (_datum, {index}) => markerIcon(markerAt(layer, points, index)),
        sizeUnits: 'pixels',
        sizeMinPixels: 2,
        sizeMaxPixels: 24,
        pickable: true,
      }),
    );
  } else {
    const id = `points-native-${layer.id}`;
    const data: BinaryLayerData = {
      length: points.x.length,
      attributes: {
        getPosition: positionAttribute(points),
        getFillColor: colorAttribute(
          layer,
          points,
          false,
          colorRange,
          alphaRange,
        ),
      },
    };
    registerPicking(id, layer.id, 'point', points);
    result.push(
      new ScatterplotLayer({
        id,
        data,
        coordinateSystem: COORDINATE_SYSTEM.CARTESIAN,
        getRadius: 0.42,
        radiusUnits: 'common',
        radiusMinPixels: 1.35,
        radiusMaxPixels: 5,
        opacity: 0.92,
        stroked: false,
        pickable: true,
      }),
    );
  }
  return result;
}

'''
text = text[:start] + replacement + text[end:]
text = text.replace(
    "  currentColorRanges.clear();\n  renderOrigin = response.origin;",
    "  currentColorRanges.clear();\n  pickingBatches.clear();\n  renderOrigin = response.origin;",
)

start = text.index('function tooltip(')
end = text.index('function escapeHtml(')
replacement = r'''function tooltip(info: PickingInfo) {
  if (!manifest || info.index < 0 || !info.layer) return null;
  const picked = pickingBatches.get(info.layer.id);
  if (!picked) return null;
  const layer = layerManifest(picked.layerId);
  const response = picked.response;
  const index = info.index;
  const relativeX = response.x[index];
  const relativeY = response.y[index];
  if (relativeX === undefined || relativeY === undefined) return null;
  const [worldX, worldY] = localToWorld(renderOrigin, [relativeX, relativeY]);
  const absoluteX = addIntegerOffset(manifest.origin.x, worldX);
  const absoluteY = addIntegerOffset(manifest.origin.y, worldY);
  const lines: string[] = [];
  const label = layer.plot?.scatter.label;
  if (label) lines.push(escapeHtml(label));
  lines.push(`x: ${escapeHtml(absoluteX)}`, `y: ${escapeHtml(absoluteY)}`);

  if (picked.kind === 'cell') {
    if (picked.level !== undefined) lines.push(`LOD: ${picked.level}`);
    lines.push(`count: ${integerFormat(responseCount(response, index))}`);
    for (const definition of layer.aggregates ?? []) {
      const value = response.aggregates?.[definition.key]?.[index];
      if (value !== undefined) {
        lines.push(
          `${escapeHtml(definition.reducer)}(${escapeHtml(definition.source)}): ` +
          escapeHtml(compactFormat(value)),
        );
      }
    }
  } else if (layer.plot) {
    for (const [name, values] of Object.entries(response.fields ?? {})) {
      const value = values[index];
      lines.push(`${escapeHtml(name)}: ${escapeHtml(String(value))}`);
    }
  } else {
    lines.push(`value: ${escapeHtml(String(legacyValueAt(response, index)))}`);
  }
  return {html: lines.join('<br/>')};
}

'''
text = text[:start] + replacement + text[end:]
path.write_text(text)
