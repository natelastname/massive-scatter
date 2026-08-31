import {
  COORDINATE_SYSTEM,
  Deck,
  type PickingInfo,
  type ViewStateChangeParameters,
} from '@deck.gl/core';
import {GridCellLayer, IconLayer, ScatterplotLayer} from '@deck.gl/layers';
import {format as d3format} from 'd3-format';
import {scaleLinear} from 'd3-scale';
import {
  localToWorld,
  toRenderViewState,
  toWorldViewState,
  type Origin,
  type OrthographicState,
} from './frame';
import {aggregateCellCorner} from './lod-cell';
import {createPlotView} from './plot-view';
import {
  colorMap,
  gradientCss,
  markerForCategory,
  markerIcon,
  parseColor,
  withAlpha,
  type RGBA,
} from './plot-style';
import './style.css';

type Scalar = string | number | boolean | null;

interface AggregateDefinition {
  key: string;
  source: string;
  storage: string;
  reducer: 'sum' | 'mean' | 'min' | 'max';
}

interface EncodingManifest {
  kind: 'constant' | 'field' | 'count';
  value?: string | number;
  source?: string;
  aggregate?: string;
}

interface ScatterManifest {
  type: 'scatter';
  color: EncodingManifest;
  marker: EncodingManifest;
  size: EncodingManifest;
  alpha: EncodingManifest;
  cmap: string;
  label: string | null;
}

interface PlotManifest {
  scatter: ScatterManifest;
  axes: {
    title: string | null;
    xlabel: string | null;
    ylabel: string | null;
    legend: boolean;
  };
  exact_fields: Record<string, string>;
  categorical_fields: Record<string, string[]>;
  numeric_ranges: Record<string, [number, number]>;
}

interface Manifest {
  point_count: number;
  origin: {x: string; y: string};
  extent: {width: number; height: number};
  base_cell_size: number;
  color_field: string | null;
  aggregates?: AggregateDefinition[];
  plot?: PlotManifest;
}

interface ViewResponse {
  mode: 'exact' | 'aggregate';
  origin: [number, number];
  x: number[];
  y: number[];
  color: number[] | null;
  fields?: Record<string, Scalar[]>;
  aggregates?: Record<string, number[]>;
  count?: number[];
  point_count?: number;
  cell_count?: number;
  cell_size?: number;
  level?: number;
}

interface PlotDatum {
  position: [number, number];
  legacyValue: number;
  count: number;
  fields: Record<string, Scalar>;
  aggregates: Record<string, number>;
}

const AXIS_LEFT = 72;
const AXIS_RIGHT = 18;
const AXIS_TOP = 18;
const AXIS_BOTTOM = 50;

const plot = requiredElement<HTMLDivElement>('plot');
const axes = requiredElement<SVGSVGElement>('axes');
const status = requiredElement<HTMLElement>('status');
const summary = requiredElement<HTMLElement>('dataset-summary');
const figureTitle = requiredElement<HTMLElement>('figure-title');
const legend = requiredElement<HTMLElement>('legend');
const homeButton = requiredElement<HTMLButtonElement>('home');
const maxPointsInput = requiredElement<HTMLInputElement>('max-points');

let manifest: Manifest;
let worldViewState: OrthographicState = {target: [0, 0, 0], zoom: 0};
let renderOrigin: Origin = [0, 0];
let currentResponse: ViewResponse | null = null;
let currentColorRange: [number, number] | null = null;
let requestTimer: number | undefined;
let activeRequest: AbortController | null = null;
const integerFormat = d3format(',');
const compactFormat = d3format('.5~g');

const deck = new Deck({
  parent: plot,
  views: createPlotView(),
  viewState: toRenderViewState(worldViewState, renderOrigin),
  layers: [],
  controller: true,
  getTooltip: (info: PickingInfo<PlotDatum>) => tooltip(info),
  onViewStateChange: ({viewState: next}: ViewStateChangeParameters) => {
    const renderViewState = next as OrthographicState;
    worldViewState = toWorldViewState(renderViewState, renderOrigin);
    deck.setProps({viewState: renderViewState});
    renderAxes();
    scheduleViewRequest();
  },
});

function requiredElement<T extends Element>(id: string): T {
  const element = document.getElementById(id);
  if (!element) throw new Error(`Missing #${id}`);
  return element as unknown as T;
}

function visibleBounds() {
  const scale = 2 ** worldViewState.zoom;
  const halfWidth = plot.clientWidth / (2 * scale);
  const halfHeight = plot.clientHeight / (2 * scale);
  return {
    minX: worldViewState.target[0] - halfWidth,
    maxX: worldViewState.target[0] + halfWidth,
    minY: worldViewState.target[1] - halfHeight,
    maxY: worldViewState.target[1] + halfHeight,
  };
}

function goHome() {
  if (!manifest) return;
  const availableWidth = Math.max(1, plot.clientWidth - 24);
  const availableHeight = Math.max(1, plot.clientHeight - 24);
  const worldWidth = Math.max(1, manifest.extent.width - 1);
  const worldHeight = Math.max(1, manifest.extent.height - 1);
  const scale = Math.min(availableWidth / worldWidth, availableHeight / worldHeight);
  worldViewState = {
    target: [worldWidth / 2, worldHeight / 2, 0],
    zoom: Math.log2(Math.max(scale, Number.MIN_VALUE)),
  };
  deck.setProps({viewState: toRenderViewState(worldViewState, renderOrigin)});
  renderAxes();
  scheduleViewRequest(0);
}

function scheduleViewRequest(delay = 75) {
  window.clearTimeout(requestTimer);
  requestTimer = window.setTimeout(() => void requestView(), delay);
}

async function requestView() {
  if (!manifest || plot.clientWidth < 1 || plot.clientHeight < 1) return;
  activeRequest?.abort();
  const controller = new AbortController();
  activeRequest = controller;
  const bounds = visibleBounds();
  const body = {
    xmin: bounds.minX,
    xmax: bounds.maxX,
    ymin: bounds.minY,
    ymax: bounds.maxY,
    width: plot.clientWidth,
    height: plot.clientHeight,
    max_points: Math.max(1, Number(maxPointsInput.value) || 200_000),
    max_cells: 200_000,
  };
  status.textContent = 'loading viewport…';

  try {
    const response = await fetch('/api/view', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(body),
      signal: controller.signal,
    });
    if (!response.ok) throw new Error(`${response.status} ${await response.text()}`);
    if (controller !== activeRequest) return;
    currentResponse = (await response.json()) as ViewResponse;
    renderLayer(currentResponse);
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') return;
    status.textContent = `request failed: ${String(error)}`;
  }
}

function responseData(response: ViewResponse): PlotDatum[] {
  const counts = response.count;
  const legacyValues = response.color ?? counts ?? response.x.map(() => 1);
  const fieldArrays = response.fields ?? {};
  const aggregateArrays = response.aggregates ?? {};
  return response.x.map((x, index) => {
    const fields: Record<string, Scalar> = {};
    for (const [name, values] of Object.entries(fieldArrays)) {
      fields[name] = values[index] ?? null;
    }
    const aggregateValues: Record<string, number> = {};
    for (const [name, values] of Object.entries(aggregateArrays)) {
      aggregateValues[name] = values[index] ?? 0;
    }
    return {
      position: [x, response.y[index] ?? 0],
      legacyValue: legacyValues[index] ?? 0,
      count: counts?.[index] ?? 1,
      fields,
      aggregates: aggregateValues,
    };
  });
}

function encodingValue(encoding: EncodingManifest, datum: PlotDatum, aggregate: boolean): Scalar {
  if (encoding.kind === 'constant') return encoding.value ?? null;
  if (encoding.kind === 'count') return datum.count;
  if (aggregate) {
    return encoding.aggregate ? datum.aggregates[encoding.aggregate] ?? null : null;
  }
  return encoding.source ? datum.fields[encoding.source] ?? null : null;
}

function numericValues(encoding: EncodingManifest, data: PlotDatum[], aggregate: boolean): number[] {
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

function aggregateDefinition(key: string | undefined): AggregateDefinition | undefined {
  if (!key) return undefined;
  return manifest.aggregates?.find(item => item.key === key);
}

function encodingRange(
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

function normalized(value: number, range: [number, number], low = 0, high = 1): number {
  if (range[0] === range[1]) return high;
  const t = Math.max(0, Math.min(1, (value - range[0]) / (range[1] - range[0])));
  return low + (high - low) * t;
}

function datumColor(datum: PlotDatum, data: PlotDatum[], aggregate: boolean): RGBA {
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

function datumMarker(datum: PlotDatum): string {
  if (!manifest.plot) return 'circle';
  const encoding = manifest.plot.scatter.marker;
  if (encoding.kind === 'constant') return String(encoding.value ?? 'circle');
  if (encoding.kind !== 'field' || !encoding.source) return 'circle';
  const value = String(datum.fields[encoding.source] ?? '');
  const categories = manifest.plot.categorical_fields[encoding.source] ?? [];
  const index = Math.max(0, categories.indexOf(value));
  return markerForCategory(index);
}

function datumSize(datum: PlotDatum): number {
  if (!manifest.plot) return 3;
  const encoding = manifest.plot.scatter.size;
  if (encoding.kind === 'constant') {
    const value = Number(encoding.value ?? 3);
    return Number.isFinite(value) ? Math.max(2, value) : 3;
  }
  if (encoding.kind !== 'field' || !encoding.source) return 3;
  const raw = datum.fields[encoding.source];
  if (typeof raw !== 'number') return 3;
  const range = manifest.plot.numeric_ranges[encoding.source] ?? [raw, raw];
  return normalized(raw, range, 3, 14);
}

function renderLayer(response: ViewResponse) {
  const data = responseData(response);
  const aggregate = response.mode === 'aggregate';
  if (manifest.plot?.scatter.color.kind === 'constant') {
    currentColorRange = null;
  } else if (manifest.plot) {
    currentColorRange = encodingRange(manifest.plot.scatter.color, data, aggregate);
  }
  renderOrigin = response.origin;
  const renderViewState = toRenderViewState(worldViewState, renderOrigin);

  const layer = aggregate
    ? new GridCellLayer<PlotDatum>({
        id: `cells-${response.level ?? 'aggregate'}`,
        data,
        coordinateSystem: COORDINATE_SYSTEM.CARTESIAN,
        cellSize: response.cell_size ?? 1,
        coverage: 1,
        extruded: false,
        getPosition: datum => aggregateCellCorner(datum.position, response.cell_size ?? 1),
        getFillColor: datum => datumColor(datum, data, true),
        opacity: 1,
        pickable: true,
      })
    : manifest.plot
      ? new IconLayer<PlotDatum>({
          id: 'points-styled',
          data,
          coordinateSystem: COORDINATE_SYSTEM.CARTESIAN,
          getPosition: datum => datum.position,
          getIcon: datum => markerIcon(datumMarker(datum)),
          getSize: datum => datumSize(datum),
          sizeUnits: 'pixels',
          sizeMinPixels: 2,
          sizeMaxPixels: 24,
          getColor: datum => datumColor(datum, data, false),
          pickable: true,
        })
      : new ScatterplotLayer<PlotDatum>({
          id: 'points-native',
          data,
          coordinateSystem: COORDINATE_SYSTEM.CARTESIAN,
          getPosition: datum => datum.position,
          getRadius: 0.42,
          radiusUnits: 'common',
          radiusMinPixels: 1.35,
          radiusMaxPixels: 5,
          getFillColor: datum => datumColor(datum, data, false),
          opacity: 0.92,
          stroked: false,
          pickable: true,
        });

  deck.setProps({viewState: renderViewState, layers: [layer]});
  renderLegend();

  const rendered = data.length;
  status.textContent = aggregate
    ? `LOD ${response.level} · ${integerFormat(rendered)} occupied cells · ${integerFormat(response.cell_size ?? 1)} units/cell`
    : `exact · ${integerFormat(rendered)} points · viewport-local GPU coordinates`;
}

function renderLegend() {
  legend.replaceChildren();
  const plotManifest = manifest.plot;
  if (!plotManifest?.axes.legend) {
    legend.hidden = true;
    return;
  }
  legend.hidden = false;
  const scatter = plotManifest.scatter;

  if (scatter.label) {
    const heading = document.createElement('div');
    heading.className = 'legend-heading';
    heading.textContent = scatter.label;
    legend.append(heading);
  }

  if (scatter.marker.kind === 'field' && scatter.marker.source) {
    const source = scatter.marker.source;
    const categories = plotManifest.categorical_fields[source] ?? [];
    for (const [index, category] of categories.entries()) {
      const row = document.createElement('div');
      row.className = 'legend-row';
      const swatch = document.createElement('span');
      swatch.className = 'marker-swatch';
      const icon = markerIcon(markerForCategory(index));
      swatch.style.maskImage = `url("${icon.url}")`;
      swatch.style.webkitMaskImage = `url("${icon.url}")`;
      row.append(swatch, document.createTextNode(category));
      legend.append(row);
    }
  }

  if (scatter.color.kind !== 'constant') {
    const label = document.createElement('div');
    label.className = 'legend-color-label';
    if (scatter.color.kind === 'count') {
      label.textContent = 'count';
    } else {
      const definition = aggregateDefinition(scatter.color.aggregate);
      label.textContent = definition
        ? `${definition.reducer}(${definition.source})`
        : scatter.color.source ?? 'color';
    }
    const strip = document.createElement('div');
    strip.className = 'legend-gradient';
    strip.style.backgroundImage = gradientCss(scatter.cmap);
    const ticks = document.createElement('div');
    ticks.className = 'legend-gradient-ticks';
    const range = currentColorRange;
    ticks.innerHTML = range
      ? `<span>${escapeHtml(compactFormat(range[0]))}</span><span>${escapeHtml(compactFormat(range[1]))}</span>`
      : '';
    legend.append(label, strip, ticks);
  }
}

function tooltip(info: PickingInfo<PlotDatum>) {
  if (!manifest || !currentResponse || !info.object) return null;
  const [relativeX, relativeY] = localToWorld(renderOrigin, info.object.position);
  const absoluteX = addIntegerOffset(manifest.origin.x, relativeX);
  const absoluteY = addIntegerOffset(manifest.origin.y, relativeY);
  const lines = [`x: ${escapeHtml(absoluteX)}`, `y: ${escapeHtml(absoluteY)}`];

  if (currentResponse.mode === 'aggregate') {
    lines.push(`count: ${integerFormat(info.object.count)}`);
    for (const definition of manifest.aggregates ?? []) {
      const value = info.object.aggregates[definition.key];
      if (value !== undefined) {
        lines.push(`${escapeHtml(definition.reducer)}(${escapeHtml(definition.source)}): ${escapeHtml(compactFormat(value))}`);
      }
    }
  } else if (manifest.plot) {
    for (const [name, value] of Object.entries(info.object.fields)) {
      lines.push(`${escapeHtml(name)}: ${escapeHtml(String(value))}`);
    }
  } else {
    lines.push(`value: ${escapeHtml(String(info.object.legacyValue))}`);
  }
  return {html: lines.join('<br/>')};
}

function escapeHtml(value: string): string {
  return value
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;');
}

function addIntegerOffset(origin: string, offset: number): string {
  return (BigInt(origin) + BigInt(Math.round(offset))).toLocaleString('en-US');
}

function renderAxes() {
  if (!manifest) return;
  const width = axes.clientWidth;
  const height = axes.clientHeight;
  if (width < 1 || height < 1) return;
  const bounds = visibleBounds();
  const xScale = scaleLinear().domain([bounds.minX, bounds.maxX]).range([AXIS_LEFT, width - AXIS_RIGHT]);
  const yScale = scaleLinear().domain([bounds.minY, bounds.maxY]).range([height - AXIS_BOTTOM, AXIS_TOP]);
  axes.replaceChildren();

  appendLine(AXIS_LEFT, height - AXIS_BOTTOM, width - AXIS_RIGHT, height - AXIS_BOTTOM, 'axis-line');
  appendLine(AXIS_LEFT, AXIS_TOP, AXIS_LEFT, height - AXIS_BOTTOM, 'axis-line');
  for (const tick of xScale.ticks(6)) {
    const x = xScale(tick);
    appendLine(x, height - AXIS_BOTTOM, x, height - AXIS_BOTTOM + 6, 'tick-line');
    appendText(x, height - AXIS_BOTTOM + 20, addIntegerOffset(manifest.origin.x, tick), 'middle', 'tick-label');
  }
  for (const tick of yScale.ticks(6)) {
    const y = yScale(tick);
    appendLine(AXIS_LEFT - 6, y, AXIS_LEFT, y, 'tick-line');
    appendText(AXIS_LEFT - 9, y + 4, addIntegerOffset(manifest.origin.y, tick), 'end', 'tick-label');
  }

  const axesManifest = manifest.plot?.axes;
  if (axesManifest?.xlabel) {
    appendText((AXIS_LEFT + width - AXIS_RIGHT) / 2, height - 6, axesManifest.xlabel, 'middle', 'axis-label');
  }
  if (axesManifest?.ylabel) {
    const text = appendText(15, (AXIS_TOP + height - AXIS_BOTTOM) / 2, axesManifest.ylabel, 'middle', 'axis-label');
    text.setAttribute('transform', `rotate(-90 15 ${(AXIS_TOP + height - AXIS_BOTTOM) / 2})`);
  }
}

function appendLine(x1: number, y1: number, x2: number, y2: number, className: string) {
  const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
  line.setAttribute('x1', String(x1));
  line.setAttribute('y1', String(y1));
  line.setAttribute('x2', String(x2));
  line.setAttribute('y2', String(y2));
  line.setAttribute('class', className);
  axes.append(line);
}

function appendText(
  x: number,
  y: number,
  value: string,
  anchor: 'start' | 'middle' | 'end',
  className: string,
): SVGTextElement {
  const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
  text.setAttribute('x', String(x));
  text.setAttribute('y', String(y));
  text.setAttribute('text-anchor', anchor);
  text.setAttribute('class', className);
  text.textContent = value;
  axes.append(text);
  return text;
}

homeButton.addEventListener('click', goHome);
window.addEventListener('keydown', event => {
  if (event.key === 'Home' && document.activeElement !== maxPointsInput) goHome();
});
maxPointsInput.addEventListener('change', () => scheduleViewRequest(0));
new ResizeObserver(() => {
  renderAxes();
  scheduleViewRequest(100);
}).observe(plot);

async function start() {
  const response = await fetch('/api/manifest');
  if (!response.ok) throw new Error(`${response.status} ${await response.text()}`);
  manifest = (await response.json()) as Manifest;
  const title = manifest.plot?.axes.title ?? 'massive-scatter';
  figureTitle.textContent = title;
  document.title = title;
  summary.textContent = `${integerFormat(manifest.point_count)} points · ${integerFormat(manifest.extent.width)} × ${integerFormat(manifest.extent.height)} units`;
  renderLegend();
  goHome();
}

void start().catch(error => {
  status.textContent = `startup failed: ${String(error)}`;
});
