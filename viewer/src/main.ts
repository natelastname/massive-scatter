import {
  COORDINATE_SYSTEM,
  Deck,
  type PickingInfo,
  type ViewStateChangeParameters,
} from '@deck.gl/core';
import {GridCellLayer, ScatterplotLayer} from '@deck.gl/layers';
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
import './style.css';

interface Manifest {
  point_count: number;
  origin: {x: string; y: string};
  extent: {width: number; height: number};
  base_cell_size: number;
  color_field: string | null;
}

interface ViewResponse {
  mode: 'exact' | 'aggregate';
  origin: [number, number];
  x: number[];
  y: number[];
  color: number[] | null;
  count?: number[];
  point_count?: number;
  cell_count?: number;
  cell_size?: number;
  level?: number;
}

interface PlotDatum {
  position: [number, number];
  value: number;
  count: number;
}

const plot = requiredElement<HTMLDivElement>('plot');
const axes = requiredElement<SVGSVGElement>('axes');
const status = requiredElement<HTMLElement>('status');
const summary = requiredElement<HTMLElement>('dataset-summary');
const homeButton = requiredElement<HTMLButtonElement>('home');
const maxPointsInput = requiredElement<HTMLInputElement>('max-points');

let manifest: Manifest;
let worldViewState: OrthographicState = {target: [0, 0, 0], zoom: 0};
let renderOrigin: Origin = [0, 0];
let currentResponse: ViewResponse | null = null;
let requestTimer: number | undefined;
let activeRequest: AbortController | null = null;
const integerFormat = d3format(',');

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
  // The deck canvas already occupies exactly the interior axes rectangle.
  // Keep only a small visual pad so edge markers are not clipped by the frame.
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

function renderLayer(response: ViewResponse) {
  const counts = response.count;
  const values = response.color ?? counts ?? response.x.map(() => 1);
  let minValue = Infinity;
  let maxValue = -Infinity;
  for (const value of values) {
    minValue = Math.min(minValue, value);
    maxValue = Math.max(maxValue, value);
  }
  if (!Number.isFinite(minValue)) {
    minValue = 0;
    maxValue = 1;
  }

  const data: PlotDatum[] = response.x.map((x, index) => ({
    position: [x, response.y[index] ?? 0],
    value: values[index] ?? 0,
    count: counts?.[index] ?? 1,
  }));
  const aggregate = response.mode === 'aggregate';

  // The response positions are local to response.origin. Rebase the camera into
  // the same local frame before replacing the layer. This keeps a change in LOD
  // or viewport origin from translating the rendered data and keeps large world
  // coordinates out of GPU float32 state.
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
        getPosition: datum =>
          aggregateCellCorner(datum.position, response.cell_size ?? 1),
        getFillColor: datum => colorFor(datum.value, minValue, maxValue),
        opacity: 0.88,
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
        getFillColor: datum => colorFor(datum.value, minValue, maxValue),
        opacity: 0.92,
        stroked: false,
        pickable: true,
      });

  deck.setProps({
    viewState: renderViewState,
    layers: [layer],
  });

  const rendered = data.length;
  status.textContent = aggregate
    ? `LOD ${response.level} · ${integerFormat(rendered)} occupied cells · ${integerFormat(response.cell_size ?? 1)} units/cell`
    : `exact · ${integerFormat(rendered)} points · viewport-local GPU coordinates`;
}

function colorFor(value: number, minValue: number, maxValue: number): [number, number, number, number] {
  const t = maxValue === minValue ? 0.6 : Math.max(0, Math.min(1, (value - minValue) / (maxValue - minValue)));
  const anchors: [number, number, number][] = [
    [68, 1, 84],
    [59, 82, 139],
    [33, 145, 140],
    [94, 201, 98],
    [253, 231, 37],
  ];
  const scaled = t * (anchors.length - 1);
  const left = Math.min(anchors.length - 2, Math.floor(scaled));
  const fraction = scaled - left;
  const a = anchors[left] ?? anchors[0]!;
  const b = anchors[left + 1] ?? anchors[anchors.length - 1]!;
  return [
    Math.round(a[0] + (b[0] - a[0]) * fraction),
    Math.round(a[1] + (b[1] - a[1]) * fraction),
    Math.round(a[2] + (b[2] - a[2]) * fraction),
    255,
  ];
}

function tooltip(info: PickingInfo<PlotDatum>) {
  if (!manifest || !currentResponse || !info.object) return null;
  const [relativeX, relativeY] = localToWorld(renderOrigin, info.object.position);
  const absoluteX = addIntegerOffset(manifest.origin.x, relativeX);
  const absoluteY = addIntegerOffset(manifest.origin.y, relativeY);
  const countLine = currentResponse.mode === 'aggregate' ? `<br/>count: ${integerFormat(info.object.count)}` : '';
  return {html: `x: ${absoluteX}<br/>y: ${absoluteY}${countLine}<br/>value: ${info.object.value}`};
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
  const xScale = scaleLinear().domain([bounds.minX, bounds.maxX]).range([52, width - 18]);
  const yScale = scaleLinear().domain([bounds.minY, bounds.maxY]).range([height - 30, 18]);
  axes.replaceChildren();

  appendLine(52, height - 30, width - 18, height - 30, 'axis-line');
  appendLine(52, 18, 52, height - 30, 'axis-line');
  for (const tick of xScale.ticks(6)) {
    const x = xScale(tick);
    appendLine(x, height - 30, x, height - 24, 'tick-line');
    appendText(x, height - 8, addIntegerOffset(manifest.origin.x, tick), 'middle');
  }
  for (const tick of yScale.ticks(6)) {
    const y = yScale(tick);
    appendLine(46, y, 52, y, 'tick-line');
    appendText(43, y + 4, addIntegerOffset(manifest.origin.y, tick), 'end');
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

function appendText(x: number, y: number, value: string, anchor: 'start' | 'middle' | 'end') {
  const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
  text.setAttribute('x', String(x));
  text.setAttribute('y', String(y));
  text.setAttribute('text-anchor', anchor);
  text.setAttribute('class', 'tick-label');
  text.textContent = value;
  axes.append(text);
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
  summary.textContent = `${integerFormat(manifest.point_count)} points · ${integerFormat(manifest.extent.width)} × ${integerFormat(manifest.extent.height)} units`;
  goHome();
}

void start().catch(error => {
  status.textContent = `startup failed: ${String(error)}`;
});
