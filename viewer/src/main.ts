import {
  COORDINATE_SYSTEM,
  Deck,
  OrthographicView,
  type PickingInfo,
  type ViewStateChangeParameters,
} from '@deck.gl/core';
import {ScatterplotLayer} from '@deck.gl/layers';
import {format as d3format} from 'd3-format';
import {scaleLinear} from 'd3-scale';
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

interface OrthographicState {
  target: [number, number, number];
  zoom: number;
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
const fitButton = requiredElement<HTMLButtonElement>('fit');
const maxPointsInput = requiredElement<HTMLInputElement>('max-points');

let manifest: Manifest;
let viewState: OrthographicState = {target: [0, 0, 0], zoom: 0};
let currentResponse: ViewResponse | null = null;
let requestTimer: number | undefined;
let activeRequest: AbortController | null = null;
const integerFormat = d3format(',');

const deck = new Deck({
  parent: plot,
  views: new OrthographicView({id: 'scatter', controller: true}),
  viewState,
  layers: [],
  controller: true,
  getTooltip: (info: PickingInfo<PlotDatum>) => tooltip(info),
  onViewStateChange: ({viewState: next}: ViewStateChangeParameters) => {
    viewState = next as OrthographicState;
    deck.setProps({viewState});
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
  const scale = 2 ** viewState.zoom;
  const halfWidth = plot.clientWidth / (2 * scale);
  const halfHeight = plot.clientHeight / (2 * scale);
  return {
    minX: viewState.target[0] - halfWidth,
    maxX: viewState.target[0] + halfWidth,
    minY: viewState.target[1] - halfHeight,
    maxY: viewState.target[1] + halfHeight,
  };
}

function fit() {
  if (!manifest) return;
  const availableWidth = Math.max(1, plot.clientWidth - 100);
  const availableHeight = Math.max(1, plot.clientHeight - 80);
  const worldWidth = Math.max(1, manifest.extent.width - 1);
  const worldHeight = Math.max(1, manifest.extent.height - 1);
  const scale = Math.min(availableWidth / worldWidth, availableHeight / worldHeight);
  viewState = {
    target: [worldWidth / 2, worldHeight / 2, 0],
    zoom: Math.log2(Math.max(scale, Number.MIN_VALUE)),
  };
  deck.setProps({viewState});
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
  activeRequest = new AbortController();
  const bounds = visibleBounds();
  const query = new URLSearchParams({
    xmin: String(bounds.minX),
    xmax: String(bounds.maxX),
    ymin: String(bounds.minY),
    ymax: String(bounds.maxY),
    width: String(plot.clientWidth),
    height: String(plot.clientHeight),
    max_points: String(Math.max(1, Number(maxPointsInput.value) || 200_000)),
    max_cells: '200000',
  });
  status.textContent = 'loading viewport…';

  try {
    const response = await fetch(`/api/view?${query}`, {signal: activeRequest.signal});
    if (!response.ok) throw new Error(`${response.status} ${await response.text()}`);
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
  const radius = aggregate ? (response.cell_size ?? 1) * 0.46 : 0.42;

  deck.setProps({
    layers: [
      new ScatterplotLayer<PlotDatum>({
        id: `points-${response.mode}-${response.level ?? 'native'}`,
        data,
        coordinateSystem: COORDINATE_SYSTEM.CARTESIAN,
        coordinateOrigin: [response.origin[0], response.origin[1], 0],
        getPosition: datum => datum.position,
        getRadius: radius,
        radiusUnits: 'common',
        radiusMinPixels: aggregate ? 0.8 : 1.35,
        radiusMaxPixels: aggregate ? 72 : 5,
        getFillColor: datum => colorFor(datum.value, minValue, maxValue),
        opacity: aggregate ? 0.88 : 0.92,
        stroked: false,
        pickable: true,
      }),
    ],
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
  const relativeX = currentResponse.origin[0] + info.object.position[0];
  const relativeY = currentResponse.origin[1] + info.object.position[1];
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

fitButton.addEventListener('click', fit);
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
  fit();
}

void start().catch(error => {
  status.textContent = `startup failed: ${String(error)}`;
});
