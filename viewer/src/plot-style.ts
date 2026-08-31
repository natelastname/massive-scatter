export type RGBA = [number, number, number, number];

const NAMED_COLORS: Record<string, RGBA> = {
  black: [0, 0, 0, 255],
  white: [255, 255, 255, 255],
  red: [214, 39, 40, 255],
  blue: [31, 119, 180, 255],
  green: [44, 160, 44, 255],
  orange: [255, 127, 14, 255],
  purple: [148, 103, 189, 255],
  gray: [127, 127, 127, 255],
  grey: [127, 127, 127, 255],
};

const COLORMAPS: Record<string, [number, number, number][]> = {
  viridis: [
    [68, 1, 84],
    [59, 82, 139],
    [33, 145, 140],
    [94, 201, 98],
    [253, 231, 37],
  ],
  plasma: [
    [13, 8, 135],
    [126, 3, 168],
    [204, 71, 120],
    [248, 149, 64],
    [240, 249, 33],
  ],
  magma: [
    [0, 0, 4],
    [81, 18, 124],
    [183, 55, 121],
    [252, 137, 97],
    [252, 253, 191],
  ],
};

const MARKERS = ['circle', 'square', 'triangle', 'diamond', 'x', 'plus'] as const;
export type MarkerName = (typeof MARKERS)[number];

const MARKER_ALIASES: Record<string, MarkerName> = {
  o: 'circle',
  circle: 'circle',
  s: 'square',
  square: 'square',
  '^': 'triangle',
  triangle: 'triangle',
  D: 'diamond',
  diamond: 'diamond',
  x: 'x',
  '+': 'plus',
  plus: 'plus',
};

export function parseColor(value: string): RGBA {
  const normalized = value.trim();
  const named = NAMED_COLORS[normalized.toLowerCase()];
  if (named) return [...named];
  const match = /^#([0-9a-f]{6}|[0-9a-f]{8})$/i.exec(normalized);
  if (!match) throw new Error(`Unsupported color ${JSON.stringify(value)}; use a CSS name or #RRGGBB[/AA].`);
  const hex = match[1]!;
  return [
    Number.parseInt(hex.slice(0, 2), 16),
    Number.parseInt(hex.slice(2, 4), 16),
    Number.parseInt(hex.slice(4, 6), 16),
    hex.length === 8 ? Number.parseInt(hex.slice(6, 8), 16) : 255,
  ];
}

export function colorMap(value: number, minimum: number, maximum: number, cmap = 'viridis'): RGBA {
  const anchors = COLORMAPS[cmap] ?? COLORMAPS.viridis!;
  const t = maximum === minimum ? 0.6 : Math.max(0, Math.min(1, (value - minimum) / (maximum - minimum)));
  const scaled = t * (anchors.length - 1);
  const left = Math.min(anchors.length - 2, Math.floor(scaled));
  const fraction = scaled - left;
  const a = anchors[left]!;
  const b = anchors[left + 1]!;
  return [
    Math.round(a[0] + (b[0] - a[0]) * fraction),
    Math.round(a[1] + (b[1] - a[1]) * fraction),
    Math.round(a[2] + (b[2] - a[2]) * fraction),
    255,
  ];
}

export function withAlpha(color: RGBA, alpha: number): RGBA {
  const factor = Math.max(0, Math.min(1, alpha));
  return [color[0], color[1], color[2], Math.round(color[3] * factor)];
}

export function normalizeMarker(value: string): MarkerName {
  return MARKER_ALIASES[value] ?? 'circle';
}

export function markerForCategory(index: number): MarkerName {
  return MARKERS[index % MARKERS.length]!;
}

function markerSvg(marker: MarkerName): string {
  const common = 'stroke="black" stroke-width="10" stroke-linecap="round" stroke-linejoin="round"';
  switch (marker) {
    case 'circle':
      return '<circle cx="50" cy="50" r="40" fill="black"/>';
    case 'square':
      return '<rect x="12" y="12" width="76" height="76" fill="black"/>';
    case 'triangle':
      return '<path d="M50 8 L94 90 L6 90 Z" fill="black"/>';
    case 'diamond':
      return '<path d="M50 5 L95 50 L50 95 L5 50 Z" fill="black"/>';
    case 'x':
      return `<path d="M16 16 L84 84 M84 16 L16 84" ${common} fill="none"/>`;
    case 'plus':
      return `<path d="M50 10 L50 90 M10 50 L90 50" ${common} fill="none"/>`;
  }
}

export interface IconDefinition {
  url: string;
  width: number;
  height: number;
  anchorX: number;
  anchorY: number;
  mask: boolean;
}

const ICON_CACHE = new Map<MarkerName, IconDefinition>();

export function markerIcon(marker: string): IconDefinition {
  const normalized = normalizeMarker(marker);
  const cached = ICON_CACHE.get(normalized);
  if (cached) return cached;
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">${markerSvg(normalized)}</svg>`;
  const icon: IconDefinition = {
    url: `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svg)}`,
    width: 100,
    height: 100,
    anchorX: 50,
    anchorY: 50,
    mask: true,
  };
  ICON_CACHE.set(normalized, icon);
  return icon;
}

export function gradientCss(cmap: string): string {
  const anchors = COLORMAPS[cmap] ?? COLORMAPS.viridis!;
  return `linear-gradient(to right, ${anchors.map(rgb => `rgb(${rgb.join(',')})`).join(', ')})`;
}
