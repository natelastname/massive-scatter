export type Point2D = readonly [number, number];

/**
 * Convert the center coordinate returned by the aggregate API into the
 * bottom-left coordinate expected by deck.gl's GridCellLayer.
 */
export function aggregateCellCorner(
  center: Point2D,
  cellSize: number,
): [number, number] {
  if (!Number.isFinite(cellSize) || cellSize <= 0) {
    throw new RangeError('cellSize must be a finite positive number');
  }
  const half = cellSize / 2;
  return [center[0] - half, center[1] - half];
}
