export interface OrthographicState {
  target: [number, number, number];
  zoom: number;
}

export type Origin = readonly [number, number];

/**
 * Convert the logical dataset-relative camera into the response-local frame
 * used by deck.gl. Both camera and point positions must use the same frame.
 */
export function toRenderViewState(
  worldState: OrthographicState,
  origin: Origin,
): OrthographicState {
  return {
    target: [
      worldState.target[0] - origin[0],
      worldState.target[1] - origin[1],
      worldState.target[2],
    ],
    zoom: worldState.zoom,
  };
}

/** Convert a deck.gl camera interaction back into dataset-relative space. */
export function toWorldViewState(
  renderState: OrthographicState,
  origin: Origin,
): OrthographicState {
  return {
    target: [
      renderState.target[0] + origin[0],
      renderState.target[1] + origin[1],
      renderState.target[2],
    ],
    zoom: renderState.zoom,
  };
}

/** Reconstruct a dataset-relative point from a response-local position. */
export function localToWorld(
  origin: Origin,
  position: readonly [number, number],
): [number, number] {
  return [origin[0] + position[0], origin[1] + position[1]];
}
