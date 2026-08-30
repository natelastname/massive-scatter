import {OrthographicView} from '@deck.gl/core';

/**
 * Create the 2D plot view using the conventional Cartesian orientation.
 *
 * deck.gl's OrthographicView defaults to flipY=true, which uses top-left
 * screen-style coordinates. Scientific plots need y to increase upward, so
 * keep flipY disabled explicitly rather than negating dataset coordinates.
 */
export function createPlotView(): OrthographicView {
  return new OrthographicView({
    id: 'scatter',
    controller: true,
    flipY: false,
  });
}
