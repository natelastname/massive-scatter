import assert from 'node:assert/strict';
import test from 'node:test';

import {createPlotView} from '../src/plot-view.ts';

test('plot view uses conventional Cartesian y-up orientation', () => {
  const view = createPlotView();
  assert.equal(view.props.flipY, false);

  const viewport = view.makeViewport({
    width: 100,
    height: 100,
    viewState: {target: [0, 0, 0], zoom: 0},
  });
  assert.ok(viewport);

  const origin = viewport.project([0, 0, 0]);
  const diagonal = viewport.project([1, 1, 0]);

  // Screen pixel y increases downward, so a larger mathematical y coordinate
  // must project to a smaller screen y coordinate. Thus y=x rises visually.
  assert.ok(diagonal[0] > origin[0]);
  assert.ok(diagonal[1] < origin[1]);
});
