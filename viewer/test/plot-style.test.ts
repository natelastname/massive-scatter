import assert from 'node:assert/strict';
import test from 'node:test';

import {
  colorMap,
  markerForCategory,
  markerIcon,
  normalizeMarker,
  parseColor,
  withAlpha,
} from '../src/plot-style.ts';

test('constant colors accept names and hex values', () => {
  assert.deepEqual(parseColor('red'), [214, 39, 40, 255]);
  assert.deepEqual(parseColor('#010203'), [1, 2, 3, 255]);
  assert.deepEqual(parseColor('#01020380'), [1, 2, 3, 128]);
});

test('continuous colors and alpha clamp predictably', () => {
  const low = colorMap(0, 0, 10, 'viridis');
  const high = colorMap(10, 0, 10, 'viridis');
  assert.notDeepEqual(low, high);
  assert.deepEqual(withAlpha([10, 20, 30, 200], 0.5), [10, 20, 30, 100]);
  assert.deepEqual(withAlpha([10, 20, 30, 200], 5), [10, 20, 30, 200]);
});

test('marker aliases and category cycle are stable', () => {
  assert.equal(normalizeMarker('^'), 'triangle');
  assert.equal(normalizeMarker('D'), 'diamond');
  assert.equal(markerForCategory(0), 'circle');
  assert.equal(markerForCategory(6), 'circle');
  assert.equal(markerIcon('square').mask, true);
  assert.match(markerIcon('square').url, /^data:image\/svg\+xml/);
});
