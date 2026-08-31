import assert from 'node:assert/strict';
import test from 'node:test';

import {aggregateCellCorner} from '../src/lod-cell.ts';

test('aggregate cell centers map to their exact lower-left corners', () => {
  assert.deepEqual(aggregateCellCorner([32, 32], 64), [0, 0]);
  assert.deepEqual(aggregateCellCorner([96, 32], 64), [64, 0]);
});

test('adjacent aggregate cells tile without gaps', () => {
  const cellSize = 64;
  const left = aggregateCellCorner([32, 32], cellSize);
  const right = aggregateCellCorner([96, 32], cellSize);
  const above = aggregateCellCorner([32, 96], cellSize);

  assert.equal(left[0] + cellSize, right[0]);
  assert.equal(left[1] + cellSize, above[1]);
});

test('aggregate cell size must be positive and finite', () => {
  assert.throws(() => aggregateCellCorner([0, 0], 0), RangeError);
  assert.throws(() => aggregateCellCorner([0, 0], Number.POSITIVE_INFINITY), RangeError);
});
