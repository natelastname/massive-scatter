import assert from 'node:assert/strict';
import test from 'node:test';

import {packColors, packFloat32, packPositions} from '../src/binary-data.ts';

test('packPositions creates interleaved viewport-local GPU coordinates', () => {
  assert.deepEqual(
    Array.from(packPositions([10, 20], [30, 40], -2, -3)),
    [8, 27, 18, 37],
  );
});

test('packColors and packFloat32 create one attribute value per primitive', () => {
  assert.deepEqual(
    Array.from(packColors(2, index => (index ? [5, 6, 7, 8] : [1, 2, 3, 4]))),
    [1, 2, 3, 4, 5, 6, 7, 8],
  );
  assert.deepEqual(Array.from(packFloat32(3, index => index + 0.5)), [0.5, 1.5, 2.5]);
});
