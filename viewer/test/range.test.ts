import assert from 'node:assert/strict';
import test from 'node:test';

import {finiteRangeBy} from '../src/range.ts';

test('finiteRangeBy scans each datum once', () => {
  const values = Array.from({length: 10_000}, (_, index) => ({value: index - 5000}));
  let visits = 0;
  const range = finiteRangeBy(values, item => {
    visits += 1;
    return item.value;
  });
  assert.deepEqual(range, [-5000, 4999]);
  assert.equal(visits, values.length);
});

test('finiteRangeBy ignores non-finite values and uses fallback for empty data', () => {
  assert.deepEqual(finiteRangeBy([1, Number.NaN, 7, Number.POSITIVE_INFINITY], value => value), [1, 7]);
  assert.deepEqual(finiteRangeBy([], value => Number(value), [4, 9]), [4, 9]);
});
