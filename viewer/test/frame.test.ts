import assert from 'node:assert/strict';
import test from 'node:test';

import {
  localToWorld,
  toRenderViewState,
  toWorldViewState,
  type OrthographicState,
} from '../src/frame.ts';

test('LOD origin changes do not move the logical camera', () => {
  const world: OrthographicState = {
    target: [1_000_123.5, 900_456.25, 0],
    zoom: 7,
  };
  const exactOrigin = [1_000_100, 900_440] as const;
  const lodOrigin = [999_936, 899_840] as const;

  const exact = toRenderViewState(world, exactOrigin);
  const lod = toRenderViewState(world, lodOrigin);

  assert.deepEqual(toWorldViewState(exact, exactOrigin), world);
  assert.deepEqual(toWorldViewState(lod, lodOrigin), world);
  assert.notDeepEqual(exact.target, lod.target);
});

test('local point positions reconstruct the same world position after rebasing', () => {
  const worldPoint: [number, number] = [1_000_137, 900_221];
  const firstOrigin = [1_000_000, 900_000] as const;
  const secondOrigin = [999_936, 899_840] as const;

  const firstLocal: [number, number] = [
    worldPoint[0] - firstOrigin[0],
    worldPoint[1] - firstOrigin[1],
  ];
  const secondLocal: [number, number] = [
    worldPoint[0] - secondOrigin[0],
    worldPoint[1] - secondOrigin[1],
  ];

  assert.deepEqual(localToWorld(firstOrigin, firstLocal), worldPoint);
  assert.deepEqual(localToWorld(secondOrigin, secondLocal), worldPoint);
});
