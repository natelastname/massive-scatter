import assert from 'node:assert/strict';
import test from 'node:test';

import {LatestRequestRunner} from '../src/latest-request.ts';

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((resolver, rejecter) => {
    resolve = resolver;
    reject = rejecter;
  });
  return {promise, resolve, reject};
}

test('coalesces intermediate requests while one request is active', async () => {
  const runner = new LatestRequestRunner<number>();
  const first = deferred<number>();
  const last = deferred<number>();
  const started: string[] = [];
  const committed: number[] = [];

  runner.enqueue(async () => {
    started.push('first');
    return first.promise;
  }, value => committed.push(value));

  runner.enqueue(async () => {
    started.push('middle');
    return 2;
  }, value => committed.push(value));

  runner.enqueue(async () => {
    started.push('last');
    return last.promise;
  }, value => committed.push(value));

  assert.deepEqual(started, ['first']);
  first.resolve(1);
  await new Promise(resolve => setImmediate(resolve));
  assert.deepEqual(started, ['first', 'last']);
  assert.deepEqual(committed, []);

  last.resolve(3);
  await new Promise(resolve => setImmediate(resolve));
  assert.deepEqual(committed, [3]);
});

test('commits a request when it is not superseded', async () => {
  const runner = new LatestRequestRunner<number>();
  const committed: number[] = [];
  runner.enqueue(async () => 7, value => committed.push(value));
  await new Promise(resolve => setImmediate(resolve));
  assert.deepEqual(committed, [7]);
});

test('ignores stale failures but reports the latest failure', async () => {
  const runner = new LatestRequestRunner<number>();
  const first = deferred<number>();
  const last = deferred<number>();
  const rejected: unknown[] = [];

  runner.enqueue(
    async () => first.promise,
    () => {},
    error => rejected.push(error),
  );
  runner.enqueue(
    async () => last.promise,
    () => {},
    error => rejected.push(error),
  );

  first.reject(new Error('stale'));
  await new Promise(resolve => setImmediate(resolve));
  assert.deepEqual(rejected, []);

  const latest = new Error('latest');
  last.reject(latest);
  await new Promise(resolve => setImmediate(resolve));
  assert.deepEqual(rejected, [latest]);
});
