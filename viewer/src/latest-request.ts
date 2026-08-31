export type AsyncTask<T> = () => Promise<T>;
export type Commit<T> = (value: T) => void;

/**
 * Run at most one asynchronous request at a time while retaining only the
 * newest request that arrives during an in-flight operation.
 *
 * A superseded result is discarded rather than committed. Intermediate queued
 * requests are overwritten, so a slow backend cannot accumulate an unbounded
 * backlog during pan/zoom interaction.
 */
export class LatestRequestRunner<T> {
  private running = false;
  private generation = 0;
  private pending: {generation: number; task: AsyncTask<T>} | null = null;

  enqueue(task: AsyncTask<T>, commit: Commit<T>): void {
    const generation = ++this.generation;
    this.pending = {generation, task};
    if (!this.running) void this.drain(commit);
  }

  private async drain(commit: Commit<T>): Promise<void> {
    this.running = true;
    try {
      while (this.pending) {
        const current = this.pending;
        this.pending = null;
        const value = await current.task();
        if (current.generation === this.generation) commit(value);
      }
    } finally {
      this.running = false;
      // A task can be enqueued between the loop observing no pending work and
      // the finally block. Ensure that race still gets drained.
      if (this.pending) void this.drain(commit);
    }
  }
}
