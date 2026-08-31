# batchserve

A dynamic batching inference server in Go. Requests arrive one at a time and are
grouped into batches before hitting the model backend, trading a bounded amount
of latency for a large gain in throughput.

Standard library only — no external dependencies.

## Why batching works

The simulated backend has the cost profile of real accelerator inference:

```
latency(n) = FixedCost + n * MarginalCost
           = 18ms      + n * 0.35ms
```

The fixed term dominates. One call with 32 items costs ~29ms; thirty-two calls
with 1 item each cost ~587ms. The scheduler's entire job is to find batches
worth forming without making anyone wait too long for one.

## Architecture

```
HTTP handler                      per-request deadline attached here
     |
     v
Submit() ------> [ bounded queue ] ------> collector ------> [ batches ] ---> worker pool
     |                   |                     |                                  |
  admission          full => 503          flush when:                        drop expired,
  control            immediately          - batch full                       run backend,
                                          - MaxWait elapsed                  fan out results
                                          - shutdown
```

Four mechanisms carry the design:

**Admission control.** The queue is bounded and `Submit` uses a non-blocking
send. When the queue is full the request is refused immediately with 503 and
`Retry-After`. Blocking instead would convert a full queue into unbounded
latency — which is the exact failure this exists to prevent.

**Deadline dropping.** Before a batch runs, jobs whose caller has already given
up are discarded. Under overload this is what stops the backend spending its
whole capacity computing answers nobody is waiting for.

**Bounded flush.** A batch leaves when it is full *or* when its oldest member
has waited `MaxWait`. Without the time bound a partially filled batch would sit
forever at low traffic.

**Graceful drain.** On SIGTERM the HTTP listener stops first, then the queue
closes, the partial batch flushes, and workers finish. Reversing that order
would admit requests into a closed scheduler.

## Results

Measured on the machine described under *Limitations*. Same offered load, same
backend, only the batching config differs.

### Batching on vs off — 300 req/s offered

| | batch size 1 | batch size 32, 5ms window |
|---|---|---|
| goodput | 23.3 req/s | **298.9 req/s** |
| succeeded | 326 | 3599 |
| timed out (504) | 3271 | **0** |
| p50 latency | — | 51.7 ms |
| p99 latency | — | 64.6 ms |
| avg batch size | 1.00 | 2.93 |
| queue wait p99 | 2295.9 ms | 41.4 ms |

A 12.8x goodput improvement, and the failure count goes to zero. Latency
percentiles are absent for the unbatched run because essentially no request
succeeded after the warmup window — the queue had already backed up past the
2s request deadline and stayed there.

Note the average batch size of 2.93. At 300 req/s with a 5ms window only about
1.5 requests arrive per window, so batches stay small — and the win is still
enormous, because even a batch of 3 amortises the 18ms fixed cost three ways.
Batching does not need to be efficient to be transformative.

### Queue sizing decides how overload feels

Both runs are overloaded. They fail completely differently.

| | oversized queue (1024) | tight queue (16) |
|---|---|---|
| shed fast (503) | 0 | **4709** |
| timed out (504) | **3271** | 0 |
| goodput | 23.3 req/s | 125.0 req/s |
| p99 latency | — | 318 ms, stable |

With a large queue, nothing is ever refused. Every request is accepted, waits
behind 1000 others, and eventually times out. The server fails *everyone*,
slowly, while reporting no rejections at all.

With a small queue, excess load is refused at the door in microseconds. The
requests that get in are served at a stable, predictable latency. Total useful
work goes up 5.4x.

This is the most useful thing in the project. A queue is not a buffer that makes
overload go away; it decides whether overload shows up as honest fast rejection
or as universal slow failure. The oversized-queue result was not a designed
demo — it came out of the first benchmark run and the small-queue config was
written in response to it.

## Running

```bash
go run ./cmd/server -batch-size 32 -max-wait 5ms -workers 2 -queue 256

curl -s localhost:8080/predict -d '{"input":[0.1,0.9,0.4]}' | jq
curl -s localhost:8080/metrics | jq

# reproduce the comparison
go run ./cmd/loadgen -rate 300 -duration 20s
```

Set `-batch-size 1 -max-wait 0` for the unbatched baseline.

### Endpoints

| method | path | notes |
|---|---|---|
| POST | `/predict` | single input, blocks for result |
| POST | `/predict/batch` | many inputs, each scheduled independently |
| GET | `/healthz` | liveness |
| GET | `/metrics` | JSON counters and latency percentiles |

`/predict/batch` deliberately does *not* hand its items to the backend as one
batch. The scheduler owns batching decisions; letting a client assemble its own
batch would let one caller monopolise a backend call and starve everyone else.

Its items succeed and fail independently, so the response status summarises
them rather than describing only the HTTP request:

| outcome | status |
|---|---|
| every item succeeded | 200 |
| every item failed the same way | that failure's code (400, 500, 503, 504) |
| mixed | 207, read `status` per item |

Every item carries the code it would have received from `/predict` on its own,
plus a `succeeded`/`failed` summary. A batch where nothing succeeded must never
answer 200 — a caller checking only the status would read total failure as
success.

### Status codes

| code | meaning | retry? |
|---|---|---|
| 503 | shed at admission, never attempted | yes, `Retry-After` set |
| 504 | deadline expired while queued | maybe, with backoff |
| 500 | backend genuinely failed | no |
| 207 | batch only: outcomes differ, read per item | per item |

Collapsing these into one code makes the server unoperable — a caller cannot
tell "I was never tried" from "I broke". Both endpoints derive their codes from
the same mapping, so an item inside a batch never disagrees with what
`/predict` would have said about the same input.

### Reading the metrics

Two decisions in `/metrics` are worth knowing about, because both were bugs
first.

**`end_to_end_latency` counts only requests the scheduler attempted.** Load shed
at admission is excluded and appears in `rejected_queue_full` instead. A 503
costs microseconds, so including them averages the cost of *refusing* work into
the cost of *doing* it — and the more the server sheds, the better it looks.
Measured on the overload config below, 3795 shed against 406 served:

| | reported p50 |
|---|---|
| shed folded in | 0.10 ms |
| shed excluded | 496.16 ms |

The first number is not a rounding artifact. It is a server failing 90% of its
traffic while its own dashboard reports a tenth of a millisecond.

**`batch_size` percentiles are exact, unlike the latency ones.** Batch sizes are
small integers bounded by the backend maximum, so there is one counter per size
and no estimation. It is reported alongside `average_batch_size` because the
average alone hides shape: a run split evenly between batches of 1 and 31
averages 16, a batch size that never once occurred.

## Tests

```bash
go test -race ./...
```

44 test functions, 54 cases including subtests. The ones that matter:

- **`TestNoCrossTalkUnderConcurrency`** — 600 concurrent requests, each
  verifying it received the answer computed from *its own* input. Positional
  fan-out means an off-by-one anywhere in the batch path silently hands caller
  A the result for caller B, and every request still returns 200 with a
  plausible body. This is the failure mode that would never show up in
  production monitoring.
- **`TestExpiredJobsAreDroppedBeforeReachingBackend`** — asserts abandoned work
  never reaches the model, by inspecting what the backend actually saw.
- **`TestSaturationReturns503NotTimeout`** — shed load must never surface as 500.
- **`TestBackendPanicDoesNotKillWorker`** — without `recover`, a panicking model
  kills workers one at a time until throughput silently collapses with nothing
  in the logs.
- **`TestMismatchedResultCountFailsLoudly`** — a backend returning the wrong
  number of results fails the whole batch rather than mis-aligning replies.
- **`TestConcurrentSubmitDuringShutdown`** — race-detector coverage for the
  send-on-closed-channel window between `Submit` and `Shutdown`.
- **`TestSnapshotRacesWithIncreasingObservations`** — the histogram's
  percentile path read `max` outside the mutex. The pre-existing concurrency
  test could not catch it: it observed a constant value, so `max` was written
  once and never raced. Only a strictly increasing series keeps that field hot.
  A concurrency test whose workload never exercises the mutated field is
  reassurance, not coverage.
- **`TestShedRequestsStayOutOfLatencyHistogram`** — 503s cost microseconds, so
  recording them made `end_to_end_latency` look best exactly when the server
  was serving least. See *Reading the metrics*.
- **`TestBatchAllFailedDoesNotReturn200`** — a batch in which every item failed
  used to answer 200, so a caller checking only the status line read total
  failure as success.
- **`TestBatchPartialFailureReturns207`** — also asserts that an item's code
  inside a batch matches what `/predict` returns for the same input, so the two
  endpoints cannot drift apart.

## Limitations

Stated plainly, because a benchmark without its caveats is marketing.

- **Measured on 1 vCPU with 4 GB RAM.** The load generator and the server share
  that single core, so the generator becomes the bottleneck above roughly
  1000 req/s. Absolute throughput numbers here understate what the design does
  on real hardware; the *relative* comparisons are the meaningful part, since
  both sides of each comparison ran under identical constraints.
- **The backend is simulated.** It sleeps according to a cost curve rather than
  running a model. That isolates the serving layer, which is what is being
  measured, but it means there is no GPU memory pressure, no real tail from
  weight streaming, and no CPU contention from actual compute.
- **Latency percentiles are bucket estimates**, not exact. Buckets grow by
  1.25x, so a reported value is the upper bound of the bucket containing the
  true value — it overstates by at most 20% and never understates. An early
  version reported p99 above the observed max because of this; percentiles are
  now clamped to the true max, with a regression test. (`batch_size`
  percentiles are exact — different type, one counter per size.)
- **No priority or fairness.** The queue is strictly FIFO. A single client
  can consume the whole queue. Per-tenant fairness would need weighted queues.
- **No adaptive batching.** `MaxWait` and `MaxBatchSize` are static. A real
  system would tune the window against observed arrival rate and the latency
  SLO instead of holding it fixed.
- **The load generator is open-loop**, which is correct for measuring a queueing
  system, but it issues requests from goroutines on the same host, so client-side
  scheduling delay is included in the reported latencies.

## Things I would do next

- Adaptive `MaxWait` driven by measured arrival rate and a latency target.
- Per-tenant queues with weighted fair sharing.
- A real ONNX backend behind the same `model.Runner` interface, which is the
  only thing that would have to change.
- Prometheus exposition instead of the hand-rolled histogram.
