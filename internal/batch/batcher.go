// Package batch implements dynamic request batching in front of an inference
// backend.
//
// The scheduler trades a bounded amount of latency for throughput: incoming
// requests are held for up to MaxWait so that several can be executed in one
// backend call. Because the backend's cost is dominated by a per-call fixed
// term, one call of 32 is dramatically cheaper than 32 calls of 1.
//
// Four mechanisms matter here, and they are the reason this is a systems
// project rather than a queue with a timer:
//
//  1. Admission control. The queue is bounded. When it is full, new work is
//     refused immediately rather than accepted into an unbounded backlog that
//     would make every latency number meaningless.
//  2. Deadline dropping. A request whose caller has already given up is
//     discarded before it reaches the backend. Under overload this is what
//     stops the server spending all its capacity computing answers nobody is
//     waiting for.
//  3. Bounded flush. A batch leaves when it is full OR when the oldest request
//     in it has waited MaxWait. Without the time bound, a partially filled
//     batch at low traffic would wait forever.
//  4. Graceful drain. On shutdown the queue is closed, the in-flight batch is
//     flushed, and workers finish before the process exits.
package batch

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"

	"batchserve/internal/metrics"
	"batchserve/internal/model"
)

var (
	// ErrQueueFull is returned when admission control sheds load. Callers
	// should surface this as HTTP 503 with Retry-After, not 500: the request
	// was never attempted and is safe to retry.
	ErrQueueFull = errors.New("batch: queue full, load shed at admission")

	// ErrShuttingDown is returned once Shutdown has begun.
	ErrShuttingDown = errors.New("batch: server shutting down")

	// ErrDeadlineExceeded is returned to a caller whose context expired while
	// its job was queued.
	ErrDeadlineExceeded = errors.New("batch: deadline exceeded while queued")
)

type Config struct {
	// MaxBatchSize is the flush-by-size trigger. Clamped to the backend's
	// declared maximum.
	MaxBatchSize int
	// MaxWait is the flush-by-time trigger, measured from the arrival of the
	// first job in the current batch. This is the latency budget you are
	// spending to buy throughput.
	MaxWait time.Duration
	// QueueCapacity bounds admitted-but-not-yet-batched work.
	QueueCapacity int
	// Workers is the number of concurrent backend calls.
	Workers int
	// BackendTimeout caps a single RunBatch call.
	BackendTimeout time.Duration
}

func DefaultConfig() Config {
	return Config{
		MaxBatchSize:   32,
		MaxWait:        5 * time.Millisecond,
		QueueCapacity:  1024,
		Workers:        2,
		BackendTimeout: 30 * time.Second,
	}
}

func (c *Config) validate() error {
	if c.MaxBatchSize < 1 {
		return fmt.Errorf("MaxBatchSize must be >= 1, got %d", c.MaxBatchSize)
	}
	if c.MaxWait < 0 {
		return fmt.Errorf("MaxWait must be >= 0, got %v", c.MaxWait)
	}
	if c.QueueCapacity < 1 {
		return fmt.Errorf("QueueCapacity must be >= 1, got %d", c.QueueCapacity)
	}
	if c.Workers < 1 {
		return fmt.Errorf("Workers must be >= 1, got %d", c.Workers)
	}
	if c.BackendTimeout <= 0 {
		return fmt.Errorf("BackendTimeout must be > 0, got %v", c.BackendTimeout)
	}
	return nil
}

type jobResult struct {
	res model.Result
	err error
}

type job struct {
	input      []float32
	ctx        context.Context
	resCh      chan jobResult // buffered(1): workers must never block on reply
	enqueuedAt time.Time
}

// reply delivers a result without ever blocking, and at most once per job.
//
// The default arm is load-bearing, not decoration. runBatch's panic recovery
// replies to every job in the batch, including ones already answered with
// ErrDeadlineExceeded during the drop pass — without the default, that second
// send into a full buffered channel would wedge the worker.
func (j *job) reply(r jobResult) {
	select {
	case j.resCh <- r:
	default:
	}
}

type Batcher struct {
	cfg     Config
	runner  model.Runner
	metrics *metrics.Metrics

	queue   chan *job
	batches chan []*job

	// mu guards closed and serialises it against sends on queue, so that
	// Shutdown can close(queue) without racing a concurrent Submit into a
	// send-on-closed-channel panic. Submit takes RLock (cheap, concurrent);
	// Shutdown takes Lock exactly once.
	mu     sync.RWMutex
	closed bool

	workerWG    sync.WaitGroup
	collectorWG sync.WaitGroup
}

func New(runner model.Runner, cfg Config, m *metrics.Metrics) (*Batcher, error) {
	if runner == nil {
		return nil, errors.New("batch: runner is nil")
	}
	if m == nil {
		return nil, errors.New("batch: metrics is nil")
	}
	if err := cfg.validate(); err != nil {
		return nil, err
	}
	if max := runner.MaxBatchSize(); max > 0 && cfg.MaxBatchSize > max {
		cfg.MaxBatchSize = max
	}

	b := &Batcher{
		cfg:     cfg,
		runner:  runner,
		metrics: m,
		queue:   make(chan *job, cfg.QueueCapacity),
		batches: make(chan []*job, cfg.Workers),
	}

	b.collectorWG.Add(1)
	go b.collect()

	b.workerWG.Add(cfg.Workers)
	for i := 0; i < cfg.Workers; i++ {
		go b.worker()
	}
	return b, nil
}

func (b *Batcher) Config() Config { return b.cfg }

// Submit admits one input and blocks until a result is available, the caller's
// context expires, or the request is shed.
func (b *Batcher) Submit(ctx context.Context, input []float32) (model.Result, error) {
	j := &job{
		input:      input,
		ctx:        ctx,
		resCh:      make(chan jobResult, 1),
		enqueuedAt: time.Now(),
	}

	b.mu.RLock()
	if b.closed {
		b.mu.RUnlock()
		b.metrics.RejectedShutdown.Add(1)
		return model.Result{}, ErrShuttingDown
	}
	// Non-blocking send is the admission decision. Blocking here instead
	// would convert a full queue into unbounded latency, which is the exact
	// failure this design exists to prevent.
	select {
	case b.queue <- j:
		b.mu.RUnlock()
	default:
		b.mu.RUnlock()
		b.metrics.Rejected.Add(1)
		return model.Result{}, ErrQueueFull
	}
	b.metrics.Accepted.Add(1)

	select {
	case r := <-j.resCh:
		return r.res, r.err
	case <-ctx.Done():
		// The job may still be queued or in flight. It is not cancelled here:
		// the worker checks ctx before executing and drops it there. resCh is
		// buffered, so an in-flight worker replying after this return neither
		// blocks nor leaks.
		return model.Result{}, ctx.Err()
	}
}

// collect accumulates jobs into batches and flushes on size or time.
func (b *Batcher) collect() {
	defer b.collectorWG.Done()
	defer close(b.batches)

	var batch []*job

	timer := time.NewTimer(time.Hour)
	if !timer.Stop() {
		<-timer.C
	}
	timerArmed := false

	stopTimer := func() {
		if !timerArmed {
			return
		}
		if !timer.Stop() {
			// Timer already fired; drain the value so the next Reset is clean.
			select {
			case <-timer.C:
			default:
			}
		}
		timerArmed = false
	}

	flush := func() {
		if len(batch) == 0 {
			return
		}
		stopTimer()
		b.batches <- batch
		batch = nil
	}

	for {
		select {
		case j, ok := <-b.queue:
			if !ok {
				flush() // final partial batch on shutdown
				return
			}
			batch = append(batch, j)
			if len(batch) == 1 && b.cfg.MaxWait > 0 {
				timer.Reset(b.cfg.MaxWait)
				timerArmed = true
			}
			if len(batch) >= b.cfg.MaxBatchSize || b.cfg.MaxWait == 0 {
				flush()
			}
		case <-timer.C:
			timerArmed = false
			flush()
		}
	}
}

func (b *Batcher) worker() {
	defer b.workerWG.Done()
	for batch := range b.batches {
		b.runBatch(batch)
	}
}

func (b *Batcher) runBatch(batch []*job) {
	// A panic in the backend must not kill a worker. Losing workers one at a
	// time degrades the pool silently until throughput collapses with no
	// error in the logs, which is a genuinely nasty way to fail.
	defer func() {
		if r := recover(); r != nil {
			err := fmt.Errorf("batch: backend panic: %v", r)
			b.metrics.BackendErrors.Add(int64(len(batch)))
			for _, j := range batch {
				j.reply(jobResult{err: err})
			}
		}
	}()

	now := time.Now()

	// Drop work whose caller has already gone away. Under overload this is
	// the difference between shedding cheaply and spending the whole backend
	// budget on results nobody will read.
	live := make([]*job, 0, len(batch))
	for _, j := range batch {
		if j.ctx.Err() != nil {
			b.metrics.DroppedDeadline.Add(1)
			j.reply(jobResult{err: ErrDeadlineExceeded})
			continue
		}
		b.metrics.QueueWait.Observe(now.Sub(j.enqueuedAt))
		live = append(live, j)
	}
	if len(live) == 0 {
		return
	}

	inputs := make([][]float32, len(live))
	for i, j := range live {
		inputs[i] = j.input
	}

	ctx, cancel := context.WithTimeout(context.Background(), b.cfg.BackendTimeout)
	defer cancel()

	started := time.Now()
	results, err := b.runner.RunBatch(ctx, inputs)
	elapsed := time.Since(started)

	b.metrics.Inference.Observe(elapsed)
	b.metrics.BatchesRun.Add(1)
	b.metrics.ItemsBatched.Add(int64(len(live)))
	b.metrics.BatchSize.Observe(len(live))

	if err != nil {
		b.metrics.BackendErrors.Add(int64(len(live)))
		for _, j := range live {
			j.reply(jobResult{err: err})
		}
		return
	}

	// Contract check. If the backend returns a different count, positional
	// fan-out would hand one caller another caller's result. Failing the whole
	// batch loudly is correct; silently truncating is not.
	if len(results) != len(live) {
		err := fmt.Errorf("batch: backend returned %d results for %d inputs", len(results), len(live))
		b.metrics.BackendErrors.Add(int64(len(live)))
		for _, j := range live {
			j.reply(jobResult{err: err})
		}
		return
	}

	for i, j := range live {
		b.metrics.Completed.Add(1)
		j.reply(jobResult{res: results[i]})
	}
}

// Shutdown stops admission, drains queued work, and waits for in-flight
// batches. It returns ctx.Err() if the drain outlives the supplied context.
func (b *Batcher) Shutdown(ctx context.Context) error {
	b.mu.Lock()
	if b.closed {
		b.mu.Unlock()
		return nil
	}
	b.closed = true
	close(b.queue)
	b.mu.Unlock()

	done := make(chan struct{})
	go func() {
		b.collectorWG.Wait()
		b.workerWG.Wait()
		close(done)
	}()

	select {
	case <-done:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}
