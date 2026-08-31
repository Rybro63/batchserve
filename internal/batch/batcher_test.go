package batch

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"testing"
	"time"

	"batchserve/internal/metrics"
	"batchserve/internal/model"
)

// mockRunner records every batch it is handed and can be gated so a test can
// hold a worker busy at a chosen moment.
type mockRunner struct {
	mu         sync.Mutex
	batchSizes []int
	seen       [][]float32

	gate     chan struct{} // if non-nil, RunBatch blocks until a value is sent
	err      error
	panicOn  bool
	shortBy  int // return this many fewer results than inputs
	maxBatch int
	delay    time.Duration
}

func newMockRunner() *mockRunner {
	return &mockRunner{maxBatch: 1024}
}

func (m *mockRunner) Name() string      { return "mock" }
func (m *mockRunner) MaxBatchSize() int { return m.maxBatch }

func (m *mockRunner) RunBatch(ctx context.Context, inputs [][]float32) ([]model.Result, error) {
	if m.gate != nil {
		select {
		case <-m.gate:
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
	if m.delay > 0 {
		select {
		case <-time.After(m.delay):
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}

	m.mu.Lock()
	m.batchSizes = append(m.batchSizes, len(inputs))
	for _, in := range inputs {
		cp := make([]float32, len(in))
		copy(cp, in)
		m.seen = append(m.seen, cp)
	}
	m.mu.Unlock()

	if m.panicOn {
		panic("mock backend exploded")
	}
	if m.err != nil {
		return nil, m.err
	}

	n := len(inputs) - m.shortBy
	if n < 0 {
		n = 0
	}
	out := make([]model.Result, n)
	for i := 0; i < n; i++ {
		out[i] = model.Result{Label: labelFor(inputs[i]), Score: float32(i)}
	}
	return out, nil
}

func (m *mockRunner) sizes() []int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return append([]int(nil), m.batchSizes...)
}

func (m *mockRunner) totalSeen() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return len(m.seen)
}

// labelFor is an injective-enough function of the input so a test can prove a
// caller received the answer to its own question.
func labelFor(in []float32) string {
	var sum float32
	for _, v := range in {
		sum += v
	}
	return fmt.Sprintf("sum=%.4f", sum)
}

func newTestBatcher(t *testing.T, r model.Runner, cfg Config) (*Batcher, *metrics.Metrics) {
	t.Helper()
	m := metrics.New()
	b, err := New(r, cfg, m)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	t.Cleanup(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_ = b.Shutdown(ctx)
	})
	return b, m
}

func input(v float32) []float32 { return []float32{v} }

// --- flush triggers -------------------------------------------------------

func TestFlushesWhenBatchFills(t *testing.T) {
	r := newMockRunner()
	// MaxWait is long enough that the timer cannot be what flushes this.
	b, _ := newTestBatcher(t, r, Config{
		MaxBatchSize: 4, MaxWait: 10 * time.Second,
		QueueCapacity: 64, Workers: 1, BackendTimeout: time.Second,
	})

	var wg sync.WaitGroup
	for i := 0; i < 4; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
			defer cancel()
			if _, err := b.Submit(ctx, input(float32(i))); err != nil {
				t.Errorf("submit %d: %v", i, err)
			}
		}(i)
	}

	done := make(chan struct{})
	go func() { wg.Wait(); close(done) }()

	select {
	case <-done:
	case <-time.After(3 * time.Second):
		t.Fatal("batch did not flush on size trigger; timer must not be the only path")
	}

	if got := r.sizes(); len(got) != 1 || got[0] != 4 {
		t.Fatalf("expected exactly one batch of 4, got %v", got)
	}
}

func TestFlushesOnTimeoutWhenPartiallyFull(t *testing.T) {
	r := newMockRunner()
	b, _ := newTestBatcher(t, r, Config{
		MaxBatchSize: 64, MaxWait: 20 * time.Millisecond,
		QueueCapacity: 64, Workers: 1, BackendTimeout: time.Second,
	})

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	start := time.Now()
	if _, err := b.Submit(ctx, input(1)); err != nil {
		t.Fatalf("submit: %v", err)
	}
	elapsed := time.Since(start)

	if elapsed < 15*time.Millisecond {
		t.Fatalf("returned in %v, before the %v window elapsed", elapsed, 20*time.Millisecond)
	}
	if elapsed > 500*time.Millisecond {
		t.Fatalf("took %v; a single request should leave after roughly MaxWait", elapsed)
	}
	if got := r.sizes(); len(got) != 1 || got[0] != 1 {
		t.Fatalf("expected one batch of 1, got %v", got)
	}
}

func TestZeroMaxWaitFlushesImmediately(t *testing.T) {
	r := newMockRunner()
	b, _ := newTestBatcher(t, r, Config{
		MaxBatchSize: 32, MaxWait: 0,
		QueueCapacity: 64, Workers: 1, BackendTimeout: time.Second,
	})

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	for i := 0; i < 3; i++ {
		if _, err := b.Submit(ctx, input(float32(i))); err != nil {
			t.Fatalf("submit: %v", err)
		}
	}
	for _, n := range r.sizes() {
		if n != 1 {
			t.Fatalf("MaxWait=0 should never group requests, saw batch of %d", n)
		}
	}
}

// --- admission control ----------------------------------------------------

func TestQueueFullShedsInsteadOfBlocking(t *testing.T) {
	r := newMockRunner()
	r.gate = make(chan struct{}) // pin the worker so nothing drains

	b, m := newTestBatcher(t, r, Config{
		MaxBatchSize: 1, MaxWait: time.Millisecond,
		QueueCapacity: 2, Workers: 1, BackendTimeout: 5 * time.Second,
	})

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	// Fill the pipeline: one batch in a blocked worker, then saturate the
	// queue. Exact admitted count varies with scheduling, so assert on the
	// property that matters rather than an exact number.
	var shedCount int
	for i := 0; i < 200; i++ {
		go func(i int) { _, _ = b.Submit(ctx, input(float32(i))) }(i)
	}
	time.Sleep(200 * time.Millisecond)

	for i := 0; i < 50; i++ {
		if _, err := b.Submit(ctx, input(99)); errors.Is(err, ErrQueueFull) {
			shedCount++
		}
	}
	if shedCount == 0 {
		t.Fatal("expected admission control to shed load once the queue saturated")
	}
	if m.Rejected.Load() == 0 {
		t.Fatal("rejected counter never incremented")
	}
	close(r.gate)
}

// --- deadline handling ----------------------------------------------------

func TestExpiredJobsAreDroppedBeforeReachingBackend(t *testing.T) {
	r := newMockRunner()
	r.gate = make(chan struct{})

	b, m := newTestBatcher(t, r, Config{
		MaxBatchSize: 1, MaxWait: time.Millisecond,
		QueueCapacity: 32, Workers: 1, BackendTimeout: 5 * time.Second,
	})

	// Occupy the single worker with a batch that will not complete yet.
	blockCtx, blockCancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer blockCancel()
	go func() { _, _ = b.Submit(blockCtx, input(0)) }()
	time.Sleep(50 * time.Millisecond)

	// This one expires while stuck behind the blocked worker.
	shortCtx, shortCancel := context.WithTimeout(context.Background(), 30*time.Millisecond)
	defer shortCancel()
	_, err := b.Submit(shortCtx, input(1))
	if err == nil {
		t.Fatal("expected an error for a request that outlived its deadline")
	}

	time.Sleep(100 * time.Millisecond)
	close(r.gate)
	time.Sleep(200 * time.Millisecond)

	if m.DroppedDeadline.Load() == 0 {
		t.Fatal("expired job was not counted as dropped")
	}
	// The whole point: abandoned work must not consume backend capacity.
	for _, in := range r.seen {
		if len(in) > 0 && in[0] == 1 {
			t.Fatal("backend executed a request whose caller had already given up")
		}
	}
}

// --- correctness under concurrency ---------------------------------------

// TestNoCrossTalkUnderConcurrency is the test that matters most. Positional
// fan-out means an off-by-one anywhere in the batch path hands caller A the
// answer computed for caller B. That corruption is silent: every request still
// returns 200 with a plausible-looking body.
func TestNoCrossTalkUnderConcurrency(t *testing.T) {
	r := newMockRunner()
	r.delay = time.Millisecond

	b, _ := newTestBatcher(t, r, Config{
		MaxBatchSize: 16, MaxWait: 3 * time.Millisecond,
		QueueCapacity: 2048, Workers: 4, BackendTimeout: 5 * time.Second,
	})

	const n = 600
	var wg sync.WaitGroup
	errs := make(chan error, n)

	for i := 0; i < n; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
			defer cancel()

			in := input(float32(i))
			want := labelFor(in)

			res, err := b.Submit(ctx, in)
			if err != nil {
				errs <- fmt.Errorf("request %d: %w", i, err)
				return
			}
			if res.Label != want {
				errs <- fmt.Errorf("request %d got %q, want %q: results crossed between callers", i, res.Label, want)
			}
		}(i)
	}
	wg.Wait()
	close(errs)

	for err := range errs {
		t.Error(err)
	}
}

// --- backend failure modes ------------------------------------------------

func TestBackendErrorReachesEveryCallerInBatch(t *testing.T) {
	r := newMockRunner()
	r.err = errors.New("backend unavailable")

	b, m := newTestBatcher(t, r, Config{
		MaxBatchSize: 4, MaxWait: 5 * time.Millisecond,
		QueueCapacity: 32, Workers: 1, BackendTimeout: time.Second,
	})

	var wg sync.WaitGroup
	var failures int64
	var mu sync.Mutex
	for i := 0; i < 4; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
			defer cancel()
			if _, err := b.Submit(ctx, input(float32(i))); err != nil {
				mu.Lock()
				failures++
				mu.Unlock()
			}
		}(i)
	}
	wg.Wait()

	if failures != 4 {
		t.Fatalf("expected all 4 callers to see the failure, got %d", failures)
	}
	if m.BackendErrors.Load() != 4 {
		t.Fatalf("backend error counter = %d, want 4", m.BackendErrors.Load())
	}
}

func TestBackendPanicDoesNotKillWorker(t *testing.T) {
	r := newMockRunner()
	r.panicOn = true

	b, _ := newTestBatcher(t, r, Config{
		MaxBatchSize: 1, MaxWait: time.Millisecond,
		QueueCapacity: 32, Workers: 1, BackendTimeout: time.Second,
	})

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()

	if _, err := b.Submit(ctx, input(1)); err == nil {
		t.Fatal("expected an error from a panicking backend")
	}

	// If the recover had not re-armed the pool, the sole worker would be gone
	// and this second call would hang until its deadline.
	r.panicOn = false
	if _, err := b.Submit(ctx, input(2)); err != nil {
		t.Fatalf("worker did not survive the panic: %v", err)
	}
}

func TestMismatchedResultCountFailsLoudly(t *testing.T) {
	r := newMockRunner()
	r.shortBy = 1

	b, _ := newTestBatcher(t, r, Config{
		MaxBatchSize: 2, MaxWait: 5 * time.Millisecond,
		QueueCapacity: 32, Workers: 1, BackendTimeout: time.Second,
	})

	var wg sync.WaitGroup
	errCount := 0
	var mu sync.Mutex
	for i := 0; i < 2; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
			defer cancel()
			if _, err := b.Submit(ctx, input(float32(i))); err != nil {
				mu.Lock()
				errCount++
				mu.Unlock()
			}
		}(i)
	}
	wg.Wait()

	if errCount != 2 {
		t.Fatalf("a short result set must fail the whole batch rather than "+
			"silently mis-align replies; got %d errors, want 2", errCount)
	}
}

// --- shutdown -------------------------------------------------------------

func TestShutdownDrainsQueuedWork(t *testing.T) {
	r := newMockRunner()
	r.delay = 5 * time.Millisecond

	m := metrics.New()
	b, err := New(r, Config{
		MaxBatchSize: 8, MaxWait: 50 * time.Millisecond,
		QueueCapacity: 256, Workers: 2, BackendTimeout: 5 * time.Second,
	}, m)
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	const n = 40
	var wg sync.WaitGroup
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
			defer cancel()
			_, _ = b.Submit(ctx, input(float32(i)))
		}(i)
	}
	time.Sleep(20 * time.Millisecond)

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err := b.Shutdown(ctx); err != nil {
		t.Fatalf("Shutdown: %v", err)
	}
	wg.Wait()

	completed := m.Completed.Load()
	if completed == 0 {
		t.Fatal("shutdown discarded all queued work instead of draining it")
	}
	if r.totalSeen() == 0 {
		t.Fatal("no work reached the backend during drain")
	}
}

func TestSubmitAfterShutdownIsRejected(t *testing.T) {
	r := newMockRunner()
	m := metrics.New()
	b, err := New(r, DefaultConfig(), m)
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := b.Shutdown(ctx); err != nil {
		t.Fatalf("Shutdown: %v", err)
	}

	if _, err := b.Submit(ctx, input(1)); !errors.Is(err, ErrShuttingDown) {
		t.Fatalf("got %v, want ErrShuttingDown", err)
	}

	// A refusal that increments no counter is a refusal nobody can see. This
	// used to be the one shed path invisible in /metrics: neither Accepted nor
	// Rejected moved, so a deploy that dropped traffic during drain left no
	// trace at all.
	if got := m.RejectedShutdown.Load(); got != 1 {
		t.Errorf("RejectedShutdown = %d, want 1", got)
	}
	if got := m.Accepted.Load(); got != 0 {
		t.Errorf("Accepted = %d, want 0: the job never entered the queue", got)
	}
	if got := m.Rejected.Load(); got != 0 {
		t.Errorf("Rejected = %d, want 0: the queue was not full, it was closed", got)
	}
}

// TestConcurrentSubmitDuringShutdown exists for the race detector. Submitting
// while Shutdown closes the queue is the classic send-on-closed-channel panic;
// the RWMutex in Submit/Shutdown is what prevents it.
func TestConcurrentSubmitDuringShutdown(t *testing.T) {
	r := newMockRunner()
	m := metrics.New()
	b, err := New(r, Config{
		MaxBatchSize: 8, MaxWait: 2 * time.Millisecond,
		QueueCapacity: 512, Workers: 4, BackendTimeout: 5 * time.Second,
	}, m)
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	var wg sync.WaitGroup
	for i := 0; i < 200; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			_, _ = b.Submit(ctx, input(float32(i)))
		}(i)
	}

	time.Sleep(5 * time.Millisecond)
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err := b.Shutdown(ctx); err != nil {
		t.Fatalf("Shutdown: %v", err)
	}
	wg.Wait()
}

func TestShutdownIsIdempotent(t *testing.T) {
	r := newMockRunner()
	m := metrics.New()
	b, err := New(r, DefaultConfig(), m)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	for i := 0; i < 3; i++ {
		if err := b.Shutdown(ctx); err != nil {
			t.Fatalf("Shutdown call %d: %v", i, err)
		}
	}
}

// --- config ---------------------------------------------------------------

func TestInvalidConfigIsRejected(t *testing.T) {
	cases := map[string]Config{
		"zero batch size": {MaxBatchSize: 0, QueueCapacity: 1, Workers: 1, BackendTimeout: time.Second},
		"zero queue":      {MaxBatchSize: 1, QueueCapacity: 0, Workers: 1, BackendTimeout: time.Second},
		"zero workers":    {MaxBatchSize: 1, QueueCapacity: 1, Workers: 0, BackendTimeout: time.Second},
		"no timeout":      {MaxBatchSize: 1, QueueCapacity: 1, Workers: 1, BackendTimeout: 0},
		"negative wait":   {MaxBatchSize: 1, MaxWait: -1, QueueCapacity: 1, Workers: 1, BackendTimeout: time.Second},
	}
	for name, cfg := range cases {
		t.Run(name, func(t *testing.T) {
			if _, err := New(newMockRunner(), cfg, metrics.New()); err == nil {
				t.Fatal("expected config validation to fail")
			}
		})
	}
}

func TestBatchSizeClampedToBackendMax(t *testing.T) {
	r := newMockRunner()
	r.maxBatch = 4

	cfg := DefaultConfig()
	cfg.MaxBatchSize = 128
	b, _ := newTestBatcher(t, r, cfg)

	if got := b.Config().MaxBatchSize; got != 4 {
		t.Fatalf("MaxBatchSize = %d, want it clamped to the backend max of 4", got)
	}
}
