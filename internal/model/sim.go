package model

import (
	"context"
	"fmt"
	"math"
	"sync/atomic"
	"time"
)

// SimConfig describes the cost curve of the simulated backend.
//
// Real accelerator-backed inference has a large fixed cost per call (kernel
// launch, weight streaming, host<->device copies) and a comparatively small
// marginal cost per item. That asymmetry is the entire reason batching wins:
// a batch of 32 costs far less than 32x a batch of 1.
//
// Defaults here are loosely shaped like a small vision model on a busy GPU.
type SimConfig struct {
	// FixedCost is paid once per RunBatch call regardless of batch size.
	FixedCost time.Duration
	// MarginalCost is paid per item in the batch.
	MarginalCost time.Duration
	// MaxBatch is the largest batch the backend will accept.
	MaxBatch int
	// FailureRate in [0,1] injects synthetic backend errors. Used to exercise
	// error paths under load; leave at 0 for benchmarking.
	FailureRate float64
}

func DefaultSimConfig() SimConfig {
	return SimConfig{
		FixedCost:    18 * time.Millisecond,
		MarginalCost: 350 * time.Microsecond,
		MaxBatch:     64,
	}
}

// Sim is a Runner that sleeps according to a configured cost curve and
// returns deterministic results derived from each input.
//
// Results are deterministic on purpose: tests assert that the caller who
// submitted input X receives the result derived from X, which is how
// fan-out cross-talk bugs get caught.
type Sim struct {
	cfg SimConfig

	calls     atomic.Int64
	itemsSeen atomic.Int64
	rngState  atomic.Uint64
}

func NewSim(cfg SimConfig) *Sim {
	if cfg.MaxBatch <= 0 {
		cfg.MaxBatch = 64
	}
	s := &Sim{cfg: cfg}
	s.rngState.Store(0x9E3779B97F4A7C15)
	return s
}

func (s *Sim) Name() string { return "sim" }

func (s *Sim) MaxBatchSize() int { return s.cfg.MaxBatch }

// Calls and Items expose backend-side counters so benchmarks can compute the
// achieved average batch size independently of the batcher's own accounting.
func (s *Sim) Calls() int64 { return s.calls.Load() }
func (s *Sim) Items() int64 { return s.itemsSeen.Load() }

func (s *Sim) RunBatch(ctx context.Context, inputs [][]float32) ([]Result, error) {
	n := len(inputs)
	if n == 0 {
		return nil, nil
	}
	if n > s.cfg.MaxBatch {
		return nil, fmt.Errorf("batch of %d exceeds backend max %d", n, s.cfg.MaxBatch)
	}

	s.calls.Add(1)
	s.itemsSeen.Add(int64(n))

	cost := s.cfg.FixedCost + time.Duration(n)*s.cfg.MarginalCost

	// Sleep, but stay responsive to cancellation. A real backend would be
	// mid-kernel here and typically could not be cancelled; we honour ctx so
	// shutdown does not hang on an in-flight batch.
	select {
	case <-time.After(cost):
	case <-ctx.Done():
		return nil, ctx.Err()
	}

	if s.cfg.FailureRate > 0 && s.nextFloat() < s.cfg.FailureRate {
		return nil, fmt.Errorf("sim: injected backend failure")
	}

	out := make([]Result, n)
	for i, in := range inputs {
		out[i] = predict(in)
	}
	return out, nil
}

// predict is a pure function of the input so results are traceable back to
// their caller.
func predict(in []float32) Result {
	var sum float64
	for _, v := range in {
		sum += float64(v)
	}
	score := float32(1 / (1 + math.Exp(-sum/8)))
	label := "negative"
	if score >= 0.5 {
		label = "positive"
	}
	return Result{Label: label, Score: score}
}

// nextFloat is a lock-free xorshift used only for failure injection.
func (s *Sim) nextFloat() float64 {
	for {
		old := s.rngState.Load()
		x := old
		x ^= x << 13
		x ^= x >> 7
		x ^= x << 17
		if s.rngState.CompareAndSwap(old, x) {
			return float64(x>>11) / float64(1<<53)
		}
	}
}
