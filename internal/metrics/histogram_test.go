package metrics

import (
	"math"
	"sync"
	"testing"
	"time"
)

func TestEmptyHistogram(t *testing.T) {
	h := NewLatencyHistogram()
	s := h.Snapshot()
	if s.Count != 0 || s.P99 != 0 {
		t.Fatalf("empty histogram should report zeros, got %+v", s)
	}
}

func TestPercentilesAreWithinBucketError(t *testing.T) {
	h := NewLatencyHistogram()
	// 1..1000 ms uniformly.
	for i := 1; i <= 1000; i++ {
		h.Observe(time.Duration(i) * time.Millisecond)
	}
	s := h.Snapshot()

	if s.Count != 1000 {
		t.Fatalf("count = %d, want 1000", s.Count)
	}

	// Buckets grow by 1.25x, so a reported value is at most 25% above truth
	// and never below it. Assert that contract explicitly rather than
	// pretending the estimate is exact.
	check := func(name string, got time.Duration, want time.Duration) {
		t.Helper()
		gotMs := float64(got) / float64(time.Millisecond)
		wantMs := float64(want) / float64(time.Millisecond)
		if gotMs < wantMs*0.99 {
			t.Errorf("%s = %.1fms, below true value %.1fms", name, gotMs, wantMs)
		}
		if gotMs > wantMs*1.30 {
			t.Errorf("%s = %.1fms, more than 30%% above true %.1fms", name, gotMs, wantMs)
		}
	}
	check("p50", s.P50, 500*time.Millisecond)
	check("p90", s.P90, 900*time.Millisecond)
	check("p99", s.P99, 990*time.Millisecond)
}

func TestMaxAndMeanAreExact(t *testing.T) {
	h := NewLatencyHistogram()
	h.Observe(10 * time.Millisecond)
	h.Observe(20 * time.Millisecond)
	h.Observe(30 * time.Millisecond)

	s := h.Snapshot()
	if s.Max != 30*time.Millisecond {
		t.Errorf("max = %v, want 30ms", s.Max)
	}
	if s.Min != 10*time.Millisecond {
		t.Errorf("min = %v, want 10ms", s.Min)
	}
	if s.Mean != 20*time.Millisecond {
		t.Errorf("mean = %v, want 20ms", s.Mean)
	}
}

func TestOutlierGoesToOverflowBucket(t *testing.T) {
	h := NewLatencyHistogram()
	for i := 0; i < 99; i++ {
		h.Observe(time.Millisecond)
	}
	h.Observe(10 * time.Minute)

	s := h.Snapshot()
	if s.Max != 10*time.Minute {
		t.Errorf("max = %v, want the 10m outlier preserved", s.Max)
	}
	if s.P50 > 10*time.Millisecond {
		t.Errorf("p50 = %v, a single outlier should not move the median", s.P50)
	}
}

func TestNegativeDurationClamped(t *testing.T) {
	h := NewLatencyHistogram()
	h.Observe(-5 * time.Second)
	if s := h.Snapshot(); s.Count != 1 || s.Min < 0 {
		t.Fatalf("negative observation mishandled: %+v", s)
	}
}

func TestConcurrentObserve(t *testing.T) {
	h := NewLatencyHistogram()
	const goroutines, each = 16, 500

	var wg sync.WaitGroup
	for g := 0; g < goroutines; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < each; i++ {
				h.Observe(time.Duration(i%50) * time.Millisecond)
			}
		}()
	}
	wg.Wait()

	if got := h.Snapshot().Count; got != goroutines*each {
		t.Fatalf("count = %d, want %d; observations were lost", got, goroutines*each)
	}
}

func TestSnapshotDuringWritesIsConsistent(t *testing.T) {
	h := NewLatencyHistogram()
	stop := make(chan struct{})
	var wg sync.WaitGroup

	wg.Add(1)
	go func() {
		defer wg.Done()
		for {
			select {
			case <-stop:
				return
			default:
				h.Observe(time.Millisecond)
			}
		}
	}()

	for i := 0; i < 200; i++ {
		s := h.Snapshot()
		if s.Count > 0 && s.Mean <= 0 {
			t.Error("snapshot observed a torn state: positive count with zero mean")
			break
		}
		if math.IsNaN(s.P99Ms) {
			t.Error("p99 was NaN")
			break
		}
	}
	close(stop)
	wg.Wait()
}

// TestSnapshotRacesWithIncreasingObservations is a race-detector regression
// test. quantile used to read h.max directly, outside the mutex that Observe
// writes it under.
//
// TestSnapshotDuringWritesIsConsistent cannot catch that: it observes a
// constant 1ms, so h.max is written exactly once and there is never a
// concurrent write to race with. Only a strictly increasing series keeps
// h.max hot, which is why this test exists alongside it. Meaningful only
// under -race.
func TestSnapshotRacesWithIncreasingObservations(t *testing.T) {
	h := NewLatencyHistogram()
	stop := make(chan struct{})
	var writers sync.WaitGroup

	for g := 0; g < 4; g++ {
		writers.Add(1)
		go func() {
			defer writers.Done()
			for i := 1; ; i++ {
				select {
				case <-stop:
					return
				default:
					h.Observe(time.Duration(i) * time.Microsecond)
				}
			}
		}()
	}

	var readers sync.WaitGroup
	for g := 0; g < 4; g++ {
		readers.Add(1)
		go func() {
			defer readers.Done()
			for i := 0; i < 2000; i++ {
				s := h.Snapshot()
				if s.P99 > s.Max {
					t.Errorf("p99 %v exceeds max %v", s.P99, s.Max)
					return
				}
			}
		}()
	}
	readers.Wait()
	close(stop)
	writers.Wait()
}

// TestPercentileNeverExceedsMax is a regression test. Bucket upper bounds can
// overshoot every value actually recorded, which surfaced as a benchmark
// reporting p99 = 64.62ms against an observed max of 55.43ms. A percentile
// that exceeds a real measurement is not a rounding artifact, it is wrong.
func TestPercentileNeverExceedsMax(t *testing.T) {
	h := NewLatencyHistogram()
	for i := 0; i < 500; i++ {
		h.Observe(51 * time.Millisecond)
	}
	h.Observe(55 * time.Millisecond)

	s := h.Snapshot()
	for name, v := range map[string]time.Duration{"p50": s.P50, "p90": s.P90, "p99": s.P99} {
		if v > s.Max {
			t.Errorf("%s = %v exceeds observed max %v", name, v, s.Max)
		}
	}
}
