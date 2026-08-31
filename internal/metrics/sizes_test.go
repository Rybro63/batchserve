package metrics

import (
	"sync"
	"testing"
)

func TestSizeHistogramEmpty(t *testing.T) {
	h := NewSizeHistogram()
	s := h.Snapshot()
	if s.Count != 0 || s.P99 != 0 || s.Mean != 0 {
		t.Fatalf("empty distribution should report zeros, got %+v", s)
	}
}

// TestSizePercentilesAreExact is the point of this type. The latency histogram
// reports bucket upper bounds; because batch sizes are small integers with one
// counter each, these percentiles are the observed values themselves.
func TestSizePercentilesAreExact(t *testing.T) {
	h := NewSizeHistogram()
	for i := 1; i <= 100; i++ {
		h.Observe(i)
	}
	s := h.Snapshot()

	if s.Count != 100 {
		t.Fatalf("count = %d, want 100", s.Count)
	}
	if s.Min != 1 || s.Max != 100 {
		t.Errorf("min/max = %d/%d, want 1/100", s.Min, s.Max)
	}
	if s.Mean != 50.5 {
		t.Errorf("mean = %v, want exactly 50.5", s.Mean)
	}
	if s.P50 != 50 || s.P90 != 90 || s.P99 != 99 {
		t.Errorf("p50/p90/p99 = %d/%d/%d, want exactly 50/90/99", s.P50, s.P90, s.P99)
	}
}

// TestSizeDistributionSeparatesBimodal is why an average alone is not enough.
// A run split evenly between batches of 1 and batches of 31 averages 16, a
// size that never once occurred; the percentiles show the two real modes.
func TestSizeDistributionSeparatesBimodal(t *testing.T) {
	h := NewSizeHistogram()
	for i := 0; i < 500; i++ {
		h.Observe(1)
		h.Observe(31)
	}
	s := h.Snapshot()

	if s.Mean != 16 {
		t.Fatalf("mean = %v, want 16", s.Mean)
	}
	if s.P50 != 1 {
		t.Errorf("p50 = %d, want 1: half the batches were size 1", s.P50)
	}
	if s.P90 != 31 {
		t.Errorf("p90 = %d, want 31: the upper mode must be visible", s.P90)
	}
}

// TestSizeHistogramDoesNotCollapseSmallValues guards the bug this type
// replaced: batch sizes were fed to the latency histogram as nanosecond
// durations, and every size from 1 to a few hundred fell below its first
// bucket bound of 100us. Every batch reported identically.
func TestSizeHistogramDoesNotCollapseSmallValues(t *testing.T) {
	h := NewSizeHistogram()
	for i := 0; i < 100; i++ {
		h.Observe(1)
	}
	for i := 0; i < 100; i++ {
		h.Observe(64)
	}
	s := h.Snapshot()
	if s.P50 == s.P99 {
		t.Fatalf("p50 and p99 both %d; distinct sizes were collapsed into one bucket", s.P50)
	}
}

func TestSizeHistogramOverflowAndNegative(t *testing.T) {
	h := NewSizeHistogram()
	h.Observe(-3) // clamped to 0
	h.Observe(maxTrackedSize * 4)

	s := h.Snapshot()
	if s.Count != 2 {
		t.Fatalf("count = %d, want 2", s.Count)
	}
	if s.Min != 0 {
		t.Errorf("min = %d, want 0 (negative clamped)", s.Min)
	}
	if s.Max != maxTrackedSize*4 {
		t.Errorf("max = %d, want the overflow value preserved", s.Max)
	}
	if s.P99 != maxTrackedSize*4 {
		t.Errorf("p99 = %d, want the overflow value", s.P99)
	}
}

func TestSizeHistogramConcurrent(t *testing.T) {
	h := NewSizeHistogram()
	const goroutines, each = 16, 500

	var wg sync.WaitGroup
	for g := 0; g < goroutines; g++ {
		wg.Add(1)
		go func(g int) {
			defer wg.Done()
			for i := 0; i < each; i++ {
				h.Observe(g + 1)
			}
		}(g)
	}
	// Snapshot concurrently so -race covers the read path too.
	done := make(chan struct{})
	go func() {
		defer close(done)
		for i := 0; i < 500; i++ {
			_ = h.Snapshot()
		}
	}()
	wg.Wait()
	<-done

	if got := h.Snapshot().Count; got != goroutines*each {
		t.Fatalf("count = %d, want %d; observations were lost", got, goroutines*each)
	}
}
