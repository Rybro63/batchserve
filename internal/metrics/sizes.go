package metrics

import (
	"math"
	"sync"
)

// maxTrackedSize bounds the per-size counter table. Realised batch sizes are
// capped by the backend's max batch, which is far below this; anything larger
// lands in the overflow bucket and is still reflected in Count, Mean and Max.
const maxTrackedSize = 1024

// SizeHistogram records small non-negative integer observations exactly.
//
// This is deliberately not the duration Histogram. Batch sizes are bounded by
// the backend's maximum, so one counter per distinct size costs a few kilobytes
// and needs no bucket estimation at all — percentiles here are exact, not upper
// bounds. Routing item counts through the latency buckets instead would put
// every batch in the first bucket (whose bound is 100us, versus a batch size of
// at most a few hundred "nanoseconds") and report the result in milliseconds,
// which is worse than having no metric.
type SizeHistogram struct {
	mu       sync.Mutex
	counts   []uint64 // counts[i] is the number of observations of exactly i
	overflow uint64   // observations >= maxTrackedSize
	total    uint64
	sum      uint64
	max      int
	min      int
}

func NewSizeHistogram() *SizeHistogram {
	return &SizeHistogram{
		counts: make([]uint64, maxTrackedSize),
		min:    math.MaxInt,
	}
}

func (h *SizeHistogram) Observe(n int) {
	if n < 0 {
		n = 0
	}

	h.mu.Lock()
	if n < len(h.counts) {
		h.counts[n]++
	} else {
		h.overflow++
	}
	h.total++
	h.sum += uint64(n)
	if n > h.max {
		h.max = n
	}
	if n < h.min {
		h.min = n
	}
	h.mu.Unlock()
}

// SizeSnapshot is an immutable read of the distribution at one instant.
// Percentiles are exact observed values, so they carry no unit suffix and need
// no clamping against Max the way the bucketed latency percentiles do.
type SizeSnapshot struct {
	Count uint64  `json:"count"`
	Mean  float64 `json:"mean"`
	Min   int     `json:"min"`
	Max   int     `json:"max"`
	P50   int     `json:"p50"`
	P90   int     `json:"p90"`
	P99   int     `json:"p99"`
}

func (h *SizeHistogram) Snapshot() SizeSnapshot {
	h.mu.Lock()
	total := h.total
	counts := make([]uint64, len(h.counts))
	copy(counts, h.counts)
	sum := h.sum
	max := h.max
	min := h.min
	h.mu.Unlock()

	s := SizeSnapshot{Count: total}
	if total == 0 {
		return s
	}
	s.Mean = float64(sum) / float64(total)
	s.Min = min
	s.Max = max
	s.P50 = quantileOfSizes(counts, total, max, 0.50)
	s.P90 = quantileOfSizes(counts, total, max, 0.90)
	s.P99 = quantileOfSizes(counts, total, max, 0.99)
	return s
}

// quantileOfSizes takes every value it needs as a parameter so it cannot read
// mutable state outside the lock. See Histogram.quantile for why that matters.
//
// counts is indexed by size, so the index at which the cumulative count crosses
// the target IS the answer — no bucket bound to report. If the target falls
// past the table the observation was in the overflow bucket, and max is the
// only exact value available.
func quantileOfSizes(counts []uint64, total uint64, max int, q float64) int {
	target := uint64(math.Ceil(q * float64(total)))
	if target == 0 {
		target = 1
	}
	var cum uint64
	for size, c := range counts {
		cum += c
		if cum >= target {
			return size
		}
	}
	return max
}
