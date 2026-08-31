// Package metrics provides the minimal instrumentation needed to characterise
// the serving layer: a bucketed latency histogram and a set of counters.
//
// This is deliberately not Prometheus. The point of the project is that the
// interesting mechanisms are visible, and a histogram is about sixty lines.
package metrics

import (
	"math"
	"sync"
	"time"
)

// Histogram records durations into exponentially-spaced buckets.
//
// Percentiles are estimates: a reported value is the upper bound of the
// bucket the true value falls in. Buckets grow by 1.25x, so a reported value
// overstates the truth by at most (1 - 1/1.25) = 20% and never understates it.
// That is fine for comparing configurations and is the same tradeoff every
// production histogram makes. Do not read these numbers as exact.
type Histogram struct {
	mu      sync.Mutex
	bounds  []time.Duration
	counts  []uint64
	total   uint64
	sum     time.Duration
	max     time.Duration
	minSeen time.Duration
}

// NewLatencyHistogram builds buckets from 100us to ~65s, growing by ~1.25x.
func NewLatencyHistogram() *Histogram {
	var bounds []time.Duration
	cur := 100 * time.Microsecond
	for cur < 65*time.Second {
		bounds = append(bounds, cur)
		next := time.Duration(float64(cur) * 1.25)
		if next == cur {
			next = cur + time.Microsecond
		}
		cur = next
	}
	return &Histogram{
		bounds:  bounds,
		counts:  make([]uint64, len(bounds)+1), // +1 overflow bucket
		minSeen: time.Duration(math.MaxInt64),
	}
}

func (h *Histogram) Observe(d time.Duration) {
	if d < 0 {
		d = 0
	}
	idx := h.bucketFor(d)

	h.mu.Lock()
	h.counts[idx]++
	h.total++
	h.sum += d
	if d > h.max {
		h.max = d
	}
	if d < h.minSeen {
		h.minSeen = d
	}
	h.mu.Unlock()
}

// bucketFor binary-searches the bucket whose upper bound first exceeds d.
func (h *Histogram) bucketFor(d time.Duration) int {
	lo, hi := 0, len(h.bounds)
	for lo < hi {
		mid := (lo + hi) / 2
		if h.bounds[mid] < d {
			lo = mid + 1
		} else {
			hi = mid
		}
	}
	return lo
}

// Snapshot is an immutable read of the histogram at one instant.
type Snapshot struct {
	Count uint64        `json:"count"`
	Mean  time.Duration `json:"-"`
	Min   time.Duration `json:"-"`
	Max   time.Duration `json:"-"`
	P50   time.Duration `json:"-"`
	P90   time.Duration `json:"-"`
	P99   time.Duration `json:"-"`

	// Millisecond mirrors for JSON output.
	MeanMs float64 `json:"mean_ms"`
	MinMs  float64 `json:"min_ms"`
	MaxMs  float64 `json:"max_ms"`
	P50Ms  float64 `json:"p50_ms"`
	P90Ms  float64 `json:"p90_ms"`
	P99Ms  float64 `json:"p99_ms"`
}

func (h *Histogram) Snapshot() Snapshot {
	h.mu.Lock()
	total := h.total
	counts := make([]uint64, len(h.counts))
	copy(counts, h.counts)
	sum := h.sum
	max := h.max
	min := h.minSeen
	h.mu.Unlock()

	s := Snapshot{Count: total}
	if total == 0 {
		return s
	}
	s.Mean = sum / time.Duration(total)
	s.Min = min
	s.Max = max
	s.P50 = h.quantile(counts, total, max, 0.50)
	s.P90 = h.quantile(counts, total, max, 0.90)
	s.P99 = h.quantile(counts, total, max, 0.99)

	s.MeanMs = ms(s.Mean)
	s.MinMs = ms(s.Min)
	s.MaxMs = ms(s.Max)
	s.P50Ms = ms(s.P50)
	s.P90Ms = ms(s.P90)
	s.P99Ms = ms(s.P99)
	return s
}

// quantile works entirely from values the caller snapshotted under the lock.
// max in particular MUST be passed in rather than read from h: Observe mutates
// h.max under the mutex, and reading the field here (outside it) is a data race
// that the race detector only catches when observations are increasing.
func (h *Histogram) quantile(counts []uint64, total uint64, max time.Duration, q float64) time.Duration {
	target := uint64(math.Ceil(q * float64(total)))
	if target == 0 {
		target = 1
	}
	var cum uint64
	for i, c := range counts {
		cum += c
		if cum >= target {
			if i >= len(h.bounds) {
				return max
			}
			// A bucket's upper bound can exceed every value actually observed
			// in it, which produces the nonsense of p99 > max. Clamp: a
			// reported percentile must never exceed a real measurement.
			if h.bounds[i] > max {
				return max
			}
			return h.bounds[i]
		}
	}
	return max
}

func ms(d time.Duration) float64 {
	return float64(d.Nanoseconds()) / 1e6
}
