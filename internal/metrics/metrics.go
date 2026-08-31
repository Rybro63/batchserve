package metrics

import (
	"sync/atomic"
	"time"
)

// Metrics is the full instrumentation surface for the serving layer.
//
// The counters are split finely on purpose. "Requests failed" is not a useful
// number when debugging a saturated server; you need to know whether load was
// shed at admission (Rejected), abandoned while queued (DroppedDeadline), or
// broke in the backend (BackendErrors). Those three have different fixes.
type Metrics struct {
	start time.Time

	Accepted         atomic.Int64 // admitted to the queue
	Rejected         atomic.Int64 // refused at admission, queue full
	RejectedShutdown atomic.Int64 // refused because the scheduler is draining
	DroppedDeadline  atomic.Int64 // client context expired before execution
	Completed        atomic.Int64 // returned a result
	BackendErrors    atomic.Int64 // backend returned an error
	BatchesRun       atomic.Int64
	ItemsBatched     atomic.Int64

	// Wall time from HTTP handler entry to response, for requests the
	// scheduler actually attempted. Load shed at admission is deliberately
	// excluded: a rejection costs microseconds, and mixing those in makes the
	// percentiles look best exactly when the server is failing most. Shedding
	// is counted in Rejected, which is where you should look for it.
	EndToEnd *Histogram
	// Time spent waiting in queue before the batch was dispatched.
	QueueWait *Histogram
	// Time spent inside Runner.RunBatch.
	Inference *Histogram
	// Realised batch sizes, in items. Exact, not bucketed — see SizeHistogram.
	BatchSize *SizeHistogram
}

func New() *Metrics {
	return &Metrics{
		start:     time.Now(),
		EndToEnd:  NewLatencyHistogram(),
		QueueWait: NewLatencyHistogram(),
		Inference: NewLatencyHistogram(),
		BatchSize: NewSizeHistogram(),
	}
}

// Stats is the JSON shape served at /metrics.
type Stats struct {
	UptimeSeconds float64 `json:"uptime_seconds"`

	Accepted         int64 `json:"accepted"`
	Rejected         int64 `json:"rejected_queue_full"`
	RejectedShutdown int64 `json:"rejected_shutting_down"`
	DroppedDeadline  int64 `json:"dropped_deadline_exceeded"`
	Completed        int64 `json:"completed"`
	BackendErrors    int64 `json:"backend_errors"`

	BatchesRun       int64   `json:"batches_run"`
	ItemsBatched     int64   `json:"items_batched"`
	AverageBatchSize float64 `json:"average_batch_size"`

	Throughput float64 `json:"completed_per_second"`

	EndToEnd  Snapshot     `json:"end_to_end_latency"`
	QueueWait Snapshot     `json:"queue_wait"`
	Inference Snapshot     `json:"inference"`
	BatchSize SizeSnapshot `json:"batch_size"`
}

func (m *Metrics) Snapshot() Stats {
	uptime := time.Since(m.start).Seconds()
	batches := m.BatchesRun.Load()
	items := m.ItemsBatched.Load()
	completed := m.Completed.Load()

	s := Stats{
		UptimeSeconds:    uptime,
		Accepted:         m.Accepted.Load(),
		Rejected:         m.Rejected.Load(),
		RejectedShutdown: m.RejectedShutdown.Load(),
		DroppedDeadline:  m.DroppedDeadline.Load(),
		Completed:        completed,
		BackendErrors:    m.BackendErrors.Load(),
		BatchesRun:       batches,
		ItemsBatched:     items,
		EndToEnd:         m.EndToEnd.Snapshot(),
		QueueWait:        m.QueueWait.Snapshot(),
		Inference:        m.Inference.Snapshot(),
		BatchSize:        m.BatchSize.Snapshot(),
	}
	if batches > 0 {
		s.AverageBatchSize = float64(items) / float64(batches)
	}
	if uptime > 0 {
		s.Throughput = float64(completed) / uptime
	}
	return s
}
