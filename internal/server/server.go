// Package server exposes the batcher over HTTP.
//
// The interesting part here is status mapping. A saturated server that returns
// 500 for everything is unoperable: the caller cannot tell "retry me, I was
// never attempted" from "I broke, do not retry". Load shedding is 503 with
// Retry-After, deadline expiry is 504, and only genuine backend failure is
// 500.
package server

import (
	"context"
	"encoding/json"
	"errors"
	"log/slog"
	"net/http"
	"strconv"
	"sync"
	"sync/atomic"
	"time"

	"batchserve/internal/batch"
	"batchserve/internal/metrics"
)

type Config struct {
	// RequestTimeout bounds a single client request end to end. It becomes
	// the caller's context deadline, which is what lets the batcher drop
	// abandoned work instead of computing it.
	RequestTimeout time.Duration
	// MaxInputLen rejects oversized payloads before they reach the queue.
	MaxInputLen int
	// MaxBodyBytes caps the request body read.
	MaxBodyBytes int64
	// MaxBatchRequestItems caps items in one /predict/batch call.
	MaxBatchRequestItems int
}

func DefaultConfig() Config {
	return Config{
		RequestTimeout:       2 * time.Second,
		MaxInputLen:          4096,
		MaxBodyBytes:         1 << 20,
		MaxBatchRequestItems: 128,
	}
}

type Server struct {
	cfg     Config
	batcher *batch.Batcher
	metrics *metrics.Metrics
	log     *slog.Logger
}

func New(b *batch.Batcher, m *metrics.Metrics, cfg Config, log *slog.Logger) *Server {
	if log == nil {
		log = slog.Default()
	}
	return &Server{cfg: cfg, batcher: b, metrics: m, log: log}
}

func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("POST /predict", s.handlePredict)
	mux.HandleFunc("POST /predict/batch", s.handlePredictBatch)
	mux.HandleFunc("GET /healthz", s.handleHealth)
	mux.HandleFunc("GET /metrics", s.handleMetrics)
	return mux
}

type predictRequest struct {
	Input []float32 `json:"input"`
}

type predictResponse struct {
	Label     string  `json:"label"`
	Score     float32 `json:"score"`
	LatencyMs float64 `json:"latency_ms"`
	ServedBy  string  `json:"served_by"`
}

type errorResponse struct {
	Error string `json:"error"`
}

func (s *Server) handlePredict(w http.ResponseWriter, r *http.Request) {
	start := time.Now()

	var req predictRequest
	if !s.decode(w, r, &req) {
		return
	}
	if len(req.Input) == 0 {
		writeJSON(w, http.StatusBadRequest, errorResponse{"input must be a non-empty array"})
		return
	}
	if len(req.Input) > s.cfg.MaxInputLen {
		writeJSON(w, http.StatusRequestEntityTooLarge,
			errorResponse{"input exceeds max length " + strconv.Itoa(s.cfg.MaxInputLen)})
		return
	}

	ctx, cancel := context.WithTimeout(r.Context(), s.cfg.RequestTimeout)
	defer cancel()

	res, err := s.batcher.Submit(ctx, req.Input)
	elapsed := time.Since(start)
	if !shedAtAdmission(err) {
		s.metrics.EndToEnd.Observe(elapsed)
	}

	if err != nil {
		s.writeSubmitError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, predictResponse{
		Label:     res.Label,
		Score:     res.Score,
		LatencyMs: float64(elapsed.Nanoseconds()) / 1e6,
		ServedBy:  "batchserve",
	})
}

type batchRequest struct {
	Inputs [][]float32 `json:"inputs"`
}

type batchItem struct {
	Index int    `json:"index"`
	Label string `json:"label,omitempty"`
	// Score carries no omitempty: a genuine score of 0 (reachable when the
	// sigmoid underflows on a strongly negative input) is a result, not an
	// absent field, and Status already says whether it means anything.
	Score  float32 `json:"score"`
	Error  string  `json:"error,omitempty"`
	Status int     `json:"status"` // the code this item would get from /predict
}

type batchResponse struct {
	Results   []batchItem `json:"results"`
	Succeeded int         `json:"succeeded"`
	Failed    int         `json:"failed"`
	LatencyMs float64     `json:"latency_ms"`
}

// handlePredictBatch submits every item independently and concurrently. It
// does NOT pass them to the backend as one batch: the scheduler owns batching
// decisions, and letting a client hand-assemble batches would let one caller
// monopolise a backend call and starve everyone else.
//
// Items succeed and fail independently, so the response status summarises them
// rather than reporting on the HTTP request alone — see aggregateBatchStatus.
func (s *Server) handlePredictBatch(w http.ResponseWriter, r *http.Request) {
	start := time.Now()

	var req batchRequest
	if !s.decode(w, r, &req) {
		return
	}
	if len(req.Inputs) == 0 {
		writeJSON(w, http.StatusBadRequest, errorResponse{"inputs must be a non-empty array"})
		return
	}
	if len(req.Inputs) > s.cfg.MaxBatchRequestItems {
		writeJSON(w, http.StatusRequestEntityTooLarge,
			errorResponse{"too many items, max " + strconv.Itoa(s.cfg.MaxBatchRequestItems)})
		return
	}

	ctx, cancel := context.WithTimeout(r.Context(), s.cfg.RequestTimeout)
	defer cancel()

	items := make([]batchItem, len(req.Inputs))
	var attempted atomic.Bool

	// Backend failures are logged once for the whole request rather than once
	// per item: a 128-item batch hitting a dead backend should not write 128
	// identical lines. The raw error stays server-side, as it does on
	// /predict; the client gets the same generic message.
	var (
		logMu       sync.Mutex
		backendFail int
		sampleErr   error
	)

	var wg sync.WaitGroup
	for i, in := range req.Inputs {
		wg.Add(1)
		go func(i int, in []float32) {
			defer wg.Done()
			items[i].Index = i

			// Mirror /predict's own validation codes so an item means the same
			// thing whichever endpoint it arrived through.
			switch {
			case len(in) == 0:
				items[i].Status = http.StatusBadRequest
				items[i].Error = "input must be a non-empty array"
				return
			case len(in) > s.cfg.MaxInputLen:
				items[i].Status = http.StatusRequestEntityTooLarge
				items[i].Error = "input exceeds max length " + strconv.Itoa(s.cfg.MaxInputLen)
				return
			}

			res, err := s.batcher.Submit(ctx, in)
			if !shedAtAdmission(err) {
				attempted.Store(true)
			}
			if err != nil {
				status, msg := submitOutcome(err)
				items[i].Status = status
				items[i].Error = msg
				if status == http.StatusInternalServerError {
					logMu.Lock()
					backendFail++
					if sampleErr == nil {
						sampleErr = err
					}
					logMu.Unlock()
				}
				return
			}
			items[i].Status = http.StatusOK
			items[i].Label = res.Label
			items[i].Score = res.Score
		}(i, in)
	}
	wg.Wait()

	if backendFail > 0 {
		s.log.Error("inference failed for batch items",
			"failed_items", backendFail, "of", len(items), "err", sampleErr)
	}

	succeeded := 0
	for _, it := range items {
		if it.Status == http.StatusOK {
			succeeded++
		}
	}

	elapsed := time.Since(start)
	// Same rule as /predict: a request whose every item bounced off admission
	// control measures the cost of saying no, not the cost of serving.
	if attempted.Load() {
		s.metrics.EndToEnd.Observe(elapsed)
	}

	status := aggregateBatchStatus(items)
	if status == http.StatusServiceUnavailable {
		w.Header().Set("Retry-After", "1")
	}
	writeJSON(w, status, batchResponse{
		Results:   items,
		Succeeded: succeeded,
		Failed:    len(items) - succeeded,
		LatencyMs: float64(elapsed.Nanoseconds()) / 1e6,
	})
}

// aggregateBatchStatus reduces per-item outcomes to one response code.
//
// A batch where every item went the same way gets that exact code, so the
// common cases stay actionable from the status line alone: all shed is a
// retryable 503, all expired is a 504, all fine is a 200. Anything mixed is
// 207, which means precisely "the outcomes differ, read them per item".
//
// The point is that a batch where every item failed must not answer 200. That
// makes "did my work happen?" unanswerable without parsing the body, and any
// caller checking only the status silently treats total failure as success.
func aggregateBatchStatus(items []batchItem) int {
	first := items[0].Status
	for _, it := range items[1:] {
		if it.Status != first {
			return http.StatusMultiStatus
		}
	}
	return first
}

func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

func (s *Server) handleMetrics(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, s.metrics.Snapshot())
}

// shedAtAdmission reports whether the scheduler refused the work outright
// rather than attempting it. Such a request costs microseconds, so folding it
// into the latency histogram would make the server look fastest at the moment
// it is serving least.
func shedAtAdmission(err error) bool {
	return errors.Is(err, batch.ErrQueueFull) || errors.Is(err, batch.ErrShuttingDown)
}

// submitOutcome maps a scheduler error onto the status code and client-safe
// message a caller can act on without reading the log.
//
// Both endpoints go through here so they cannot drift: an item inside
// /predict/batch reports the same code it would have received had it been sent
// to /predict on its own. Backend errors collapse to a generic message because
// the raw text is internal detail; callers get the code, the log gets the error.
func submitOutcome(err error) (int, string) {
	switch {
	case err == nil:
		return http.StatusOK, ""
	case errors.Is(err, batch.ErrQueueFull):
		return http.StatusServiceUnavailable, "server at capacity, retry shortly"
	case errors.Is(err, batch.ErrShuttingDown):
		return http.StatusServiceUnavailable, "server shutting down"
	case errors.Is(err, batch.ErrDeadlineExceeded),
		errors.Is(err, context.DeadlineExceeded):
		return http.StatusGatewayTimeout, "request deadline exceeded"
	case errors.Is(err, context.Canceled):
		return http.StatusRequestTimeout, "client cancelled the request"
	default:
		return http.StatusInternalServerError, "inference failed"
	}
}

// writeSubmitError answers a single /predict request that did not succeed.
func (s *Server) writeSubmitError(w http.ResponseWriter, err error) {
	if errors.Is(err, context.Canceled) {
		// Client hung up. Nothing useful to write; the connection is gone.
		return
	}

	status, msg := submitOutcome(err)
	switch {
	case errors.Is(err, batch.ErrQueueFull):
		w.Header().Set("Retry-After", "1")
	case errors.Is(err, batch.ErrShuttingDown):
		w.Header().Set("Connection", "close")
	}
	if status == http.StatusInternalServerError {
		s.log.Error("inference failed", "err", err)
	}
	writeJSON(w, status, errorResponse{msg})
}

func (s *Server) decode(w http.ResponseWriter, r *http.Request, dst any) bool {
	r.Body = http.MaxBytesReader(w, r.Body, s.cfg.MaxBodyBytes)
	dec := json.NewDecoder(r.Body)
	dec.DisallowUnknownFields()
	if err := dec.Decode(dst); err != nil {
		// An oversized body is not malformed JSON. Reporting it as 400
		// "invalid JSON" sends the caller hunting for a syntax error that
		// isn't there; it is the same class of failure as an over-long input
		// array, which already answers 413.
		var tooLarge *http.MaxBytesError
		if errors.As(err, &tooLarge) {
			writeJSON(w, http.StatusRequestEntityTooLarge,
				errorResponse{"request body exceeds " + strconv.FormatInt(s.cfg.MaxBodyBytes, 10) + " bytes"})
			return false
		}
		writeJSON(w, http.StatusBadRequest, errorResponse{"invalid JSON body: " + err.Error()})
		return false
	}
	return true
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}
