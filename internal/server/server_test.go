package server

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"batchserve/internal/batch"
	"batchserve/internal/metrics"
	"batchserve/internal/model"
)

type slowRunner struct {
	delay time.Duration
	gate  chan struct{}
	mu    sync.Mutex
}

func (s *slowRunner) Name() string      { return "slow" }
func (s *slowRunner) MaxBatchSize() int { return 64 }

func (s *slowRunner) RunBatch(ctx context.Context, inputs [][]float32) ([]model.Result, error) {
	if s.gate != nil {
		select {
		case <-s.gate:
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
	if s.delay > 0 {
		select {
		case <-time.After(s.delay):
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
	out := make([]model.Result, len(inputs))
	for i := range inputs {
		out[i] = model.Result{Label: "positive", Score: 0.9}
	}
	return out, nil
}

// failingRunner always errors, with a message containing detail that must not
// reach a client.
type failingRunner struct{ msg string }

func (f *failingRunner) Name() string      { return "failing" }
func (f *failingRunner) MaxBatchSize() int { return 64 }
func (f *failingRunner) RunBatch(ctx context.Context, inputs [][]float32) ([]model.Result, error) {
	return nil, errors.New(f.msg)
}

func newTestServer(t *testing.T, r model.Runner, bc batch.Config, sc Config) *httptest.Server {
	t.Helper()
	srv, _ := newTestServerWithBatcher(t, r, bc, sc)
	return srv
}

// newTestServerWithBatcher also hands back the scheduler, for tests that need
// to drive its lifecycle (shutting it down mid-test to force a uniform shed).
func newTestServerWithBatcher(t *testing.T, r model.Runner, bc batch.Config, sc Config) (*httptest.Server, *batch.Batcher) {
	t.Helper()
	m := metrics.New()
	b, err := batch.New(r, bc, m)
	if err != nil {
		t.Fatalf("batch.New: %v", err)
	}
	srv := httptest.NewServer(New(b, m, sc, nil).Handler())
	t.Cleanup(func() {
		srv.Close()
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_ = b.Shutdown(ctx)
	})
	return srv, b
}

func post(t *testing.T, url, body string) *http.Response {
	t.Helper()
	resp, err := http.Post(url, "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatalf("post: %v", err)
	}
	return resp
}

func TestPredictHappyPath(t *testing.T) {
	srv := newTestServer(t, &slowRunner{},
		batch.Config{MaxBatchSize: 4, MaxWait: 5 * time.Millisecond,
			QueueCapacity: 64, Workers: 2, BackendTimeout: time.Second},
		DefaultConfig())

	resp := post(t, srv.URL+"/predict", `{"input":[0.5,0.5]}`)
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(resp.Body)
		t.Fatalf("status %d: %s", resp.StatusCode, b)
	}
	var out predictResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if out.Label == "" {
		t.Fatal("empty label in response")
	}
	if out.LatencyMs <= 0 {
		t.Fatal("latency_ms should be populated")
	}
}

func TestValidationRejectsBadInput(t *testing.T) {
	cfg := DefaultConfig()
	cfg.MaxInputLen = 4
	srv := newTestServer(t, &slowRunner{},
		batch.Config{MaxBatchSize: 4, MaxWait: time.Millisecond,
			QueueCapacity: 64, Workers: 1, BackendTimeout: time.Second},
		cfg)

	cases := []struct {
		name string
		body string
		want int
	}{
		{"malformed json", `{"input":`, http.StatusBadRequest},
		{"empty input", `{"input":[]}`, http.StatusBadRequest},
		{"missing field", `{}`, http.StatusBadRequest},
		{"unknown field", `{"input":[1],"rogue":true}`, http.StatusBadRequest},
		{"oversized input", `{"input":[1,2,3,4,5,6]}`, http.StatusRequestEntityTooLarge},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			resp := post(t, srv.URL+"/predict", tc.body)
			defer resp.Body.Close()
			if resp.StatusCode != tc.want {
				t.Fatalf("status = %d, want %d", resp.StatusCode, tc.want)
			}
		})
	}
}

// TestOversizedBodyReturns413NotBadRequest pins a distinction that used to be
// lost: exceeding MaxBodyBytes answered 400 "invalid JSON body", which sends
// the caller hunting for a syntax error that does not exist. It is a size
// failure, and the sibling case (an over-long input array) already said 413.
func TestOversizedBodyReturns413NotBadRequest(t *testing.T) {
	cfg := DefaultConfig()
	cfg.MaxBodyBytes = 64
	cfg.MaxInputLen = 100000 // ensure the body cap is what trips, not input length
	srv := newTestServer(t, &slowRunner{},
		batch.Config{MaxBatchSize: 4, MaxWait: time.Millisecond,
			QueueCapacity: 64, Workers: 1, BackendTimeout: time.Second},
		cfg)

	body := `{"input":[` + strings.Repeat("1,", 200) + `1]}`
	if int64(len(body)) <= cfg.MaxBodyBytes {
		t.Fatalf("test body of %d bytes does not exceed the %d-byte cap", len(body), cfg.MaxBodyBytes)
	}

	resp := post(t, srv.URL+"/predict", body)
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusRequestEntityTooLarge {
		t.Fatalf("status = %d, want 413", resp.StatusCode)
	}
	var out errorResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if !strings.Contains(out.Error, "body") {
		t.Errorf("error %q should name the body as the problem, not the JSON", out.Error)
	}
}

// TestShedRequestsStayOutOfLatencyHistogram guards a metric that lied exactly
// when it mattered. A 503 costs microseconds; folding those into
// end_to_end_latency made the percentiles look best at the moment the server
// was serving least. Shedding belongs in rejected_queue_full.
func TestShedRequestsStayOutOfLatencyHistogram(t *testing.T) {
	r := &slowRunner{gate: make(chan struct{})}
	srv := newTestServer(t, r,
		batch.Config{MaxBatchSize: 1, MaxWait: time.Millisecond,
			QueueCapacity: 1, Workers: 1, BackendTimeout: 10 * time.Second},
		DefaultConfig())

	var wg sync.WaitGroup
	var mu sync.Mutex
	var shed, served int

	for i := 0; i < 40; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			resp, err := http.Post(srv.URL+"/predict", "application/json",
				strings.NewReader(`{"input":[1]}`))
			if err != nil {
				return
			}
			io.Copy(io.Discard, resp.Body)
			resp.Body.Close()
			mu.Lock()
			if resp.StatusCode == http.StatusServiceUnavailable {
				shed++
			} else {
				served++
			}
			mu.Unlock()
		}()
	}
	close(r.gate)
	wg.Wait()

	if shed == 0 {
		t.Skip("no shedding occurred on this scheduling pass")
	}

	resp, err := http.Get(srv.URL + "/metrics")
	if err != nil {
		t.Fatalf("metrics: %v", err)
	}
	defer resp.Body.Close()
	var stats metrics.Stats
	if err := json.NewDecoder(resp.Body).Decode(&stats); err != nil {
		t.Fatalf("decode metrics: %v", err)
	}

	if int(stats.Rejected) != shed {
		t.Errorf("rejected_queue_full = %d, want %d (one per 503)", stats.Rejected, shed)
	}
	if int(stats.EndToEnd.Count) != served {
		t.Errorf("end_to_end_latency.count = %d, want %d (attempted requests only, excluding %d shed)",
			stats.EndToEnd.Count, served, shed)
	}
}

// TestSaturationReturns503NotTimeout checks the distinction an operator
// actually needs: shed load must be retryable (503), not indistinguishable
// from a server that broke.
func TestSaturationReturns503NotTimeout(t *testing.T) {
	r := &slowRunner{gate: make(chan struct{})}
	srv := newTestServer(t, r,
		batch.Config{MaxBatchSize: 1, MaxWait: time.Millisecond,
			QueueCapacity: 1, Workers: 1, BackendTimeout: 10 * time.Second},
		DefaultConfig())

	var wg sync.WaitGroup
	var mu sync.Mutex
	statuses := map[int]int{}

	for i := 0; i < 40; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			resp, err := http.Post(srv.URL+"/predict", "application/json",
				strings.NewReader(`{"input":[1]}`))
			if err != nil {
				return
			}
			io.Copy(io.Discard, resp.Body)
			resp.Body.Close()
			mu.Lock()
			statuses[resp.StatusCode]++
			mu.Unlock()
		}()
	}
	wg.Wait()
	close(r.gate)

	if statuses[http.StatusServiceUnavailable] == 0 {
		t.Fatalf("expected some 503s under saturation, got %v", statuses)
	}
	if statuses[http.StatusInternalServerError] > 0 {
		t.Fatalf("shed load must never surface as 500, got %v", statuses)
	}
}

func TestRetryAfterHeaderOnShed(t *testing.T) {
	r := &slowRunner{gate: make(chan struct{})}
	srv := newTestServer(t, r,
		batch.Config{MaxBatchSize: 1, MaxWait: time.Millisecond,
			QueueCapacity: 1, Workers: 1, BackendTimeout: 10 * time.Second},
		DefaultConfig())

	var wg sync.WaitGroup
	found := make(chan string, 40)
	for i := 0; i < 40; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			resp, err := http.Post(srv.URL+"/predict", "application/json",
				strings.NewReader(`{"input":[1]}`))
			if err != nil {
				return
			}
			io.Copy(io.Discard, resp.Body)
			resp.Body.Close()
			if resp.StatusCode == http.StatusServiceUnavailable {
				found <- resp.Header.Get("Retry-After")
			}
		}()
	}
	wg.Wait()
	close(r.gate)
	close(found)

	got := false
	for v := range found {
		got = true
		if v == "" {
			t.Fatal("503 response missing Retry-After header")
		}
	}
	if !got {
		t.Skip("no shedding occurred on this scheduling pass")
	}
}

func TestDeadlineReturns504(t *testing.T) {
	r := &slowRunner{delay: 500 * time.Millisecond}
	cfg := DefaultConfig()
	cfg.RequestTimeout = 30 * time.Millisecond

	srv := newTestServer(t, r,
		batch.Config{MaxBatchSize: 1, MaxWait: time.Millisecond,
			QueueCapacity: 64, Workers: 1, BackendTimeout: 5 * time.Second},
		cfg)

	resp := post(t, srv.URL+"/predict", `{"input":[1]}`)
	defer resp.Body.Close()
	io.Copy(io.Discard, resp.Body)

	if resp.StatusCode != http.StatusGatewayTimeout {
		t.Fatalf("status = %d, want 504", resp.StatusCode)
	}
}

func TestBatchEndpoint(t *testing.T) {
	srv := newTestServer(t, &slowRunner{},
		batch.Config{MaxBatchSize: 8, MaxWait: 5 * time.Millisecond,
			QueueCapacity: 256, Workers: 2, BackendTimeout: time.Second},
		DefaultConfig())

	resp := post(t, srv.URL+"/predict/batch", `{"inputs":[[1,2],[3,4],[5,6]]}`)
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d", resp.StatusCode)
	}
	var out batchResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(out.Results) != 3 {
		t.Fatalf("got %d results, want 3", len(out.Results))
	}
	for i, r := range out.Results {
		if r.Index != i {
			t.Fatalf("result %d has index %d; ordering must be preserved", i, r.Index)
		}
		if r.Error != "" {
			t.Fatalf("result %d errored: %s", i, r.Error)
		}
		if r.Status != http.StatusOK {
			t.Fatalf("result %d has status %d, want 200", i, r.Status)
		}
	}
	if out.Succeeded != 3 || out.Failed != 0 {
		t.Fatalf("succeeded/failed = %d/%d, want 3/0", out.Succeeded, out.Failed)
	}
}

// TestBatchPartialFailureReturns207 pins the rule that a batch reports what
// actually happened. Mixed outcomes cannot be summarised by a single code, so
// the response is 207 and the caller reads per-item status.
func TestBatchPartialFailureReturns207(t *testing.T) {
	cfg := DefaultConfig()
	cfg.MaxInputLen = 4
	srv := newTestServer(t, &slowRunner{},
		batch.Config{MaxBatchSize: 8, MaxWait: 5 * time.Millisecond,
			QueueCapacity: 256, Workers: 2, BackendTimeout: time.Second},
		cfg)

	// index 0 valid, 1 empty (400), 2 valid, 3 over-long (413).
	resp := post(t, srv.URL+"/predict/batch", `{"inputs":[[1,2],[],[3,4],[1,2,3,4,5,6]]}`)
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusMultiStatus {
		t.Fatalf("status = %d, want 207 for a mixed batch", resp.StatusCode)
	}
	var out batchResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if out.Succeeded != 2 || out.Failed != 2 {
		t.Fatalf("succeeded/failed = %d/%d, want 2/2", out.Succeeded, out.Failed)
	}

	want := []int{http.StatusOK, http.StatusBadRequest, http.StatusOK, http.StatusRequestEntityTooLarge}
	for i, w := range want {
		if out.Results[i].Status != w {
			t.Errorf("item %d status = %d, want %d", i, out.Results[i].Status, w)
		}
	}
	// Per-item codes must match what /predict would have said on its own.
	single := post(t, srv.URL+"/predict", `{"input":[1,2,3,4,5,6]}`)
	single.Body.Close()
	if single.StatusCode != out.Results[3].Status {
		t.Errorf("/predict said %d for the same input the batch called %d; the endpoints have drifted",
			single.StatusCode, out.Results[3].Status)
	}
}

// TestBatchAllFailedDoesNotReturn200 is the substance of the change. A batch
// in which nothing succeeded used to answer 200, so a caller checking only the
// status treated total failure as success.
func TestBatchAllFailedDoesNotReturn200(t *testing.T) {
	srv := newTestServer(t, &slowRunner{},
		batch.Config{MaxBatchSize: 8, MaxWait: 5 * time.Millisecond,
			QueueCapacity: 256, Workers: 2, BackendTimeout: time.Second},
		DefaultConfig())

	resp := post(t, srv.URL+"/predict/batch", `{"inputs":[[],[],[]]}`)
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusOK {
		t.Fatal("a batch where every item failed must not answer 200")
	}
	// Uniform failures collapse to the precise code rather than 207.
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400: every item failed the same way", resp.StatusCode)
	}
	var out batchResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if out.Succeeded != 0 || out.Failed != 3 {
		t.Fatalf("succeeded/failed = %d/%d, want 0/3", out.Succeeded, out.Failed)
	}
}

// TestBatchAllShedReturns503WithRetryAfter drives the uniform-shed path
// deterministically by draining the scheduler first, so every item comes back
// ErrShuttingDown rather than relying on a saturation race.
func TestBatchAllShedReturns503WithRetryAfter(t *testing.T) {
	srv, b := newTestServerWithBatcher(t, &slowRunner{},
		batch.Config{MaxBatchSize: 8, MaxWait: 5 * time.Millisecond,
			QueueCapacity: 256, Workers: 2, BackendTimeout: time.Second},
		DefaultConfig())

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := b.Shutdown(ctx); err != nil {
		t.Fatalf("Shutdown: %v", err)
	}

	resp := post(t, srv.URL+"/predict/batch", `{"inputs":[[1,2],[3,4],[5,6]]}`)
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusServiceUnavailable {
		t.Fatalf("status = %d, want 503: every item was refused", resp.StatusCode)
	}
	if resp.Header.Get("Retry-After") == "" {
		t.Error("503 batch response missing Retry-After; the work was never attempted and is safe to retry")
	}
	var out batchResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if out.Succeeded != 0 || out.Failed != 3 {
		t.Fatalf("succeeded/failed = %d/%d, want 0/3", out.Succeeded, out.Failed)
	}
	for i, r := range out.Results {
		if r.Status != http.StatusServiceUnavailable {
			t.Errorf("item %d status = %d, want 503", i, r.Status)
		}
	}
}

// TestBatchErrorsDoNotLeakBackendDetail keeps the two endpoints honest about
// what they tell a client. /predict answers a generic "inference failed" and
// logs the real error; the batch path used to hand back err.Error() verbatim.
func TestBatchErrorsDoNotLeakBackendDetail(t *testing.T) {
	r := &failingRunner{msg: "postgres://user:hunter2@10.0.0.5/weights unreachable"}
	srv := newTestServer(t, r,
		batch.Config{MaxBatchSize: 8, MaxWait: 5 * time.Millisecond,
			QueueCapacity: 256, Workers: 2, BackendTimeout: time.Second},
		DefaultConfig())

	resp := post(t, srv.URL+"/predict/batch", `{"inputs":[[1,2],[3,4]]}`)
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusInternalServerError {
		t.Fatalf("status = %d, want 500: every item hit the backend error", resp.StatusCode)
	}
	body, _ := io.ReadAll(resp.Body)
	if strings.Contains(string(body), "hunter2") || strings.Contains(string(body), "10.0.0.5") {
		t.Fatalf("backend error detail leaked to the client: %s", body)
	}
}

func TestHealthAndMetrics(t *testing.T) {
	srv := newTestServer(t, &slowRunner{},
		batch.Config{MaxBatchSize: 4, MaxWait: time.Millisecond,
			QueueCapacity: 64, Workers: 1, BackendTimeout: time.Second},
		DefaultConfig())

	resp, err := http.Get(srv.URL + "/healthz")
	if err != nil {
		t.Fatalf("healthz: %v", err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("healthz status = %d", resp.StatusCode)
	}

	post(t, srv.URL+"/predict", `{"input":[1,2]}`).Body.Close()

	resp, err = http.Get(srv.URL + "/metrics")
	if err != nil {
		t.Fatalf("metrics: %v", err)
	}
	defer resp.Body.Close()

	var stats metrics.Stats
	if err := json.NewDecoder(resp.Body).Decode(&stats); err != nil {
		t.Fatalf("decode metrics: %v", err)
	}
	if stats.Completed == 0 {
		t.Fatal("metrics did not record the completed request")
	}
	if stats.AverageBatchSize <= 0 {
		t.Fatal("average batch size not reported")
	}
}

func TestMethodNotAllowed(t *testing.T) {
	srv := newTestServer(t, &slowRunner{},
		batch.Config{MaxBatchSize: 4, MaxWait: time.Millisecond,
			QueueCapacity: 64, Workers: 1, BackendTimeout: time.Second},
		DefaultConfig())

	resp, err := http.Get(srv.URL + "/predict")
	if err != nil {
		t.Fatalf("get: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusMethodNotAllowed {
		t.Fatalf("status = %d, want 405", resp.StatusCode)
	}
}
