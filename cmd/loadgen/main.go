// Command loadgen drives the server at a fixed arrival rate and reports
// client-observed latency.
//
//	go run ./cmd/loadgen -rate 400 -duration 20s
//
// This is an OPEN-loop generator: requests are issued on a schedule regardless
// of whether earlier ones have returned. That matters. A closed-loop generator
// (N workers each looping "send, wait, send") stops offering load exactly when
// the server slows down, so queueing delay never appears in the numbers. That
// is coordinated omission, and it is why naive benchmarks report beautiful p99s
// for servers that fall over in production.
package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math/rand"
	"net/http"
	"os"
	"sync"
	"sync/atomic"
	"time"

	"batchserve/internal/metrics"
)

// behindScheduleTolerance is the fraction of send slots that may run late
// before the run is called into question. Occasional lateness is OS timer
// granularity, not a broken generator.
const behindScheduleTolerance = 0.01

func main() {
	var (
		url         = flag.String("url", "http://localhost:8080", "server base URL")
		rate        = flag.Int("rate", 200, "target requests per second")
		duration    = flag.Duration("duration", 20*time.Second, "test duration")
		inputLen    = flag.Int("input-len", 16, "length of each input vector")
		timeout     = flag.Duration("timeout", 5*time.Second, "client timeout")
		maxInflight = flag.Int("max-inflight", 20000, "safety cap on concurrent requests")
		warmup      = flag.Duration("warmup", 2*time.Second, "discard results from this initial period")
	)
	flag.Parse()

	// Without these guards -rate 0 divides by zero into an infinite interval
	// and an absurd -rate rounds the interval to zero, both of which used to
	// panic inside the scheduling loop rather than say what was wrong.
	if *rate <= 0 {
		fmt.Fprintf(os.Stderr, "-rate must be > 0, got %d\n", *rate)
		os.Exit(2)
	}
	if time.Duration(float64(time.Second)/float64(*rate)) <= 0 {
		fmt.Fprintf(os.Stderr, "-rate %d is too high to schedule (sub-nanosecond interval)\n", *rate)
		os.Exit(2)
	}
	if *maxInflight < 1 {
		fmt.Fprintf(os.Stderr, "-max-inflight must be >= 1, got %d\n", *maxInflight)
		os.Exit(2)
	}
	// A non-positive duration schedules nothing, which would divide by zero in
	// the schedule-slip ratio at the end.
	if *duration <= 0 {
		fmt.Fprintf(os.Stderr, "-duration must be > 0, got %v\n", *duration)
		os.Exit(2)
	}
	if *warmup < 0 {
		fmt.Fprintf(os.Stderr, "-warmup must be >= 0, got %v\n", *warmup)
		os.Exit(2)
	}
	if *warmup >= *duration {
		fmt.Fprintf(os.Stderr, "-warmup %v must be less than -duration %v, or nothing is measured\n",
			*warmup, *duration)
		os.Exit(2)
	}

	client := &http.Client{
		Timeout: *timeout,
		Transport: &http.Transport{
			MaxIdleConns:        *maxInflight,
			MaxIdleConnsPerHost: *maxInflight,
			MaxConnsPerHost:     0,
			IdleConnTimeout:     90 * time.Second,
		},
	}

	var (
		sent      atomic.Int64
		skipped   atomic.Int64
		ok        atomic.Int64
		okPost    atomic.Int64
		shed      atomic.Int64
		timedOut  atomic.Int64
		failed    atomic.Int64
		transport atomic.Int64
	)
	hist := metrics.NewLatencyHistogram()

	sem := make(chan struct{}, *maxInflight)
	var wg sync.WaitGroup

	interval := time.Duration(float64(time.Second) / float64(*rate))

	start := time.Now()
	warmupUntil := start.Add(*warmup)
	deadline := start.Add(*duration)

	fmt.Printf("driving %s at %d req/s for %v (warmup %v)\n", *url, *rate, *duration, *warmup)

	rng := rand.New(rand.NewSource(time.Now().UnixNano()))
	body := makeBody(rng, *inputLen)

	// Request i is due at start + i*interval, computed from start rather than
	// from "now" so that a slow iteration cannot quietly reduce the offered
	// rate. A time.Ticker is the wrong tool here: its channel buffers exactly
	// one tick, and any tick arriving while this loop is busy is dropped with
	// no record of it — the generator would offer less load than asked for and
	// report nothing amiss, which is the coordinated omission this command
	// exists to avoid.
	var scheduled, behind int64
	for i := int64(0); ; i++ {
		due := start.Add(time.Duration(i) * interval)
		if !due.Before(deadline) {
			break
		}
		if wait := time.Until(due); wait > 0 {
			time.Sleep(wait)
		} else if -wait > interval {
			// More than a full slot late: this host cannot issue requests as
			// fast as -rate demands, so the offered load is below target and
			// the run should be treated as suspect rather than believed.
			behind++
		}
		scheduled++

		select {
		case sem <- struct{}{}:
		default:
			// Inflight cap hit: this request was never issued. Kept separate
			// from transport errors, which are requests that WERE issued and
			// failed — conflating them hides whether the generator or the
			// server ran out of capacity.
			skipped.Add(1)
			continue
		}

		wg.Add(1)
		counted := !due.Before(warmupUntil)
		go func() {
			defer wg.Done()
			defer func() { <-sem }()

			sent.Add(1)
			t0 := time.Now()
			resp, err := client.Post(*url+"/predict", "application/json", bytes.NewReader(body))
			elapsed := time.Since(t0)

			if err != nil {
				transport.Add(1)
				return
			}
			_, _ = io.Copy(io.Discard, resp.Body)
			resp.Body.Close()

			switch resp.StatusCode {
			case http.StatusOK:
				ok.Add(1)
				if counted {
					okPost.Add(1)
					hist.Observe(elapsed)
				}
			case http.StatusServiceUnavailable:
				shed.Add(1)
			case http.StatusGatewayTimeout:
				timedOut.Add(1)
			default:
				failed.Add(1)
			}
		}()
	}

	wg.Wait()
	wall := time.Since(start)
	// wall >= duration > warmup, enforced at flag-parse time, so this is
	// always positive.
	measured := wall - *warmup

	snap := hist.Snapshot()

	fmt.Printf("\n=== client-observed ===\n")
	fmt.Printf("wall time        %.2fs (measured window %.2fs)\n", wall.Seconds(), measured.Seconds())
	fmt.Printf("scheduled        %d (target %d req/s)\n", scheduled, *rate)
	fmt.Printf("sent             %d\n", sent.Load())
	fmt.Printf("skipped (cap)    %d\n", skipped.Load())
	fmt.Printf("ok               %d\n", ok.Load())
	fmt.Printf("shed (503)       %d\n", shed.Load())
	fmt.Printf("timeout (504)    %d\n", timedOut.Load())
	fmt.Printf("other 4xx/5xx    %d\n", failed.Load())
	fmt.Printf("transport errors %d\n", transport.Load())
	// Goodput and the percentiles below are both post-warmup over the measured
	// window, so they describe the same set of requests.
	fmt.Printf("goodput          %.1f req/s (post-warmup)\n", float64(okPost.Load())/measured.Seconds())

	// A handful of late slots is just OS timer granularity. A sustained slip
	// means this host cannot offer the requested rate, so the server was never
	// actually tested at it — and that invalidates the run rather than
	// blemishing it.
	slip := float64(behind) / float64(scheduled)
	fmt.Printf("behind schedule  %d (%.1f%%)\n", behind, slip*100)
	if slip > behindScheduleTolerance {
		fmt.Printf("\n!! generator missed %.1f%% of its send slots, above the %.0f%% tolerance.\n",
			slip*100, behindScheduleTolerance*100)
		fmt.Printf("   Offered load was below the %d req/s target, so these numbers describe\n", *rate)
		fmt.Printf("   a slower test than you asked for. Lower -rate or use a separate host.\n\n")
	}
	fmt.Printf("latency (successful requests, post-warmup)\n")
	fmt.Printf("  p50  %.2f ms\n", snap.P50Ms)
	fmt.Printf("  p90  %.2f ms\n", snap.P90Ms)
	fmt.Printf("  p99  %.2f ms\n", snap.P99Ms)
	fmt.Printf("  max  %.2f ms\n", snap.MaxMs)

	if s, err := fetchStats(client, *url); err == nil {
		fmt.Printf("\n=== server-reported ===\n")
		fmt.Printf("batches run      %d\n", s.BatchesRun)
		fmt.Printf("items batched    %d\n", s.ItemsBatched)
		fmt.Printf("avg batch size   %.2f\n", s.AverageBatchSize)
		fmt.Printf("rejected         %d\n", s.Rejected)
		fmt.Printf("dropped deadline %d\n", s.DroppedDeadline)
		fmt.Printf("backend errors   %d\n", s.BackendErrors)
		fmt.Printf("queue wait p99   %.2f ms\n", s.QueueWait.P99Ms)
		fmt.Printf("inference p50    %.2f ms\n", s.Inference.P50Ms)
	} else {
		fmt.Fprintf(os.Stderr, "\ncould not fetch server metrics: %v\n", err)
	}
}

func makeBody(rng *rand.Rand, n int) []byte {
	in := make([]float32, n)
	for i := range in {
		in[i] = rng.Float32()*2 - 1
	}
	b, _ := json.Marshal(map[string]any{"input": in})
	return b
}

func fetchStats(c *http.Client, base string) (metrics.Stats, error) {
	var s metrics.Stats
	resp, err := c.Get(base + "/metrics")
	if err != nil {
		return s, err
	}
	defer resp.Body.Close()
	if err := json.NewDecoder(resp.Body).Decode(&s); err != nil {
		return s, err
	}
	return s, nil
}
