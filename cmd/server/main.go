// Command server runs the batching inference server.
//
//	go run ./cmd/server -batch-size 32 -max-wait 5ms -workers 2
//
// Setting -batch-size 1 disables batching and gives you the baseline to
// compare against.
package main

import (
	"context"
	"errors"
	"flag"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"batchserve/internal/batch"
	"batchserve/internal/metrics"
	"batchserve/internal/model"
	"batchserve/internal/server"
)

func main() {
	var (
		addr           = flag.String("addr", ":8080", "listen address")
		batchSize      = flag.Int("batch-size", 32, "max batch size (1 disables batching)")
		maxWait        = flag.Duration("max-wait", 5*time.Millisecond, "max time to hold a partial batch")
		workers        = flag.Int("workers", 2, "concurrent backend calls")
		queueCap       = flag.Int("queue", 1024, "admission queue capacity")
		reqTimeout     = flag.Duration("request-timeout", 2*time.Second, "per-request deadline")
		fixedCost      = flag.Duration("sim-fixed", 18*time.Millisecond, "simulated per-call fixed cost")
		marginalCost   = flag.Duration("sim-marginal", 350*time.Microsecond, "simulated per-item cost")
		failureRate    = flag.Float64("sim-failure-rate", 0, "injected backend failure rate [0,1]")
		shutdownGrace  = flag.Duration("shutdown-grace", 15*time.Second, "max time for each shutdown phase (connection drain, then queue drain)")
		backendTimeout = flag.Duration("backend-timeout", 30*time.Second, "cap on one backend call")
	)
	flag.Parse()

	log := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelInfo}))

	sim := model.NewSim(model.SimConfig{
		FixedCost:    *fixedCost,
		MarginalCost: *marginalCost,
		MaxBatch:     256,
		FailureRate:  *failureRate,
	})

	m := metrics.New()

	batcher, err := batch.New(sim, batch.Config{
		MaxBatchSize:   *batchSize,
		MaxWait:        *maxWait,
		QueueCapacity:  *queueCap,
		Workers:        *workers,
		BackendTimeout: *backendTimeout,
	}, m)
	if err != nil {
		log.Error("failed to build batcher", "err", err)
		os.Exit(1)
	}

	srvCfg := server.DefaultConfig()
	srvCfg.RequestTimeout = *reqTimeout
	api := server.New(batcher, m, srvCfg, log)

	httpSrv := &http.Server{
		Addr:              *addr,
		Handler:           api.Handler(),
		ReadHeaderTimeout: 5 * time.Second,
		ReadTimeout:       15 * time.Second,
		WriteTimeout:      30 * time.Second,
		IdleTimeout:       60 * time.Second,
	}

	go func() {
		log.Info("listening",
			"addr", *addr,
			"batch_size", batcher.Config().MaxBatchSize,
			"max_wait", *maxWait,
			"workers", *workers,
			"queue_capacity", *queueCap,
		)
		if err := httpSrv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			log.Error("listen failed", "err", err)
			os.Exit(1)
		}
	}()

	// Shutdown order matters. Stop accepting connections first so no new work
	// enters the queue, then drain the batcher. Reversing these would let the
	// HTTP layer keep admitting requests into a closed scheduler.
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, os.Interrupt, syscall.SIGTERM)
	sig := <-sigCh
	log.Info("shutdown signal received, draining", "signal", sig.String())

	// Each phase gets its own budget. Sharing one deadline means a slow
	// connection drain can consume the entire grace period and hand the
	// batcher an already-expired context, so Shutdown returns instantly and
	// queued work is abandoned — the drain silently not happening is exactly
	// what the grace period exists to prevent.
	httpCtx, cancelHTTP := context.WithTimeout(context.Background(), *shutdownGrace)
	defer cancelHTTP()
	if err := httpSrv.Shutdown(httpCtx); err != nil {
		log.Warn("http shutdown incomplete", "err", err)
	}

	drainCtx, cancelDrain := context.WithTimeout(context.Background(), *shutdownGrace)
	defer cancelDrain()
	if err := batcher.Shutdown(drainCtx); err != nil {
		log.Warn("batcher drain incomplete", "err", err)
	}

	s := m.Snapshot()
	log.Info("final stats",
		"completed", s.Completed,
		"rejected", s.Rejected,
		"rejected_shutting_down", s.RejectedShutdown,
		"dropped_deadline", s.DroppedDeadline,
		"backend_errors", s.BackendErrors,
		"avg_batch_size", s.AverageBatchSize,
		"p50_batch_size", s.BatchSize.P50,
		"p99_ms", s.EndToEnd.P99Ms,
	)
	log.Info("shutdown complete")
}
