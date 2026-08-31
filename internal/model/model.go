// Package model defines the inference backend contract and a simulated
// implementation used for benchmarking the serving layer in isolation.
package model

import "context"

// Result is a single inference output.
type Result struct {
	Label string  `json:"label"`
	Score float32 `json:"score"`
}

// Runner executes inference over a batch of inputs.
//
// Implementations MUST return exactly len(inputs) results, in the same order
// as the inputs. The batcher relies on positional correspondence to fan
// results back to the correct waiting caller; violating it silently returns
// one caller's answer to a different caller.
//
// RunBatch may be called concurrently from multiple workers.
type Runner interface {
	RunBatch(ctx context.Context, inputs [][]float32) ([]Result, error)

	// MaxBatchSize is the largest batch the backend accepts. The batcher
	// clamps its own configured batch size to this value.
	MaxBatchSize() int

	Name() string
}
