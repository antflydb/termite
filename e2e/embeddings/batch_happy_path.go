//go:build onnx && ORT

// Happy Path E2E Test for PooledHugotEmbedder
//
// This test verifies normal usage patterns work correctly.
//
// Run:
//   export ONNXRUNTIME_ROOT=$PWD/onnxruntime
//   export DYLD_LIBRARY_PATH=$ONNXRUNTIME_ROOT/darwin-arm64/lib:$DYLD_LIBRARY_PATH
//   go test -v -tags="onnx,ORT" -run TestHappyPath ./pkg/termite/lib/embeddings/

package embeddings

import (
	"context"
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/antflydb/antfly-go/libaf/ai"
	"go.uber.org/zap"
)

// TestHappyPath_SingleEmbed tests basic single-threaded usage.
func TestHappyPath_SingleEmbed(t *testing.T) {
	modelPath := findModelPath(t)
	if modelPath == "" {
		t.Skip("Model not found")
	}
	logger := zap.NewNop()

	embedder, err := NewPooledHugotEmbedder(modelPath, "model.onnx", 2, logger)
	if err != nil {
		t.Fatalf("Failed to create embedder: %v", err)
	}
	defer embedder.Close()

	ctx := context.Background()
	contents := [][]ai.ContentPart{
		{ai.TextContent{Text: "Hello world"}},
		{ai.TextContent{Text: "This is a test"}},
		{ai.TextContent{Text: "Embeddings are useful"}},
	}

	result, err := embedder.Embed(ctx, contents)
	if err != nil {
		t.Fatalf("Embed failed: %v", err)
	}

	if len(result) != 3 {
		t.Errorf("Expected 3 embeddings, got %d", len(result))
	}

	for i, emb := range result {
		if len(emb) == 0 {
			t.Errorf("Embedding %d is empty", i)
		}
		t.Logf("Embedding %d: dim=%d, first_val=%.4f", i, len(emb), emb[0])
	}
}

// TestHappyPath_MultipleSequentialEmbeds tests multiple sequential calls.
func TestHappyPath_MultipleSequentialEmbeds(t *testing.T) {
	modelPath := findModelPath(t)
	if modelPath == "" {
		t.Skip("Model not found")
	}
	logger := zap.NewNop()

	embedder, err := NewPooledHugotEmbedder(modelPath, "model.onnx", 2, logger)
	if err != nil {
		t.Fatalf("Failed to create embedder: %v", err)
	}
	defer embedder.Close()

	ctx := context.Background()

	for i := 0; i < 5; i++ {
		contents := [][]ai.ContentPart{
			{ai.TextContent{Text: fmt.Sprintf("Sequential test %d", i)}},
		}

		result, err := embedder.Embed(ctx, contents)
		if err != nil {
			t.Fatalf("Embed %d failed: %v", i, err)
		}

		if len(result) != 1 {
			t.Errorf("Embed %d: expected 1 result, got %d", i, len(result))
		}
	}

	t.Log("5 sequential embeds completed successfully")
}

// TestHappyPath_ConcurrentEmbeds tests concurrent usage within pool limits.
func TestHappyPath_ConcurrentEmbeds(t *testing.T) {
	modelPath := findModelPath(t)
	if modelPath == "" {
		t.Skip("Model not found")
	}
	logger := zap.NewNop()

	poolSize := 2
	embedder, err := NewPooledHugotEmbedder(modelPath, "model.onnx", poolSize, logger)
	if err != nil {
		t.Fatalf("Failed to create embedder: %v", err)
	}
	defer embedder.Close()

	ctx := context.Background()
	numWorkers := 10
	embedsPerWorker := 5

	var wg sync.WaitGroup
	errors := make(chan error, numWorkers*embedsPerWorker)

	start := time.Now()

	for w := 0; w < numWorkers; w++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()

			for i := 0; i < embedsPerWorker; i++ {
				contents := [][]ai.ContentPart{
					{ai.TextContent{Text: fmt.Sprintf("Worker %d embed %d", workerID, i)}},
				}

				_, err := embedder.Embed(ctx, contents)
				if err != nil {
					errors <- fmt.Errorf("worker %d embed %d: %w", workerID, i, err)
				}
			}
		}(w)
	}

	wg.Wait()
	close(errors)

	duration := time.Since(start)
	totalOps := numWorkers * embedsPerWorker

	var errs []error
	for err := range errors {
		errs = append(errs, err)
	}

	if len(errs) > 0 {
		for _, err := range errs {
			t.Errorf("Error: %v", err)
		}
		t.Fatalf("%d errors occurred", len(errs))
	}

	t.Logf("%d concurrent embeds completed in %v (%.1f ops/sec)",
		totalOps, duration, float64(totalOps)/duration.Seconds())
}

// TestHappyPath_CloseAfterAllComplete tests proper close after work is done.
func TestHappyPath_CloseAfterAllComplete(t *testing.T) {
	modelPath := findModelPath(t)
	if modelPath == "" {
		t.Skip("Model not found")
	}
	logger := zap.NewNop()

	embedder, err := NewPooledHugotEmbedder(modelPath, "model.onnx", 2, logger)
	if err != nil {
		t.Fatalf("Failed to create embedder: %v", err)
	}

	ctx := context.Background()

	// Do some work
	for i := 0; i < 3; i++ {
		contents := [][]ai.ContentPart{
			{ai.TextContent{Text: fmt.Sprintf("Test %d", i)}},
		}
		_, err := embedder.Embed(ctx, contents)
		if err != nil {
			t.Fatalf("Embed %d failed: %v", i, err)
		}
	}

	// Close after all work is done - should succeed
	err = embedder.Close()
	if err != nil {
		t.Errorf("Close() returned error: %v", err)
	}

	t.Log("Close() after all embeds complete: success")
}

// TestHappyPath_LargeBatch tests handling of larger batches.
func TestHappyPath_LargeBatch(t *testing.T) {
	modelPath := findModelPath(t)
	if modelPath == "" {
		t.Skip("Model not found")
	}
	logger := zap.NewNop()

	embedder, err := NewPooledHugotEmbedder(modelPath, "model.onnx", 2, logger)
	if err != nil {
		t.Fatalf("Failed to create embedder: %v", err)
	}
	defer embedder.Close()

	ctx := context.Background()

	// Create a batch of 20 texts
	batchSize := 20
	contents := make([][]ai.ContentPart, batchSize)
	for i := 0; i < batchSize; i++ {
		contents[i] = []ai.ContentPart{
			ai.TextContent{Text: fmt.Sprintf("Large batch test sentence number %d with some extra text", i)},
		}
	}

	start := time.Now()
	result, err := embedder.Embed(ctx, contents)
	duration := time.Since(start)

	if err != nil {
		t.Fatalf("Large batch embed failed: %v", err)
	}

	if len(result) != batchSize {
		t.Errorf("Expected %d embeddings, got %d", batchSize, len(result))
	}

	t.Logf("Batch of %d texts embedded in %v", batchSize, duration)
}

// TestHappyPath_ContextCancellation tests that context cancellation is handled.
func TestHappyPath_ContextCancellation(t *testing.T) {
	modelPath := findModelPath(t)
	if modelPath == "" {
		t.Skip("Model not found")
	}
	logger := zap.NewNop()

	embedder, err := NewPooledHugotEmbedder(modelPath, "model.onnx", 2, logger)
	if err != nil {
		t.Fatalf("Failed to create embedder: %v", err)
	}
	defer embedder.Close()

	// Create already-cancelled context
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	contents := [][]ai.ContentPart{
		{ai.TextContent{Text: "This should fail due to cancelled context"}},
	}

	_, err = embedder.Embed(ctx, contents)
	if err == nil {
		t.Error("Expected error with cancelled context, got nil")
	} else {
		t.Logf("Cancelled context correctly returned error: %v", err)
	}
}

// TestHappyPath_EmptyInput tests handling of empty input.
func TestHappyPath_EmptyInput(t *testing.T) {
	modelPath := findModelPath(t)
	if modelPath == "" {
		t.Skip("Model not found")
	}
	logger := zap.NewNop()

	embedder, err := NewPooledHugotEmbedder(modelPath, "model.onnx", 2, logger)
	if err != nil {
		t.Fatalf("Failed to create embedder: %v", err)
	}
	defer embedder.Close()

	ctx := context.Background()
	contents := [][]ai.ContentPart{}

	result, err := embedder.Embed(ctx, contents)
	if err != nil {
		t.Errorf("Empty input should not error: %v", err)
	}

	if len(result) != 0 {
		t.Errorf("Expected 0 results for empty input, got %d", len(result))
	}

	t.Log("Empty input handled correctly")
}
