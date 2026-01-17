// Copyright 2025 Antfly, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//go:build onnx && ORT

package embeddings

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/antflydb/antfly-go/libaf/ai"
	"go.uber.org/zap"
)

// TestCloseWhileEmbedding tests the race condition between Close() and Embed().
//
// Code inspection found that Close() has no synchronization with Embed():
//
//   func (p *PooledHugotEmbedder) Close() error {
//       if p.session != nil && !p.sessionShared {
//           return p.session.Destroy()  // No lock, no WaitGroup
//       }
//       return nil
//   }
//
// Hypothesis: If Close() is called while Embed() is running, the session
// could be destroyed mid-inference, causing undefined behavior.
//
// This test verifies whether:
// 1. Hugot/ONNX has internal protection (blocks Destroy until inference completes)
// 2. Or the race causes crashes/panics
// 3. Or silent corruption occurs
func TestCloseWhileEmbedding(t *testing.T) {
	modelPath := findModelPath(t)
	if modelPath == "" {
		t.Skip("Model not found, skipping close race test")
	}

	const poolSize = 2
	logger := zap.NewNop()

	// Create embedder that OWNS its session (sessionShared=false)
	// This is important - shared sessions won't trigger the race because
	// Close() skips session.Destroy() for shared sessions.
	embedder, err := NewPooledHugotEmbedder(modelPath, "model.onnx", poolSize, logger)
	if err != nil {
		t.Fatalf("Failed to create embedder: %v", err)
	}

	// Track what happens
	var embedStarted sync.WaitGroup
	var embedErr atomic.Value // stores error
	var embedPanicked atomic.Bool
	var closeErr atomic.Value

	embedStarted.Add(1)
	ctx := context.Background()

	// Start a slow embed operation with many texts
	go func() {
		defer func() {
			if r := recover(); r != nil {
				embedPanicked.Store(true)
				t.Logf("Embed PANICKED: %v", r)
			}
		}()

		// Use multiple texts to make inference take longer
		contents := make([][]ai.ContentPart, 50)
		for i := 0; i < len(contents); i++ {
			contents[i] = []ai.ContentPart{
				ai.TextContent{Text: fmt.Sprintf("This is test sentence number %d for the close race test. We want inference to take a while.", i)},
			}
		}

		embedStarted.Done() // Signal that we're about to start

		_, err := embedder.Embed(ctx, contents)
		if err != nil {
			embedErr.Store(err)
			t.Logf("Embed returned error: %v", err)
		} else {
			t.Log("Embed completed successfully (no error)")
		}
	}()

	// Wait for embed to start, then close immediately
	embedStarted.Wait()
	time.Sleep(10 * time.Millisecond) // Let inference begin

	t.Log("Calling Close() while Embed() is running...")
	if err := embedder.Close(); err != nil {
		closeErr.Store(err)
		t.Logf("Close returned error: %v", err)
	} else {
		t.Log("Close completed successfully (no error)")
	}

	// Wait a bit for embed to finish or crash
	time.Sleep(2 * time.Second)

	// Report findings
	if embedPanicked.Load() {
		t.Error("BUG CONFIRMED: Embed panicked when Close was called during inference")
		t.Log("Severity: HIGH - session.Destroy() is immediate and causes crash")
	} else if e := embedErr.Load(); e != nil {
		t.Logf("Embed returned error after Close: %v", e)
		t.Log("This could indicate partial protection or timing-dependent behavior")
	} else {
		t.Log("No panic or error detected - possible scenarios:")
		t.Log("  1. Hugot's session.Destroy() blocks until inference completes (safe)")
		t.Log("  2. ONNX Runtime has internal reference counting (safe)")
		t.Log("  3. We got lucky with timing (race exists but wasn't triggered)")
	}
}

// TestMultipleCloseIsSafe verifies that calling Close() multiple times
// doesn't cause issues (tests assumption A5: session destroy idempotence).
func TestMultipleCloseIsSafe(t *testing.T) {
	modelPath := findModelPath(t)
	if modelPath == "" {
		t.Skip("Model not found, skipping multiple close test")
	}

	const poolSize = 2
	logger := zap.NewNop()

	embedder, err := NewPooledHugotEmbedder(modelPath, "model.onnx", poolSize, logger)
	if err != nil {
		t.Fatalf("Failed to create embedder: %v", err)
	}

	// Do one successful embed first
	ctx := context.Background()
	contents := [][]ai.ContentPart{
		{ai.TextContent{Text: "test before close"}},
	}
	_, err = embedder.Embed(ctx, contents)
	if err != nil {
		t.Fatalf("Initial embed failed: %v", err)
	}

	// Now close multiple times
	var panicCount atomic.Int32
	var wg sync.WaitGroup

	for i := 0; i < 5; i++ {
		wg.Add(1)
		go func(attempt int) {
			defer wg.Done()
			defer func() {
				if r := recover(); r != nil {
					panicCount.Add(1)
					t.Logf("Close attempt %d PANICKED: %v", attempt, r)
				}
			}()

			err := embedder.Close()
			if err != nil {
				t.Logf("Close attempt %d returned error: %v", attempt, err)
			} else {
				t.Logf("Close attempt %d succeeded", attempt)
			}
		}(i)
	}

	wg.Wait()

	if panicCount.Load() > 0 {
		t.Errorf("Multiple Close() calls caused %d panics - assumption A5 violated", panicCount.Load())
	} else {
		t.Log("Multiple Close() calls are safe (assumption A5 validated)")
	}
}

// TestEmbedAfterClose verifies behavior when Embed is called after Close.
// This tests the "use-after-close" scenario from the TLA+ model.
func TestEmbedAfterClose(t *testing.T) {
	modelPath := findModelPath(t)
	if modelPath == "" {
		t.Skip("Model not found, skipping embed-after-close test")
	}

	const poolSize = 2
	logger := zap.NewNop()

	embedder, err := NewPooledHugotEmbedder(modelPath, "model.onnx", poolSize, logger)
	if err != nil {
		t.Fatalf("Failed to create embedder: %v", err)
	}

	// Close first
	err = embedder.Close()
	if err != nil {
		t.Fatalf("Close failed: %v", err)
	}
	t.Log("Embedder closed")

	// Now try to embed
	var panicked atomic.Bool
	defer func() {
		if r := recover(); r != nil {
			panicked.Store(true)
			t.Logf("Embed after Close PANICKED: %v", r)
		}
	}()

	ctx := context.Background()
	contents := [][]ai.ContentPart{
		{ai.TextContent{Text: "test after close"}},
	}

	_, err = embedder.Embed(ctx, contents)
	if err != nil {
		t.Logf("Embed after Close returned error: %v", err)
		t.Log("This is expected behavior - the embedder gracefully rejects calls after Close")
	} else {
		t.Log("Embed after Close succeeded - the session destruction may have been skipped (sessionShared=true) or ONNX allows this")
	}

	if panicked.Load() {
		t.Error("BUG: Embed after Close caused a panic")
	} else {
		t.Log("Embed after Close did not panic (some level of protection exists)")
	}
}
