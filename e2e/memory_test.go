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

//go:build (onnx && ORT) || (xla && XLA)

package e2e

import (
	"context"
	"fmt"
	"os"
	"runtime"
	"testing"
	"time"

	"github.com/antflydb/antfly-go/libaf/ai"
	"github.com/antflydb/termite/pkg/termite"
	"github.com/antflydb/termite/pkg/termite/lib/hugot"
	"go.uber.org/zap/zaptest"
)

// TestMemoryGrowthWithVaryingShapes demonstrates memory growth from JIT cache.
//
// This test creates inference requests with varying input lengths to trigger
// JIT recompilation for each unique shape. Memory should grow proportionally
// to the number of unique shapes encountered.
//
// Run with: make e2e
//
// Expected behavior:
// - With unlimited JIT cache: memory grows with each new shape
// - With limited JIT cache: memory stabilizes after cache fills
func TestMemoryGrowthWithVaryingShapes(t *testing.T) {
	// Ensure embedder model is downloaded (lazy download)
	ensureRegistryModel(t, "BAAI/bge-small-en-v1.5", ModelTypeEmbedder)

	embedderModelsDir := getEmbedderModelsDir()

	logger := zaptest.NewLogger(t)

	// Create session manager
	sm := hugot.NewSessionManager()
	defer sm.Close()

	// Create embedder registry
	registry, err := termite.NewEmbedderRegistry(
		termite.EmbedderConfig{
			ModelsDir:       embedderModelsDir,
			KeepAlive:       5 * time.Minute,
			MaxLoadedModels: 1,
		},
		sm,
		logger.Sugar().Desugar(),
	)
	if err != nil {
		t.Fatalf("Failed to create registry: %v", err)
	}
	defer registry.Close()

	// Get an embedder
	embedder, err := registry.Get("BAAI/bge-small-en-v1.5")
	if err != nil {
		t.Skipf("Model not available: %v", err)
	}

	ctx := context.Background()

	// Generate inputs with varying lengths to trigger JIT recompilation
	// Power-of-2 buckets: 8, 16, 32, 64, 128, 256, 512
	inputLengths := []int{5, 10, 20, 40, 80, 160, 320}

	var memStats runtime.MemStats

	t.Log("Starting memory growth test...")
	t.Log("Shape | HeapAlloc (MB) | Sys (MB) | NumGC")
	t.Log("------|----------------|----------|------")

	for i, length := range inputLengths {
		// Create input of specific length
		input := textToContentParts(generateTextOfLength(length))

		// Run inference (this may trigger JIT compilation for new shape)
		_, err := embedder.Embed(ctx, input)
		if err != nil {
			t.Errorf("Embed failed for length %d: %v", length, err)
			continue
		}

		// Force GC to get accurate heap stats
		runtime.GC()
		runtime.ReadMemStats(&memStats)

		t.Logf("%5d | %14.2f | %8.2f | %5d",
			length,
			float64(memStats.HeapAlloc)/1024/1024,
			float64(memStats.Sys)/1024/1024,
			memStats.NumGC,
		)

		// Small delay to let things settle
		time.Sleep(100 * time.Millisecond)

		// After a few iterations, check if memory is growing unboundedly
		if i > 3 && memStats.Sys > 2*1024*1024*1024 { // > 2GB
			t.Logf("WARNING: System memory usage exceeds 2GB after %d shapes", i+1)
		}
	}

	// Final memory check
	runtime.GC()
	runtime.ReadMemStats(&memStats)

	t.Logf("\nFinal memory state:")
	t.Logf("  HeapAlloc: %.2f MB", float64(memStats.HeapAlloc)/1024/1024)
	t.Logf("  HeapSys:   %.2f MB", float64(memStats.HeapSys)/1024/1024)
	t.Logf("  Sys:       %.2f MB", float64(memStats.Sys)/1024/1024)
	t.Logf("  NumGC:     %d", memStats.NumGC)
}

// TestMemoryWithRepeatedShapes verifies memory is stable when reusing shapes.
//
// With proper JIT caching, repeated inference with the same shape should
// NOT grow memory after the initial compilation.
func TestMemoryWithRepeatedShapes(t *testing.T) {
	// Ensure embedder model is downloaded (lazy download)
	ensureRegistryModel(t, "BAAI/bge-small-en-v1.5", ModelTypeEmbedder)

	embedderModelsDir := getEmbedderModelsDir()

	logger := zaptest.NewLogger(t)

	sm := hugot.NewSessionManager()
	defer sm.Close()

	registry, err := termite.NewEmbedderRegistry(
		termite.EmbedderConfig{
			ModelsDir:       embedderModelsDir,
			KeepAlive:       5 * time.Minute,
			MaxLoadedModels: 1,
		},
		sm,
		logger.Sugar().Desugar(),
	)
	if err != nil {
		t.Fatalf("Failed to create registry: %v", err)
	}
	defer registry.Close()

	embedder, err := registry.Get("BAAI/bge-small-en-v1.5")
	if err != nil {
		t.Skipf("Model not available: %v", err)
	}

	ctx := context.Background()

	// Use fixed-length input to avoid JIT recompilation
	input := textToContentParts(generateTextOfLength(50)) // Will be padded to 64 (power of 2)

	var memStats runtime.MemStats
	var initialMem, finalMem uint64

	// Warmup - first call triggers JIT compilation
	_, err = embedder.Embed(ctx, input)
	if err != nil {
		t.Fatalf("Warmup embed failed: %v", err)
	}

	runtime.GC()
	runtime.ReadMemStats(&memStats)
	initialMem = memStats.Sys

	t.Logf("Initial memory after warmup: %.2f MB", float64(initialMem)/1024/1024)

	// Run many inferences with same shape
	iterations := 100
	for i := 0; i < iterations; i++ {
		_, err := embedder.Embed(ctx, input)
		if err != nil {
			t.Errorf("Embed failed at iteration %d: %v", i, err)
			break
		}
	}

	runtime.GC()
	runtime.ReadMemStats(&memStats)
	finalMem = memStats.Sys

	t.Logf("Final memory after %d iterations: %.2f MB", iterations, float64(finalMem)/1024/1024)

	// Memory should not grow significantly (allow 20% margin for GC variance)
	if finalMem > initialMem*12/10 { // 120% of initial
		t.Errorf("Memory grew unexpectedly: initial=%.2f MB, final=%.2f MB (%.1f%% increase)",
			float64(initialMem)/1024/1024,
			float64(finalMem)/1024/1024,
			float64(finalMem-initialMem)/float64(initialMem)*100,
		)
	} else {
		t.Logf("Memory stable: %.1f%% change",
			float64(finalMem-initialMem)/float64(initialMem)*100)
	}
}

// generateTextOfLength creates a text string with approximately the given word count.
func generateTextOfLength(wordCount int) string {
	words := make([]byte, 0, wordCount*6)
	for i := 0; i < wordCount; i++ {
		word := fmt.Sprintf("word%d ", i%100)
		words = append(words, word...)
	}
	return string(words)
}

// textToContentParts converts a string to the [][]ai.ContentPart format expected by Embedder.
func textToContentParts(text string) [][]ai.ContentPart {
	return [][]ai.ContentPart{
		{ai.TextContent{Text: text}},
	}
}

// BenchmarkMemoryPerShape measures memory allocation per unique input shape.
func BenchmarkMemoryPerShape(b *testing.B) {
	embedderModelsDir := getEmbedderModelsDir()
	if _, err := os.Stat(embedderModelsDir); os.IsNotExist(err) {
		b.Skipf("Embedder models directory not found: %s (run a test first to download)", embedderModelsDir)
	}

	sm := hugot.NewSessionManager()
	defer sm.Close()

	registry, err := termite.NewEmbedderRegistry(
		termite.EmbedderConfig{
			ModelsDir:       embedderModelsDir,
			KeepAlive:       5 * time.Minute,
			MaxLoadedModels: 1,
		},
		sm,
		nil,
	)
	if err != nil {
		b.Fatalf("Failed to create registry: %v", err)
	}
	defer registry.Close()

	embedder, err := registry.Get("BAAI/bge-small-en-v1.5")
	if err != nil {
		b.Skipf("Model not available: %v", err)
	}

	ctx := context.Background()
	input := textToContentParts(generateTextOfLength(50))

	// Warmup
	embedder.Embed(ctx, input)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_, err := embedder.Embed(ctx, input)
		if err != nil {
			b.Fatalf("Embed failed: %v", err)
		}
	}
}
