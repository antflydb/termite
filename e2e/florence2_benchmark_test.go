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

package e2e

import (
	"context"
	"image"
	"os"
	"runtime/pprof"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"go.uber.org/zap"

	"github.com/antflydb/termite/pkg/termite/lib/backends"
	"github.com/antflydb/termite/pkg/termite/lib/reading"
	_ "github.com/gomlx/gomlx/backends/simplego/highway"
)

const (
	florence2ModelRepo = "onnx-community/Florence-2-base-ft"
	florence2ModelName = "onnx-community/Florence-2-base-ft"
)

// loadFlorence2TestImage loads the sample PDF page image used for Florence-2 benchmarks.
func loadFlorence2TestImage(t testing.TB) image.Image {
	t.Helper()

	pageImagePath := "testdata/sample-page-1.png"
	if _, err := os.Stat(pageImagePath); os.IsNotExist(err) {
		t.Skip("Pre-rendered page image not found at:", pageImagePath)
	}

	imgFile, err := os.Open(pageImagePath)
	require.NoError(t, err)
	defer imgFile.Close()

	img, _, err := image.Decode(imgFile)
	require.NoError(t, err)
	return img
}

// createReaderWithBackend creates a PooledReader pinned to a specific backend.
func createReaderWithBackend(t testing.TB, modelPath string, backendType backends.BackendType) (*reading.PooledReader, error) {
	t.Helper()

	logger := zap.NewNop()
	sessionManager := backends.NewSessionManager()
	sessionManager.SetPriority([]backends.BackendSpec{
		{Backend: backendType, Device: backends.DeviceAuto},
	})

	cfg := &reading.PooledReaderConfig{
		ModelPath: modelPath,
		PoolSize:  1,
		Logger:    logger,
	}

	reader, _, err := reading.NewPooledReader(cfg, sessionManager, []string{string(backendType)})
	return reader, err
}

// TestCompareFlorence2Backends compares Florence-2 OCR inference timing across all
// available backends (ONNX, XLA, CoreML, Go).
func TestCompareFlorence2Backends(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping Florence-2 backend comparison in short mode")
	}

	modelPath := ensureHuggingFaceModel(t, florence2ModelName, florence2ModelRepo, ModelTypeReader)
	t.Logf("Using model: %s", modelPath)

	img := loadFlorence2TestImage(t)
	t.Logf("Image size: %dx%d", img.Bounds().Dx(), img.Bounds().Dy())

	prompt := reading.FlorencePrompt(reading.FlorenceOCR)
	ctx := context.Background()

	availableBackends := backends.ListAvailable()
	t.Logf("Available backends: %d", len(availableBackends))
	for _, b := range availableBackends {
		t.Logf("  - %s (%s)", b.Name(), b.Type())
	}

	results := make(map[backends.BackendType]time.Duration)

	for _, backend := range availableBackends {
		backendType := backend.Type()
		t.Run(string(backendType), func(t *testing.T) {
			reader, err := createReaderWithBackend(t, modelPath, backendType)
			if err != nil {
				t.Skipf("Cannot create Florence-2 reader with %s backend: %v", backendType, err)
			}
			defer reader.Close()

			// Warmup
			_, err = reader.Read(ctx, []image.Image{img}, prompt, 128)
			if err != nil {
				t.Fatalf("Warmup failed: %v", err)
			}

			// Timed iterations
			const iterations = 5
			iterTimes := make([]time.Duration, iterations)
			totalStart := time.Now()
			for i := 0; i < iterations; i++ {
				iterStart := time.Now()
				res, err := reader.Read(ctx, []image.Image{img}, prompt, 512)
				if err != nil {
					t.Fatalf("Read failed on iteration %d: %v", i, err)
				}
				iterTimes[i] = time.Since(iterStart)
				if i == 0 {
					t.Logf("Sample output (%d chars): %q", len(res[0].Text), truncate(res[0].Text, 200))
				}
			}
			elapsed := time.Since(totalStart)
			avg := elapsed / iterations

			results[backendType] = avg
			t.Logf("Backend %s: %v avg per image (%d iterations)", backendType, avg, iterations)
			for i, d := range iterTimes {
				t.Logf("  iter %d: %v", i, d)
			}
		})
	}

	// Summary
	t.Log("\n=== Florence-2 Backend Comparison Summary ===")
	for backend, duration := range results {
		t.Logf("  %s: %v per image", backend, duration)
	}

	// Calculate speedups relative to Go backend
	if goTime, ok := results[backends.BackendGo]; ok {
		for backend, duration := range results {
			if backend != backends.BackendGo {
				speedup := float64(goTime) / float64(duration)
				t.Logf("  %s speedup vs Go: %.2fx", backend, speedup)
			}
		}
	}
}

// BenchmarkFlorence2Backends runs a proper Go benchmark comparing Florence-2
// OCR inference across all available backends.
func BenchmarkFlorence2Backends(b *testing.B) {
	modelPath := ""
	// Look for model already downloaded
	patterns := []string{
		testModelsDir + "/readers/onnx-community/Florence-2-base-ft",
	}
	for _, p := range patterns {
		if _, err := os.Stat(p); err == nil {
			modelPath = p
			break
		}
	}
	if path := os.Getenv("FLORENCE2_MODEL_PATH"); path != "" {
		modelPath = path
	}
	if modelPath == "" {
		b.Skip("No Florence-2 model found — run TestCompareFlorence2Backends first to download one")
	}

	img := loadFlorence2TestImage(b)
	prompt := reading.FlorencePrompt(reading.FlorenceOCR)
	ctx := context.Background()

	for _, backend := range backends.ListAvailable() {
		backendType := backend.Type()
		b.Run(string(backendType), func(b *testing.B) {
			reader, err := createReaderWithBackend(b, modelPath, backendType)
			if err != nil {
				b.Skipf("Cannot create Florence-2 reader with %s: %v", backendType, err)
			}
			defer reader.Close()

			// Warmup
			_, _ = reader.Read(ctx, []image.Image{img}, prompt, 128)

			b.ResetTimer()
			for b.Loop() {
				_, err = reader.Read(ctx, []image.Image{img}, prompt, 512)
				if err != nil {
					b.Fatalf("Read failed: %v", err)
				}
			}

			imgsPerSec := float64(b.N) / b.Elapsed().Seconds()
			b.ReportMetric(imgsPerSec, "images/sec")
		})
	}
}

// TestFlorence2Profile runs Florence-2 on the Go backend with CPU profiling enabled.
// The profile is written to /tmp/florence2_go.prof and can be analysed with:
//
//	go tool pprof -http=:8080 /tmp/florence2_go.prof
//
// Run:
//
//	GOEXPERIMENT=simd go test -v -run TestFlorence2Profile -timeout 15m ./e2e/
func TestFlorence2Profile(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping Florence-2 profiling in short mode")
	}

	modelPath := ensureHuggingFaceModel(t, florence2ModelName, florence2ModelRepo, ModelTypeReader)
	img := loadFlorence2TestImage(t)
	prompt := reading.FlorencePrompt(reading.FlorenceOCR)
	ctx := context.Background()

	reader, err := createReaderWithBackend(t, modelPath, backends.BackendGo)
	require.NoError(t, err, "creating Go backend reader")
	defer reader.Close()

	// Warmup: compile all graph shapes so profiling captures steady-state execution.
	t.Log("Warmup...")
	_, err = reader.Read(ctx, []image.Image{img}, prompt, 128)
	require.NoError(t, err, "warmup")
	t.Log("Warmup done")

	// Start CPU profile
	profPath := "/tmp/florence2_go.prof"
	f, err := os.Create(profPath)
	require.NoError(t, err)
	defer f.Close()

	require.NoError(t, pprof.StartCPUProfile(f))

	// Run 2 profiled iterations (enough to get a representative sample
	// without waiting 5+ minutes).
	const iterations = 2
	for i := 0; i < iterations; i++ {
		start := time.Now()
		res, err := reader.Read(ctx, []image.Image{img}, prompt, 128)
		require.NoError(t, err, "iteration %d", i)
		t.Logf("iter %d: %v  output=%d chars", i, time.Since(start), len(res[0].Text))
	}

	pprof.StopCPUProfile()
	t.Logf("CPU profile written to %s", profPath)
	t.Log("Analyse with:  go tool pprof -http=:8080", profPath)
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}
