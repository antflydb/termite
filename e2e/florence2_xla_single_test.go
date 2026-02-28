package e2e

import (
	"context"
	"image"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/antflydb/termite/pkg/termite/lib/backends"
	_ "github.com/gomlx/gomlx/backends/simplego/highway"
)

// TestFlorence2XLASingleStep tests whether a single XLA compilation + forward pass completes.
func TestFlorence2XLASingleStep(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping in short mode")
	}

	modelPath := ensureHuggingFaceModel(t, florence2ModelName, florence2ModelRepo, ModelTypeReader)
	img := loadFlorence2TestImage(t)

	// Try XLA first, fall back to Go
	backendType := backends.BackendXLA
	reader, err := createReaderWithBackend(t, modelPath, backendType)
	if err != nil {
		backendType = backends.BackendGo
		t.Logf("XLA not available (%v), using Go backend", err)
		reader, err = createReaderWithBackend(t, modelPath, backendType)
		require.NoError(t, err)
	}
	defer reader.Close()

	t.Logf("Using backend: %s", backendType)

	prompt := "What is the text in the image?"
	ctx := context.Background()

	start := time.Now()
	t.Log("Starting single read (maxTokens=5)...")
	res, err := reader.Read(ctx, []image.Image{img}, prompt, 5)
	elapsed := time.Since(start)

	if err != nil {
		t.Fatalf("Read failed after %v: %v", elapsed, err)
	}

	t.Logf("Completed in %v, output: %q", elapsed, res[0].Text)
}
