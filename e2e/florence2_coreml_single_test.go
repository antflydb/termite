package e2e

import (
	"context"
	"image"
	"testing"
	"time"

	"github.com/antflydb/termite/pkg/termite/lib/backends"
	_ "github.com/gomlx/gomlx/backends/simplego/highway"
)

// TestFlorence2CoreMLSingleStep tests whether CoreML compilation + forward pass completes.
func TestFlorence2CoreMLSingleStep(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping in short mode")
	}

	modelPath := ensureHuggingFaceModel(t, florence2ModelName, florence2ModelRepo, ModelTypeReader)
	img := loadFlorence2TestImage(t)

	backendType := backends.BackendCoreML
	reader, err := createReaderWithBackend(t, modelPath, backendType)
	if err != nil {
		t.Skipf("CoreML not available: %v", err)
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
