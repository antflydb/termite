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
	"testing"

	"github.com/antflydb/antfly/pkg/libaf/embeddings"
	"github.com/antflydb/termite/pkg/termite/lib/backends"
	embeddingsLib "github.com/antflydb/termite/pkg/termite/lib/embeddings"
	_ "github.com/gomlx/gomlx/backends/simplego/highway"
	"go.uber.org/zap"
)

// BenchmarkBackendComparison runs a proper benchmark comparing embedding
// generation across all available backends.
func BenchmarkBackendComparison(b *testing.B) {
	modelPath := ensureHuggingFaceModel(b, "Snowflake/snowflake-arctic-embed-l-v2.0", "Snowflake/snowflake-arctic-embed-l-v2.0", ModelTypeEmbedder)

	texts := []string{
		"This is a test sentence for embedding generation.",
		"Machine learning models can generate dense vector representations.",
		"Semantic search uses embeddings to find similar documents.",
		"The quick brown fox jumps over the lazy dog.",
		"Artificial intelligence is transforming many industries.",
	}

	ctx := context.Background()
	logger := zap.NewNop()

	for _, backend := range backends.ListAvailable() {
		backendType := backend.Type()
		b.Run(string(backendType), func(b *testing.B) {
			sessionManager := backends.NewSessionManager()
			sessionManager.SetPriority([]backends.BackendSpec{
				{Backend: backendType, Device: backends.DeviceAuto},
			})

			cfg := embeddingsLib.PooledEmbedderConfig{
				ModelPath: modelPath,
				PoolSize:  1,
				Normalize: true,
				Logger:    logger,
			}

			embedder, _, err := embeddingsLib.NewPooledEmbedder(cfg, sessionManager)
			if err != nil {
				b.Skipf("Cannot create embedder: %v", err)
			}
			defer embedder.Close()

			// Warmup
			_, _ = embeddings.EmbedText(ctx, embedder, texts[:1])

			b.ResetTimer()
			for b.Loop() {
				_, err = embeddings.EmbedText(ctx, embedder, texts)
				if err != nil {
					b.Fatalf("Embed failed: %v", err)
				}
			}

			docsPerSec := float64(b.N*len(texts)) / b.Elapsed().Seconds()
			b.ReportMetric(docsPerSec, "docs/sec")
		})
	}
}
