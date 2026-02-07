package chunking

import (
	"context"
	"testing"

	"github.com/antflydb/antfly-go/libaf/chunking"
)

func TestFixedMediaChunker_DispatchAudio(t *testing.T) {
	wavData := generateTestWAV(t, 16000, 16, 1000) // 1 second of 16kHz 16-bit audio
	chunker := NewFixedMediaChunker()

	chunks, err := chunker.ChunkMedia(context.Background(), wavData, "audio/wav", chunking.ChunkOptions{})
	if err != nil {
		t.Fatalf("ChunkMedia returned error: %v", err)
	}

	if len(chunks) == 0 {
		t.Fatal("expected at least one chunk, got 0")
	}

	for i, chunk := range chunks {
		if chunk.MimeType != "audio/wav" {
			t.Errorf("chunk %d: expected mime_type %q, got %q", i, "audio/wav", chunk.MimeType)
		}

		bc, err := chunk.AsBinaryContent()
		if err != nil {
			t.Fatalf("chunk %d: AsBinaryContent error: %v", i, err)
		}
		if len(bc.Data) == 0 {
			t.Errorf("chunk %d: expected non-empty data", i)
		}
	}
}

func TestFixedMediaChunker_DispatchGIF(t *testing.T) {
	gifData := generateTestGIF(t, 3, 16, 16)
	chunker := NewFixedMediaChunker()

	chunks, err := chunker.ChunkMedia(context.Background(), gifData, "image/gif", chunking.ChunkOptions{})
	if err != nil {
		t.Fatalf("ChunkMedia returned error: %v", err)
	}

	if len(chunks) != 3 {
		t.Fatalf("expected 3 chunks, got %d", len(chunks))
	}

	for i, chunk := range chunks {
		if chunk.MimeType != "image/png" {
			t.Errorf("chunk %d: expected mime_type %q, got %q", i, "image/png", chunk.MimeType)
		}
	}
}

func TestFixedMediaChunker_UnsupportedType(t *testing.T) {
	chunker := NewFixedMediaChunker()

	_, err := chunker.ChunkMedia(context.Background(), []byte("some data"), "video/mp4", chunking.ChunkOptions{})
	if err == nil {
		t.Fatal("expected error for unsupported type, got nil")
	}

	if got := err.Error(); !contains(got, "unsupported") {
		t.Errorf("expected error containing %q, got %q", "unsupported", got)
	}
}

func TestFixedMediaChunker_EmptyData(t *testing.T) {
	chunker := NewFixedMediaChunker()

	_, err := chunker.ChunkMedia(context.Background(), []byte{}, "audio/wav", chunking.ChunkOptions{})
	if err == nil {
		t.Fatal("expected error for empty data, got nil")
	}
}

// contains checks if s contains substr (avoids importing strings for a single call).
func contains(s, substr string) bool {
	return len(s) >= len(substr) && searchString(s, substr)
}

func searchString(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}
