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

package chunking

import (
	"context"
	"fmt"
	"strings"

	"github.com/antflydb/antfly-go/libaf/chunking"
)

// MediaChunker splits binary media content into chunks.
type MediaChunker interface {
	ChunkMedia(ctx context.Context, data []byte, mimeType string, opts chunking.ChunkOptions) ([]chunking.Chunk, error)
}

// FixedMediaChunker dispatches media chunking to the appropriate handler
// based on MIME type. Algorithmic only (no ML models required).
type FixedMediaChunker struct {
	audio *AudioChunker
	gif   *GIFChunker
}

// NewFixedMediaChunker creates a new media chunker that dispatches by MIME type.
func NewFixedMediaChunker() *FixedMediaChunker {
	return &FixedMediaChunker{
		audio: &AudioChunker{},
		gif:   &GIFChunker{},
	}
}

// ChunkMedia dispatches to the appropriate chunker based on MIME type.
func (m *FixedMediaChunker) ChunkMedia(ctx context.Context, data []byte, mimeType string, opts chunking.ChunkOptions) ([]chunking.Chunk, error) {
	if len(data) == 0 {
		return nil, fmt.Errorf("empty media data")
	}

	mimeType = strings.ToLower(strings.TrimSpace(mimeType))

	switch {
	case mimeType == "audio/wav" || mimeType == "audio/x-wav" || mimeType == "audio/wave":
		return m.audio.ChunkAudio(ctx, data, opts)
	case mimeType == "image/gif":
		return m.gif.ChunkGIF(ctx, data, opts)
	default:
		return nil, fmt.Errorf("unsupported media type for chunking: %s", mimeType)
	}
}
