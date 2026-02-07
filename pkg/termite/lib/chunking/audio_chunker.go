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

	"github.com/antflydb/antfly-go/libaf/chunking"
)

const (
	defaultWindowDurationMs  = 30000
	defaultOverlapDurationMs = 0
)

// AudioChunker segments WAV audio into fixed-duration windows.
// Preserves the original sample rate and bit depth.
type AudioChunker struct{}

// ChunkAudio parses a WAV file and segments it into fixed-duration windows.
// Each output chunk is a valid WAV file at the original sample rate/bit depth.
func (a *AudioChunker) ChunkAudio(ctx context.Context, data []byte, opts chunking.ChunkOptions) ([]chunking.Chunk, error) {
	samples, format, err := ParseWAV(data)
	if err != nil {
		return nil, fmt.Errorf("parsing WAV: %w", err)
	}

	if len(samples) == 0 {
		return nil, fmt.Errorf("WAV file contains no samples")
	}

	windowMs := opts.WindowDurationMs
	if windowMs <= 0 {
		windowMs = defaultWindowDurationMs
	}

	overlapMs := opts.OverlapDurationMs
	if overlapMs < 0 {
		overlapMs = defaultOverlapDurationMs
	}

	windowSamples := format.SampleRate * windowMs / 1000
	overlapSamples := format.SampleRate * overlapMs / 1000
	stepSamples := windowSamples - overlapSamples

	if stepSamples <= 0 {
		return nil, fmt.Errorf("overlap_duration_ms (%d) must be less than window_duration_ms (%d)", overlapMs, windowMs)
	}

	var chunks []chunking.Chunk
	totalSamples := len(samples)

	for offset := 0; offset < totalSamples; offset += stepSamples {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}

		end := offset + windowSamples
		if end > totalSamples {
			end = totalSamples
		}

		windowData := samples[offset:end]

		// Encode window as WAV with original format
		wavBytes, err := EncodeWAV(windowData, WAVFormat{
			SampleRate:    format.SampleRate,
			BitsPerSample: format.BitsPerSample,
			NumChannels:   1, // ParseWAV outputs mono
		})
		if err != nil {
			return nil, fmt.Errorf("encoding WAV chunk %d: %w", len(chunks), err)
		}

		startTimeMs := float32(offset) * 1000.0 / float32(format.SampleRate)
		endTimeMs := float32(end) * 1000.0 / float32(format.SampleRate)

		var c chunking.Chunk
		c.Id = uint32(len(chunks))
		c.MimeType = "audio/wav"
		c.FromBinaryContent(chunking.BinaryContent{
			Data:        wavBytes,
			StartTimeMs: startTimeMs,
			EndTimeMs:   endTimeMs,
		})

		chunks = append(chunks, c)

		// Enforce max chunks limit
		if opts.MaxChunks > 0 && len(chunks) >= opts.MaxChunks {
			break
		}
	}

	return chunks, nil
}
