# VAD-Based Audio Chunker (Silero VAD)

## Context

The chunking API (`POST /api/chunk`) supports audio input, but only with fixed-duration windowing (`AudioChunker` splits into 30s segments regardless of content). This means silence gets included in chunks, words get split at arbitrary boundaries, and downstream transcription receives poorly-segmented audio.

Adding Silero VAD (Voice Activity Detection) as an audio chunker will split audio on speech/silence boundaries instead of fixed windows, producing cleaner segments for transcription and indexing.

## Approach

Keep it simple: add a `VADAudioChunker` alongside the existing `AudioChunker`, wire VAD model discovery into `termite.go` startup, and route to it from the API handler when `config.model` matches the VAD model name. No new registries, no interface changes, no OpenAPI schema changes.

Silero VAD is a ~2MB stateful LSTM ONNX model. Input: 512-sample frames at 16kHz. Output: per-frame speech probability. We run it frame-by-frame, merge probabilities into speech segments, then extract each segment as a WAV chunk (identical output format to the existing fixed chunker).

## Files to Change

### New Files

1. **`pkg/termite/lib/audio/resample.go`** — Public `Resample(samples []float32, fromRate, toRate int) []float32` function. Extracted from the private `AudioProcessor.resample` method at `lib/pipelines/audio.go:238`.

2. **`pkg/termite/lib/audio/resample_test.go`** — Tests: identity (same rate), upsampling, downsampling.

3. **`pkg/termite/lib/chunking/vad_audio_chunker.go`** — Core implementation:
   - `VADConfig` struct: `Threshold` (default 0.5), `MinSpeechDurationMs` (250), `MinSilenceDurationMs` (300), `SpeechPadMs` (30), `MaxSegmentDurationMs` (30000)
   - `VADAudioChunker` struct wrapping a `backends.Session`
   - `ChunkAudio(ctx, data, opts)` / `ChunkMP3(ctx, data, opts)` — parse audio, delegate to `ChunkPCM`
   - `ChunkPCM(ctx, samples, format, opts)` — main logic:
     1. Resample to 16kHz via `audio.Resample()`
     2. Run VAD frame-by-frame (512 samples per frame, carry LSTM h/c state)
     3. Call `MergeVADFrames()` to convert probabilities → `[]SpeechSegment`
     4. Map segment boundaries back to original sample rate
     5. Extract each segment from original samples, encode as WAV via `audio.EncodeWAV()`
     6. Return `[]chunking.Chunk` with timing metadata
   - `runVAD(samples16k []float32) ([]float32, error)` — frame-by-frame ONNX inference with state carry-forward. Inputs: `input [1,512]`, `h0 [2,1,64]`, `c0 [2,1,64]`, `sr int64(16000)`. Outputs: probability `[1,1]`, `hn`, `cn`.
   - `MergeVADFrames(probs []float32, frameSizeSamples, sampleRate int, config VADConfig) []SpeechSegment` — exported for unit testing. Handles thresholding, min duration filtering, silence gap merging, padding, max segment splitting.
   - If `opts.Threshold > 0`, override `VADConfig.Threshold` (reuses existing API field)

4. **`pkg/termite/lib/chunking/vad_audio_chunker_test.go`** — Unit tests for `MergeVADFrames` with synthetic probability arrays. No ONNX model needed. Test cases: basic merging, min speech duration filtering, min silence merging, padding, max segment splitting, all-silence, empty input, sample rate mapping.

### Modified Files

5. **`pkg/termite/lib/pipelines/audio.go`** — Change `AudioProcessor.resample` to delegate to `audio.Resample()`.

6. **`pkg/termite/termite.go`** — Add `vadChunker *mediachunking.VADAudioChunker` and `vadChunkerName string` fields to `TermiteNode`. During startup (after session manager is created), scan `models/chunkers/` for models with `"audio"` capability in their manifest (reuses existing `CapabilityAudio`). If found, create an ONNX session and instantiate `VADAudioChunker`. Add deferred cleanup.

7. **`pkg/termite/api.go`** — In `handleApiChunk`, at the two media dispatch points (~lines 614 and 645), check if `ln.vadChunker != nil && internalConfig.Model == ln.vadChunkerName`. If so, route to the VAD chunker (selecting `ChunkAudio` vs `ChunkMP3` based on MIME type) instead of `ln.mediaChunker.ChunkMedia()`.

## Key Design Decisions

- **No new model type or capability.** VAD models live in `models/chunkers/` with `ModelTypeChunker` and `capability: ["audio"]`, reusing the existing `CapabilityAudio` constant. Combined with the `chunker` model type, `"audio"` unambiguously means "audio chunker" (just as `"audio"` + `embedder` type means CLAP audio embedder).
- **No interface changes.** `MediaChunker` interface stays the same. Routing happens in the API handler.
- **No OpenAPI changes.** Reuse existing `config.model` for model selection and `config.threshold` for VAD probability threshold.
- **No registry for VAD.** Direct loading in `termite.go`. The Silero VAD model is tiny and there's typically only one. A full registry can be added later if needed.
- **Original sample rate preservation.** VAD runs on 16kHz resampled audio, but chunk extraction uses original samples to preserve audio quality.

## Verification

1. **Unit tests:** `go test ./pkg/termite/lib/audio/... ./pkg/termite/lib/chunking/...` — tests resampling and segment merging logic without models.
2. **Build:** `GOEXPERIMENT=simd go build ./...` from termite root.
3. **E2E (requires Silero VAD model):** Place model at `models/chunkers/silero/silero-vad/model.onnx` with a manifest containing `capabilities: ["audio"]`. Start termite, send a chunk request with audio + `model: "silero/silero-vad"`, verify returned chunks correspond to speech regions (not fixed 30s windows) and have correct timing metadata.
