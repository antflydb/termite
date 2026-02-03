# 005: Speech-to-Text Transcription Pipeline

## Summary

Add a `Speech2SeqPipeline` for speech-to-text transcription using encoder-decoder models like Whisper. This follows the same architecture as `Vision2SeqPipeline` - audio preprocessing feeds into an encoder, which produces hidden states for autoregressive decoding.

## Motivation

- **Complete multimodal coverage**: Termite supports text, images, and soon audio
- **Popular use case**: Transcription is a common ML task with well-supported ONNX exports
- **Reuse existing architecture**: The encoder-decoder pattern from Vision2Seq translates directly

## Supported Models

**Primary target: Whisper**
- `openai/whisper-tiny` through `openai/whisper-large-v3`
- ONNX exports available via `optimum-cli export onnx`
- Well-documented architecture with good community support

**Future candidates:**
- Wav2Vec2 (CTC-based, simpler architecture)
- HuBERT
- Conformer-based models

## Architecture

### Pipeline Structure

```
                    ┌─────────────────────────────────────────┐
                    │         Speech2SeqPipeline              │
                    │                                         │
  Audio ──────────► │  AudioProcessor ──► speech2SeqModel    │
  (WAV/bytes)       │       │                   │            │
                    │       ▼                   ▼            │
                    │  Mel Features ──► Encoder ──► Decoder  │
                    │                              ▼         │
                    │                    EncoderDecoderPipeline
                    │                    (shared generation)  │
                    │                              ▼         │
                    │                         Tokenizer       │ ──► Text
                    └─────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility |
|-----------|---------------|
| `AudioProcessor` | Load audio, resample to 16kHz, compute mel spectrogram |
| `speech2SeqModel` | Wraps encoder + decoder ONNX sessions |
| `EncoderDecoderPipeline` | Shared autoregressive generation loop |
| `Tokenizer` | Decode token IDs to text |

### Configuration Types

```go
// Speech2SeqModelConfig holds parsed configuration for a Speech2Seq model.
type Speech2SeqModelConfig struct {
    ModelPath     string
    EncoderPath   string
    DecoderPath   string
    DecoderConfig *backends.DecoderConfig
    AudioConfig   *backends.AudioConfig  // NEW type

    // Architecture details for KV-cache
    NumLayers  int
    NumHeads   int
    HeadDim    int
    HiddenSize int
}

// AudioConfig holds configuration for audio preprocessing.
type AudioConfig struct {
    SampleRate    int       // Target sample rate (typically 16000)
    FeatureSize   int       // Mel spectrogram feature dimension (typically 80 or 128)
    NFft          int       // FFT window size (typically 400 for 25ms at 16kHz)
    HopLength     int       // Hop length (typically 160 for 10ms at 16kHz)
    ChunkLength   int       // Audio chunk length in seconds (typically 30 for Whisper)
    NMels         int       // Number of mel filter banks
    PaddingValue  float32   // Value to pad with (typically 0.0)
}
```

## Implementation Phases

### Phase 1: Core Types and Audio Preprocessing

**New files:**
- `pkg/termite/lib/backends/types.go` - Add `AudioConfig` type
- `pkg/termite/lib/pipelines/audio.go` - `AudioProcessor` implementation

**AudioProcessor responsibilities:**
1. Load audio from bytes (WAV format initially)
2. Resample to target sample rate (16kHz for Whisper)
3. Compute log-mel spectrogram
4. Pad/truncate to expected length

**Audio preprocessing approach:**

Option A: Pure Go implementation
- Use `go-audio/wav` for WAV loading
- Implement mel spectrogram in pure Go (slower but portable)
- No CGO dependencies

Option B: ONNX feature extractor (recommended for Whisper)
- Export `WhisperFeatureExtractor` as ONNX
- Single inference call converts audio → features
- Consistent with Python implementation

For initial implementation, use **Option A** with pure Go. Can add ONNX preprocessor later as optimization.

### Phase 2: Speech2Seq Model and Pipeline

**New files:**
- `pkg/termite/lib/pipelines/speech2seq.go` - Model, pipeline, and loader

**Model structure:**
```go
type speech2SeqModel struct {
    config         *Speech2SeqModelConfig
    encoderSession backends.Session
    decoderSession backends.Session
    backendType    backends.BackendType
}

// Forward runs encoder or decoder based on inputs.
// - If AudioFeatures set (EncoderOutput nil): runs audio encoder
// - If EncoderOutput set: runs decoder step
func (m *speech2SeqModel) Forward(ctx context.Context, inputs *backends.ModelInputs) (*backends.ModelOutput, error)
```

**Pipeline structure:**
```go
type Speech2SeqPipeline struct {
    *EncoderDecoderPipeline  // Shared generation
    AudioProcessor *AudioProcessor
}

func (p *Speech2SeqPipeline) Transcribe(ctx context.Context, audio []byte) (*Speech2SeqResult, error)
func (p *Speech2SeqPipeline) TranscribeWithOptions(ctx context.Context, audio []byte, opts *TranscribeOptions) (*Speech2SeqResult, error)
```

### Phase 3: Registry and API Integration

**Updated files:**
- `pkg/termite/lib/backends/types.go` - Add `AudioFeatures` to `ModelInputs`
- `pkg/termite/transcriber_registry.go` - New registry for speech2seq models
- `pkg/termite/api.go` - Add `/api/transcribe` endpoint handler
- `pkg/termite/openapi.yaml` - Add transcription endpoint spec

**Registry pattern:**
```go
type TranscriberRegistry struct {
    // Same pattern as EmbedderRegistry
    cache          *ttlcache.Cache[string, *loadedTranscriber]
    sessionManager *backends.SessionManager
    // ...
}

func (r *TranscriberRegistry) Transcribe(ctx context.Context, model string, audio []byte) (*Speech2SeqResult, error)
```

**API endpoint:**
```yaml
/api/transcribe:
  post:
    summary: Transcribe audio to text
    operationId: transcribeAudio
    requestBody:
      required: true
      content:
        multipart/form-data:
          schema:
            type: object
            required:
              - audio
            properties:
              model:
                type: string
                description: Model to use for transcription
              audio:
                type: string
                format: binary
                description: Audio file (WAV, MP3, etc.)
              language:
                type: string
                description: Language code (optional, for forced language)
    responses:
      '200':
        description: Transcription result
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/TranscriptionResult'
```

### Phase 4: Testing and E2E

**New files:**
- `e2e/transcriber_test.go` - E2E tests with real Whisper models

**Test approach:**
1. Download whisper-tiny ONNX export
2. Transcribe sample audio files
3. Verify output quality

## Model Directory Structure

```
models/speech2seq/whisper-tiny/
├── config.json              # Model configuration
├── generation_config.json   # Generation parameters
├── preprocessor_config.json # Audio preprocessing config
├── encoder_model.onnx       # Audio encoder
├── decoder_model_merged.onnx # Decoder with KV-cache
├── tokenizer.json           # Tokenizer
└── vocab.json               # Vocabulary (if separate)
```

## ModelInputs Extension

Add audio fields to `backends.ModelInputs`:

```go
type ModelInputs struct {
    // ... existing fields ...

    // Audio inputs (speech2seq models)
    AudioFeatures []float32 // Preprocessed mel spectrogram [batch, time, features]
    AudioBatch    int       // Batch size
    AudioTime     int       // Time steps (frames)
    AudioMels     int       // Feature dimension (mel bins)
}
```

## Backend Support

| Backend | Support | Notes |
|---------|---------|-------|
| ONNX Runtime | Full | Recommended - all Whisper ops supported |
| GoMLX (XLA) | Partial | May work, untested |
| GoMLX (Go) | Partial | Slower, may work |

**Recommendation:** ONNX Runtime for speech2seq models, same as vision2seq.

## Whisper-Specific Considerations

1. **Special tokens**: Whisper uses task-specific tokens (`<|transcribe|>`, `<|translate|>`, `<|en|>`, etc.)
2. **Timestamps**: Optional timestamp tokens for word-level timing
3. **Language detection**: First generated token can indicate detected language
4. **Long audio**: Whisper processes 30-second chunks; longer audio needs chunking

For initial implementation, focus on basic transcription without timestamps or translation.

## Open Questions

1. **Audio format support**: Start with WAV only, or add MP3/other formats?
   - Recommendation: WAV only initially, add others via `go-audio` ecosystem

2. **Streaming transcription**: Support for streaming audio input?
   - Recommendation: Not in initial implementation; Whisper doesn't support true streaming

3. **GoMLX support**: Worth implementing for speech2seq?
   - Recommendation: ONNX-only initially; add GoMLX if there's demand

## Files to Create/Modify

### New Files
| File | Description |
|------|-------------|
| `pkg/termite/lib/pipelines/audio.go` | AudioProcessor for mel spectrogram |
| `pkg/termite/lib/pipelines/speech2seq.go` | Speech2SeqPipeline and model |
| `pkg/termite/transcriber_registry.go` | Lazy-loading registry |
| `e2e/transcriber_test.go` | E2E tests |

### Modified Files
| File | Changes |
|------|---------|
| `pkg/termite/lib/backends/types.go` | Add `AudioConfig`, audio fields to `ModelInputs` |
| `pkg/termite/api.go` | Add transcription handler |
| `pkg/termite/openapi.yaml` | Add `/api/transcribe` endpoint |
| `pkg/termite/termite.go` | Add TranscriberRegistry to TermiteNode |

## Success Criteria

- [ ] `Speech2SeqPipeline` successfully transcribes audio using Whisper ONNX models
- [ ] `AudioProcessor` computes mel spectrograms matching Python implementation
- [ ] E2E tests pass with whisper-tiny
- [ ] API endpoint accepts audio files and returns transcriptions
- [ ] Registry supports lazy loading and model caching

## Timeline

No timeline estimates - work proceeds as capacity allows.

## References

- [Whisper ONNX export](https://huggingface.co/docs/optimum/onnx/usage_guides/whisper)
- [Whisper architecture](https://github.com/openai/whisper)
- [go-audio](https://github.com/go-audio/audio) - Audio format handling
- [Mel spectrogram implementation](https://pytorch.org/audio/stable/transforms.html#melspectrogram)
