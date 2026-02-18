# Plan: Add Pix2Struct, Surya OCR, and PaddleOCR Support to Termite

## Context

Termite's reader pipeline currently supports 4 Vision2Seq model families (TrOCR, Donut, Florence-2, Nougat) through a single `Vision2SeqPipeline` in `lib/pipelines/`. The `reading` package wraps pipelines with the `Reader` interface and pooling. This plan adds:

1. **Pix2Struct** (4 variants) -- fits existing Vision2Seq pipeline
2. **Surya OCR** (4-stage: detection, recognition, layout, reading order) -- new `MultiStageOCRPipeline`
3. **PaddleOCR PP-OCRv4** (2-stage: detection + recognition) -- shares multi-stage pipeline

Following the existing architecture: **pipelines** (`lib/pipelines/`) handle model loading and inference coordination, **readers** (`lib/reading/`) wrap pipelines with the `Reader` interface and pooling.

### Reuse from existing pipelines

The multi-stage pipeline maximally reuses existing infrastructure:

| Existing component | Reused for | Notes |
|---|---|---|
| `ImageProcessor` (`pipelines/image.go`) | Detection/layout input preprocessing | Same resize + normalize + NCHW conversion |
| `Vision2SeqPipeline` (`pipelines/vision2seq.go`) | **Surya recognition** | Surya rec is modified Donut -- loads directly via `LoadVision2SeqPipeline()` |
| `EncoderDecoderPipeline` (`pipelines/encoder_decoder.go`) | Surya recognition generation | Autoregressive decoding via `GenerateFromEncoderOutput()` |
| `sessionManager.GetSessionFactoryForModel()` | All new ONNX sessions | Same backend selection (ONNX > XLA > Go) |
| `tokenizers.LoadTokenizer()` | Surya recognition tokenizer | Standard HF tokenizer loading |
| `cropImage()` (`pipelines/image.go`) | Cropping detected regions | Existing function, extend with bbox-based wrapper |

The `MultiStageOCRPipeline` embeds a `*Vision2SeqPipeline` for Surya recognition (encoder-decoder) or a `*CTCRecognizer` for PaddleOCR (CTC decoding). Only detection post-processing, CTC decoding, and pipeline coordination are new code.

### Relationship to Florence-2 pipeline (`pipelines/florence2.go`)

Florence-2 is the closest architectural precedent for multi-session model coordination:

- **`florence2Model`** manages 4 ONNX sessions (`visionEncoderSession`, `embedTokensSession`, `encoderModelSession`, `decoderSession`) in a single `backends.Model`. `MultiStageOCRPipeline` similarly coordinates detection, recognition, layout, and order sessions.
- **`LoadFlorence2Model()`** (florence2.go:85-155) demonstrates the pattern for loading multiple ONNX files from one model directory via `factory.CreateSession()`, with cascading cleanup on error. `LoadMultiStageOCRPipeline` follows the same pattern.
- **`florence2Model.Close()`** (florence2.go:550-585) shows proper multi-session teardown. `MultiStageOCRPipeline.Close()` follows suit.
- **`Florence2Pipeline`** extends `EncoderDecoderPipeline` + `ImageProcessor` -- the same composition pattern used by `Vision2SeqRecognizer` wrapping `Vision2SeqPipeline`.

**Key distinction**: Florence-2 implements `backends.Model` (the `Forward()` method) because all 4 sessions form one encoder-decoder pipeline for autoregressive generation. `MultiStageOCRPipeline` does **not** implement `backends.Model` -- it orchestrates *independent* models (detection and layout are single-pass inference, recognition may be Vision2Seq or CTC). It's a higher-level coordinator that *contains* a `Vision2SeqPipeline` (for Surya) rather than being one.

**Licensing note**: Surya model weights use a restrictive license (free for research/personal/startups <$2M revenue, code is GPL-3.0). PaddleOCR and Pix2Struct are Apache-2.0.

---

## Phase 1: Pix2Struct (fits existing Vision2Seq pipeline)

Pix2Struct is an encoder-decoder model. Optimum's `ORTModelForVision2Seq` supports it, producing encoder/decoder ONNX files identical to TrOCR/Donut.

### `pkg/termite/lib/reading/reader.go` -- modify

- Add `ModelTypePix2Struct ModelType = "pix2struct"` constant
- Add `strings.Contains(pathLower, "pix2struct")` to `detectModelType()`
- Add `case ModelTypePix2Struct:` in `parseOutput()` -- plain text trimming (like TrOCR, Pix2Struct outputs direct answers)

### `pkg/termite/lib/reading/pix2struct.go` -- new

Prompt helpers:
- `Pix2StructDocVQAPrompt(question string) string`
- `Pix2StructChartQAPrompt(question string) string`
- `Pix2StructInfographicsPrompt(question string) string`

### `scripts/exporters/pix2struct.py` -- new

Register as `@register_exporter("reader", "pix2struct")`. Uses `ORTModelForVision2Seq.from_pretrained(model_id, export=True)`. Creates `termite_metadata.json` with `model_type: "pix2struct"`.

### `scripts/exporters/__init__.py` -- modify

Add `from . import pix2struct`.

### `scripts/exporters/reader.py` -- modify

Add `"pix2struct"` pattern to `READER_MODEL_PATTERNS` for fallback detection.

---

## Phase 2: Multi-stage OCR pipeline (shared infra for Surya + PaddleOCR)

### New pipeline: `pkg/termite/lib/pipelines/multistage_ocr.go`

This is the core addition. A new pipeline type that coordinates multiple models:

```go
// MultiStageOCRConfig configures a multi-stage OCR pipeline.
type MultiStageOCRConfig struct {
    DetectionModelPath   string
    RecognitionModelPath string
    LayoutModelPath      string // optional
    OrderModelPath       string // optional
    DetConfig            *DetectionConfig
    RecConfig            *RecognitionConfig
}

// DetectionConfig configures the detection stage.
type DetectionConfig struct {
    InputWidth       int
    InputHeight      int
    Threshold        float32
    MinBoxArea       int
    PostProcessor    DetectionPostProcessor // interface for DB vs heatmap
}

// RecognitionConfig configures the recognition stage.
type RecognitionConfig struct {
    InputHeight int
    InputWidth  int
    UseVision2Seq bool   // true for Surya (encoder-decoder), false for PaddleOCR (CTC)
    CharDictPath  string // for CTC-based recognizers
}

// TextRegion represents a detected text region.
type TextRegion struct {
    BBox       [4]float64
    Polygon    [][2]float64
    Confidence float64
}

// RecognizedRegion is a TextRegion with recognized text.
type RecognizedRegion struct {
    TextRegion
    Text       string
    Confidence float64
}

// LayoutRegion is a TextRegion with a label.
type LayoutRegion struct {
    TextRegion
    Label    string // "text", "title", "table", "figure", "caption", etc.
    OrderIdx int
}

// MultiStageOCRResult holds the pipeline output.
type MultiStageOCRResult struct {
    Regions  []RecognizedRegion
    Layout   []LayoutRegion // nil if no layout model
    FullText string
}

// Recognizer abstracts over Vision2Seq (Surya) and CTC (PaddleOCR) recognition.
type Recognizer interface {
    RecognizeImage(ctx context.Context, img image.Image) (string, float64, error)
    Close() error
}

// Vision2SeqRecognizer wraps the existing Vision2SeqPipeline for recognition.
// This reuses all existing encoder-decoder infrastructure.
type Vision2SeqRecognizer struct {
    pipeline *Vision2SeqPipeline  // reuses existing pipeline directly
}

// CTCRecognizer uses a single ONNX session + character dictionary.
type CTCRecognizer struct {
    session        backends.Session
    charDict       []string
    imageProcessor *ImageProcessor
}

// MultiStageOCRPipeline coordinates detection, recognition, and optional
// layout/order models for multi-stage OCR.
type MultiStageOCRPipeline struct {
    detector       backends.Session
    recognizer     Recognizer              // Vision2SeqRecognizer or CTCRecognizer
    layout         backends.Session        // optional (Surya)
    order          backends.Session        // optional (Surya)
    detProcessor   DetectionPostProcessor
    detImgProc     *ImageProcessor         // reuses existing ImageProcessor
    config         *MultiStageOCRConfig
}

// Run processes an image through the full pipeline.
func (p *MultiStageOCRPipeline) Run(ctx context.Context, img image.Image) (*MultiStageOCRResult, error)
func (p *MultiStageOCRPipeline) Close() error
```

Pipeline `Run()` flow:
1. Preprocess image → run detection session → post-process to `[]TextRegion`
2. If layout model present: run layout → get `[]LayoutRegion`
3. If order model present: run order → reorder regions
4. Otherwise: sort regions top-to-bottom, left-to-right
5. Crop each region from original image
6. Run recognition on each crop (batch if possible)
7. Assemble `MultiStageOCRResult`

### Detection post-processing interface

```go
// DetectionPostProcessor converts raw model output to text regions.
type DetectionPostProcessor interface {
    Process(output []float32, width, height int, originalBounds image.Rectangle) []TextRegion
}
```

Two implementations:
- `HeatmapPostProcessor` (Surya) -- threshold → connected components → bboxes
- `DBPostProcessor` (PaddleOCR) -- threshold → contours → unclip → bboxes

### New file: `pkg/termite/lib/pipelines/connected_components.go`

Pure Go connected component labeling for heatmap-based detection:
- Union-find on thresholded binary map
- Extract bounding rectangles from labeled regions
- Filter by minimum area

### New file: `pkg/termite/lib/pipelines/db_postprocess.go`

Differentiable Binarization post-processing for PaddleOCR's DBNet:
- Threshold probability map
- Border-following contour detection
- Score filtering (mean probability within contour)
- Simplified unclip (expand bbox by configurable ratio)
- Convert to bounding boxes

### New file: `pkg/termite/lib/pipelines/ctc_decode.go`

CTC greedy decoder for PaddleOCR's SVTR recognition:
- Argmax at each timestep
- Collapse consecutive duplicates
- Remove blank token (index 0)
- Map indices to characters via dictionary

### New file: `pkg/termite/lib/pipelines/crop.go`

Image region cropping utilities:
- `CropBBox(img image.Image, bbox [4]float64) image.Image`
- `ResizeKeepAspect(img image.Image, targetH, maxW int) image.Image` -- for recognition input
- `SortRegionsByReadingOrder(regions []TextRegion)` -- top-to-bottom, left-to-right fallback

### Loader: `pkg/termite/lib/pipelines/multistage_ocr_loader.go`

```go
// LoadMultiStageOCRPipeline loads a multi-stage OCR pipeline from a model directory.
// It reads termite_metadata.json to determine which stages to load.
func LoadMultiStageOCRPipeline(
    modelPath string,
    sessionManager *backends.SessionManager,
    modelBackends []string,
) (*MultiStageOCRPipeline, backends.BackendType, error)
```

Reads `termite_metadata.json` to determine:
- Which stage model files to load
- Which post-processor to use (heatmap vs DB)
- Whether recognition uses Vision2Seq or CTC
- Config values (thresholds, sizes)

### Reading layer: `pkg/termite/lib/reading/multistage_reader.go` -- new

Wraps `MultiStageOCRPipeline` with the `Reader` interface:

```go
type MultiStageReader struct {
    pipeline  *pipelines.MultiStageOCRPipeline
    sem       *semaphore.Weighted
    logger    *zap.Logger
    modelType ModelType
}

func NewMultiStageReader(cfg *MultiStageReaderConfig, sessionManager *backends.SessionManager, modelBackends []string) (*MultiStageReader, backends.BackendType, error)
func (r *MultiStageReader) Read(ctx context.Context, images []image.Image, prompt string, maxTokens int) ([]Result, error)
func (r *MultiStageReader) Close() error
```

Converts `MultiStageOCRResult` → `reading.Result` (including `Regions` field).

### Extend `reading.Result`

**`pkg/termite/lib/reading/reader.go`** -- add to `Result` struct:

```go
type Result struct {
    Text    string
    Fields  map[string]string
    Regions []RecognizedRegion // populated by multi-stage models
}
```

Add new `ModelType` constants:
```go
ModelTypeSurya     ModelType = "surya"
ModelTypePaddleOCR ModelType = "paddleocr"
ModelTypePix2Struct ModelType = "pix2struct"
```

Update `detectModelType()` for surya/paddleocr.

### Extend API schema

**`openapi.yaml`** -- add `TextRegion` schema and optional `regions` to `ReadResult`:

```yaml
TextRegion:
  type: object
  required: [text, bbox]
  properties:
    text: { type: string }
    bbox: { type: array, items: { type: number }, minItems: 4, maxItems: 4 }
    confidence: { type: number }
    label: { type: string }
```

Then `make generate` to update `api.gen.go`.

**`pkg/termite/api.go`** -- update read handler to map `Result.Regions` to API response.

### Extend reader registry

**`pkg/termite/reader_registry.go`** -- modify `loadModel()`:

```go
func (r *ReaderRegistry) loadModel(info *ReaderModelInfo) (reading.Reader, error) {
    metadata, _ := loadTermiteMetadata(info.Path)
    if metadata != nil && metadata.PipelineType == "multistage_ocr" {
        return reading.NewMultiStageReader(...)
    }
    // existing Vision2Seq path
    return reading.NewPooledReader(...)
}
```

### Extend `termite_metadata.json` format

Multi-stage models use:
```json
{
  "model_type": "surya",
  "pipeline_type": "multistage_ocr",
  "stages": {
    "detection": { "model_file": "detection.onnx", "post_processor": "heatmap" },
    "recognition": { "type": "vision2seq", "encoder_file": "rec_encoder.onnx", "decoder_file": "rec_decoder.onnx" },
    "layout": { "model_file": "layout.onnx" },
    "order": { "model_file": "order.onnx" }
  }
}
```

---

## Phase 3: Surya OCR (all 4 stages)

Models: `vikp/surya_det3` (modified EfficientViT), `vikp/surya_rec2` (modified Donut with GQA/MoE/UTF-16), `vikp/surya_layout3`, `vikp/surya_order`.

### Detection (uses `HeatmapPostProcessor`)

The Surya detection model outputs a segmentation heatmap. `HeatmapPostProcessor` in `connected_components.go` handles:
- Threshold heatmap → binary mask
- Connected component labeling → regions
- Bounding box extraction → scale to original image coords

No new files needed beyond the pipeline infrastructure from Phase 2.

### Recognition (reuses `Vision2SeqPipeline` -- zero new inference code)

Surya recognition is a modified Donut (encoder-decoder). It loads directly via the existing `LoadVision2SeqPipeline()` function and is wrapped in a `Vision2SeqRecognizer`. For each cropped text region, it calls `pipeline.Run(ctx, croppedImg)` -- reusing the full `ImageProcessor` → encoder → autoregressive decoder chain already implemented in `vision2seq.go` and `encoder_decoder.go`.

### Layout Analysis

Surya layout model is similar to detection (segmentation with class channels). Output: labeled regions (Caption, Footnote, Formula, ListItem, PageFooter, PageHeader, Picture, SectionHeader, Table, Text, Title) with bboxes.

Post-processing: argmax across class channels → connected components per class → labeled regions.

### Reading Order

Surya order model takes bbox positions as input and predicts sequential indices. Runs after layout analysis to reorder regions before recognition.

### Export script: `scripts/exporters/surya.py` -- new

```python
@register_exporter("reader", "surya")
class SuryaExporter(BaseExporter):
    """Exports Surya OCR models (detection, recognition, layout, order) to ONNX."""
```

- Downloads all 4 models from HuggingFace via `surya-ocr` package
- Exports each via `torch.onnx.export` (custom architectures, not Optimum)
- Detection: `pixel_values [B,3,H,W]` → heatmap `[B,1,H,W]`
- Recognition: encoder + decoder (with past) split export
- Layout: segmentation `pixel_values [B,3,H,W]` → class heatmaps `[B,C,H,W]`
- Order: bbox coordinates → order indices
- Saves tokenizer, preprocessor configs
- Creates `termite_metadata.json` with `pipeline_type: "multistage_ocr"` and all stage files

### `scripts/exporters/__init__.py` -- modify

Add `from . import surya`.

---

## Phase 4: PaddleOCR PP-OCRv4

PP-OCRv4 components: DBNet detection + SVTR recognition with CTC decoding.

### Detection (uses `DBPostProcessor`)

PaddleOCR DBNet outputs a probability map. `DBPostProcessor` in `db_postprocess.go` handles:
- Threshold → binary mask
- Contour extraction (border-following)
- Score filtering
- Bbox expansion (unclip ratio)

No new Go files needed beyond Phase 2 infrastructure.

### Recognition (uses `CTCRecognizer`)

PaddleOCR SVTR uses CTC decoding (not autoregressive). The pipeline creates a `CTCRecognizer` (a simple session + character dictionary) instead of `Vision2SeqPipeline`:

```go
// In multistage_ocr.go
type CTCRecognizer struct {
    session   backends.Session
    charDict  []string
    imgConfig *backends.ImageConfig
}

func (r *CTCRecognizer) Recognize(ctx context.Context, img image.Image) (string, float64, error)
func (r *CTCRecognizer) RecognizeBatch(ctx context.Context, imgs []image.Image) ([]string, []float64, error)
```

Uses `CTCDecode()` from `ctc_decode.go`.

### Export script: `scripts/exporters/paddleocr.py` -- new

```python
@register_exporter("reader", "paddleocr")
class PaddleOCRExporter(BaseExporter):
    """Exports PaddleOCR PP-OCRv4 models to ONNX."""
```

Two strategies:
1. Download pre-exported ONNX from community HuggingFace repos (e.g., `monkt/paddleocr-onnx`)
2. Download Paddle format models + convert via `paddle2onnx`

Also downloads character dictionary (`ppocr_keys_v1.txt`).

Creates `termite_metadata.json` with `pipeline_type: "multistage_ocr"`, `post_processor: "db"`, recognition `type: "ctc"`.

### `scripts/exporters/__init__.py` -- modify

Add `from . import paddleocr`.

---

## Phase 5: E2E Tests

### Pix2Struct tests -- add to `e2e/reader_test.go`

- `TestPix2StructModelDownload` -- download model, verify ONNX files exist
- `TestPix2StructDocVQA` -- load sample-page-1.png, ask VQA question, assert non-empty answer
- `TestPix2StructChartQA` -- chart image + question → answer (add testdata/chart.png)

### Surya tests -- new `e2e/surya_test.go`

- `TestSuryaModelExport` -- verify export produces expected ONNX files
- `TestSuryaDetection` -- detect regions on sample-page-1.png, assert valid bboxes
- `TestSuryaRecognition` -- full pipeline, assert text output
- `TestSuryaLayout` -- run layout, assert labeled regions (text, title, etc.)
- `TestSuryaFullPipeline` -- all 4 stages, validate text matches key phrases from sample-page-1.txt
- `TestSuryaRegionsInAPIResponse` -- verify API returns regions with bboxes

### PaddleOCR tests -- new `e2e/paddleocr_test.go`

- `TestPaddleOCRModelDownload` -- verify model setup
- `TestPaddleOCRDetection` -- detect regions, assert valid bboxes
- `TestPaddleOCRFullPipeline` -- full pipeline, validate text against expected phrases
- `TestPaddleOCRRegionsInAPIResponse` -- verify API returns regions

### Shared test helpers -- add to `e2e/reader_test.go`

- `assertValidRegions(t, regions, imgBounds)` -- validate bboxes within image bounds, text non-empty
- `loadTestImage(t, path) image.Image` -- shared image loading utility

---

## Implementation Order

1. **Phase 1: Pix2Struct** -- smallest scope, validates export approach
2. **Phase 2: Multi-stage framework** -- pipeline + types + API + registry
3. **Phase 4: PaddleOCR** -- simpler (2 stages, pre-exported ONNX, CTC decode), validates framework
4. **Phase 3: Surya** -- most complex (4 stages, custom torch.onnx.export, heatmap + layout + order)
5. **Phase 5: E2E tests** -- written incrementally with each phase

---

## New files summary

**Go (pipelines):**
- `pkg/termite/lib/pipelines/multistage_ocr.go` -- pipeline types, interfaces, core Run() logic
- `pkg/termite/lib/pipelines/multistage_ocr_loader.go` -- loading from model dir + metadata
- `pkg/termite/lib/pipelines/connected_components.go` -- heatmap post-processing (Surya)
- `pkg/termite/lib/pipelines/db_postprocess.go` -- DBNet post-processing (PaddleOCR)
- `pkg/termite/lib/pipelines/ctc_decode.go` -- CTC greedy decoder (PaddleOCR)
- `pkg/termite/lib/pipelines/crop.go` -- image cropping + reading order sorting

**Go (reading):**
- `pkg/termite/lib/reading/pix2struct.go` -- Pix2Struct prompt helpers
- `pkg/termite/lib/reading/multistage_reader.go` -- MultiStageReader (Reader wrapper)

**Python (exporters):**
- `scripts/exporters/pix2struct.py`
- `scripts/exporters/surya.py`
- `scripts/exporters/paddleocr.py`

**Tests:**
- `e2e/surya_test.go`
- `e2e/paddleocr_test.go`

**Modified files:**
- `pkg/termite/lib/reading/reader.go` -- ModelType constants, detectModelType, Result.Regions
- `pkg/termite/reader_registry.go` -- loadModel dispatch for multi-stage
- `pkg/termite/api.go` -- regions in read response
- `openapi.yaml` -- TextRegion schema + ReadResult.regions
- `scripts/exporters/__init__.py` -- register new exporters
- `scripts/exporters/reader.py` -- pix2struct pattern
- `e2e/reader_test.go` -- pix2struct tests + shared helpers

---

## Verification

1. `make generate` -- regenerate API types after OpenAPI changes
2. `GOEXPERIMENT=simd go build ./...` -- verify compilation
3. `GOEXPERIMENT=simd go test ./pkg/termite/lib/pipelines/... -run MultiStage` -- unit tests with mock sessions
4. `GOEXPERIMENT=simd go test ./pkg/termite/lib/pipelines/... -run CTC` -- CTC decoder tests
5. `GOEXPERIMENT=simd go test ./pkg/termite/lib/pipelines/... -run ConnectedComponent` -- CC labeling tests
6. `GOEXPERIMENT=simd go test ./pkg/termite/lib/pipelines/... -run DBPost` -- DB post-processing tests
7. `make e2e E2E_TEST=TestPix2Struct` -- Pix2Struct e2e
8. `make e2e E2E_TEST=TestPaddleOCR` -- PaddleOCR e2e
9. `make e2e E2E_TEST=TestSurya` -- Surya e2e
10. `make e2e` -- full test suite
