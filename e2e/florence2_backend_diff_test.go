package e2e

import (
	"math"
	"sort"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/antflydb/termite/pkg/termite/lib/backends"
	"github.com/antflydb/termite/pkg/termite/lib/pipelines"
	_ "github.com/gomlx/gomlx/backends/simplego/highway"
)

// TestFlorence2BackendDiff compares intermediate tensors between Go and XLA
// backends stage-by-stage to find where numerical divergence occurs.
//
// Run with:
//
//	GOEXPERIMENT=simd go test -v -tags "xla,XLA" ./e2e/ -run TestFlorence2BackendDiff -timeout 15m
func TestFlorence2BackendDiff(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping in short mode")
	}

	modelPath := ensureHuggingFaceModel(t, florence2ModelName, florence2ModelRepo, ModelTypeReader)
	img := loadFlorence2TestImage(t)

	// Get session factories for both backends
	sm := backends.NewSessionManager()
	goFactory, err := sm.GetSessionFactory(backends.BackendGo)
	require.NoError(t, err, "getting Go session factory")
	xlaFactory, err := sm.GetSessionFactory(backends.BackendXLA)
	require.NoError(t, err, "getting XLA session factory")

	// Preprocess image using Florence-2's image config
	config, err := pipelines.LoadVision2SeqModelConfig(modelPath)
	require.NoError(t, err, "loading model config")

	imgProcessor := pipelines.NewImageProcessor(config.ImageConfig)
	pixels, err := imgProcessor.Process(img)
	require.NoError(t, err, "processing image")

	channels := config.ImageConfig.Channels
	height := config.ImageConfig.Height
	width := config.ImageConfig.Width
	t.Logf("Image preprocessed: %dx%dx%d, %d floats", channels, height, width, len(pixels))

	// Resolve ONNX file paths
	visionPath := pipelines.FindONNXFile(modelPath, []string{"vision_encoder.onnx"})
	embedPath := pipelines.FindONNXFile(modelPath, []string{"embed_tokens.onnx"})
	encoderPath := pipelines.FindONNXFile(modelPath, []string{"encoder_model.onnx"})
	decoderPath := pipelines.FindONNXFile(modelPath, []string{"decoder_model.onnx"})
	require.NotEmpty(t, visionPath, "vision_encoder.onnx not found")
	require.NotEmpty(t, embedPath, "embed_tokens.onnx not found")
	require.NotEmpty(t, encoderPath, "encoder_model.onnx not found")
	require.NotEmpty(t, decoderPath, "decoder_model.onnx not found")

	pixelInput := []backends.NamedTensor{{
		Name:  "pixel_values",
		Shape: []int64{1, int64(channels), int64(height), int64(width)},
		Data:  pixels,
	}}
	promptTokenIDs := []int64{2, 2264, 83, 5, 2788, 11, 5, 2274, 116}

	// === Stage 1: Vision Encoder ===
	t.Run("vision_encoder", func(t *testing.T) {
		goSession, err := goFactory.CreateSession(visionPath)
		require.NoError(t, err)
		defer goSession.Close()

		xlaSession, err := xlaFactory.CreateSession(visionPath)
		require.NoError(t, err)
		defer xlaSession.Close()

		t.Log("Running Go vision encoder...")
		goOut, err := goSession.Run(pixelInput)
		require.NoError(t, err)

		t.Log("Running XLA vision encoder...")
		xlaOut, err := xlaSession.Run(pixelInput)
		require.NoError(t, err)

		compareTensors(t, "vision_encoder", goOut[0], xlaOut[0])
	})

	// === Stage 2: Embed Tokens ===
	t.Run("embed_tokens", func(t *testing.T) {
		goSession, err := goFactory.CreateSession(embedPath)
		require.NoError(t, err)
		defer goSession.Close()

		xlaSession, err := xlaFactory.CreateSession(embedPath)
		require.NoError(t, err)
		defer xlaSession.Close()

		input := []backends.NamedTensor{{
			Name:  "input_ids",
			Shape: []int64{1, int64(len(promptTokenIDs))},
			Data:  promptTokenIDs,
		}}

		goOut, err := goSession.Run(input)
		require.NoError(t, err)

		xlaOut, err := xlaSession.Run(input)
		require.NoError(t, err)

		compareTensors(t, "embed_tokens", goOut[0], xlaOut[0])
	})

	// === Stage 3: Encoder Model ===
	// Uses XLA vision encoder + embed_tokens output as shared input.
	t.Run("encoder_model", func(t *testing.T) {
		encoderInput := buildEncoderInput(t, xlaFactory, visionPath, embedPath, pixelInput, promptTokenIDs)

		goEncoder, err := goFactory.CreateSession(encoderPath)
		require.NoError(t, err)
		defer goEncoder.Close()

		xlaEncoder, err := xlaFactory.CreateSession(encoderPath)
		require.NoError(t, err)
		defer xlaEncoder.Close()

		t.Log("Running Go encoder_model...")
		goOut, err := goEncoder.Run(encoderInput)
		require.NoError(t, err)

		t.Log("Running XLA encoder_model...")
		xlaOut, err := xlaEncoder.Run(encoderInput)
		require.NoError(t, err)

		compareTensors(t, "encoder_model", goOut[0], xlaOut[0])
	})

	// === Stage 4: First Decoder Step ===
	// Uses XLA full encoder pipeline output, compares first-step decoder logits.
	t.Run("decoder_first_step", func(t *testing.T) {
		// Get encoder hidden states from XLA
		encoderInput := buildEncoderInput(t, xlaFactory, visionPath, embedPath, pixelInput, promptTokenIDs)
		xlaEncoder, err := xlaFactory.CreateSession(encoderPath)
		require.NoError(t, err)
		defer xlaEncoder.Close()

		encOut, err := xlaEncoder.Run(encoderInput)
		require.NoError(t, err)

		encoderHiddenStates := encOut[0].Data.([]float32)
		encoderSeqLen := int(encOut[0].Shape[1])
		hiddenSize := int(encOut[0].Shape[2])

		// Embed BOS token for decoder input
		xlaEmbed, err := xlaFactory.CreateSession(embedPath)
		require.NoError(t, err)
		defer xlaEmbed.Close()

		bosEmbed, err := xlaEmbed.Run([]backends.NamedTensor{{
			Name: "input_ids", Shape: []int64{1, 1}, Data: []int64{2},
		}})
		require.NoError(t, err)
		decoderInputEmbeds := bosEmbed[0].Data.([]float32)

		// Create decoder sessions
		goDecoder, err := goFactory.CreateSession(decoderPath)
		require.NoError(t, err)
		defer goDecoder.Close()

		xlaDecoder, err := xlaFactory.CreateSession(decoderPath)
		require.NoError(t, err)
		defer xlaDecoder.Close()

		// Build first-step decoder inputs from session's input info
		decoderInputs := buildFirstStepDecoderInputs(t, goDecoder,
			decoderInputEmbeds, hiddenSize,
			encoderHiddenStates, encoderSeqLen,
			config.NumLayers, config.NumHeads, config.HeadDim)

		t.Log("Running Go decoder (first step)...")
		goOut, err := goDecoder.Run(decoderInputs)
		require.NoError(t, err, "Go decoder run")

		t.Log("Running XLA decoder (first step)...")
		xlaOut, err := xlaDecoder.Run(decoderInputs)
		require.NoError(t, err, "XLA decoder run")

		compareTensors(t, "decoder_logits", goOut[0], xlaOut[0])

		// Report top-5 token predictions
		goLogits := goOut[0].Data.([]float32)
		xlaLogits := xlaOut[0].Data.([]float32)
		vocabSize := int(goOut[0].Shape[len(goOut[0].Shape)-1])

		goLastLogits := goLogits[len(goLogits)-vocabSize:]
		xlaLastLogits := xlaLogits[len(xlaLogits)-vocabSize:]

		t.Log("Top-5 token predictions:")
		goTop5 := topK(goLastLogits, 5)
		xlaTop5 := topK(xlaLastLogits, 5)
		for i := range 5 {
			t.Logf("  Go  #%d: token=%d score=%.4f", i+1, goTop5[i].idx, goTop5[i].score)
			t.Logf("  XLA #%d: token=%d score=%.4f", i+1, xlaTop5[i].idx, xlaTop5[i].score)
		}
	})
}

// buildEncoderInput creates the encoder_model input tensors using XLA sessions
// for vision encoder and embed_tokens (to get shared reference inputs).
func buildEncoderInput(t *testing.T, factory backends.SessionFactory,
	visionPath, embedPath string,
	pixelInput []backends.NamedTensor, promptTokenIDs []int64,
) []backends.NamedTensor {
	t.Helper()

	vision, err := factory.CreateSession(visionPath)
	require.NoError(t, err)
	defer vision.Close()

	visionOut, err := vision.Run(pixelInput)
	require.NoError(t, err)

	imageFeatures := visionOut[0].Data.([]float32)
	imageSeqLen := int(visionOut[0].Shape[1])
	hiddenSize := int(visionOut[0].Shape[2])

	embed, err := factory.CreateSession(embedPath)
	require.NoError(t, err)
	defer embed.Close()

	embedOut, err := embed.Run([]backends.NamedTensor{{
		Name:  "input_ids",
		Shape: []int64{1, int64(len(promptTokenIDs))},
		Data:  promptTokenIDs,
	}})
	require.NoError(t, err)
	promptEmbeds := embedOut[0].Data.([]float32)
	promptLen := len(promptTokenIDs)

	totalSeqLen := imageSeqLen + promptLen
	inputsEmbeds := make([]float32, totalSeqLen*hiddenSize)
	copy(inputsEmbeds[:imageSeqLen*hiddenSize], imageFeatures)
	copy(inputsEmbeds[imageSeqLen*hiddenSize:], promptEmbeds)

	attentionMask := make([]int64, totalSeqLen)
	for i := range attentionMask {
		attentionMask[i] = 1
	}

	return []backends.NamedTensor{
		{Name: "inputs_embeds", Shape: []int64{1, int64(totalSeqLen), int64(hiddenSize)}, Data: inputsEmbeds},
		{Name: "attention_mask", Shape: []int64{1, int64(totalSeqLen)}, Data: attentionMask},
	}
}

// buildFirstStepDecoderInputs constructs inputs for the first decoder step
// by inspecting the session's InputInfo.
func buildFirstStepDecoderInputs(t *testing.T, session backends.Session,
	decoderEmbeds []float32, hiddenSize int,
	encoderHiddenStates []float32, encoderSeqLen int,
	numLayers, numHeads, headDim int,
) []backends.NamedTensor {
	t.Helper()

	var inputs []backends.NamedTensor
	for _, info := range session.InputInfo() {
		name := info.Name
		switch {
		case name == "inputs_embeds":
			inputs = append(inputs, backends.NamedTensor{
				Name: name, Shape: []int64{1, 1, int64(hiddenSize)}, Data: decoderEmbeds,
			})
		case name == "encoder_hidden_states":
			inputs = append(inputs, backends.NamedTensor{
				Name: name, Shape: []int64{1, int64(encoderSeqLen), int64(hiddenSize)}, Data: encoderHiddenStates,
			})
		case name == "encoder_attention_mask":
			mask := make([]int64, encoderSeqLen)
			for i := range mask {
				mask[i] = 1
			}
			inputs = append(inputs, backends.NamedTensor{
				Name: name, Shape: []int64{1, int64(encoderSeqLen)}, Data: mask,
			})
		case pipelines.IsPastKeyValueInput(name):
			if strings.Contains(name, ".encoder.") {
				// Encoder cross-attention KV: [batch, heads, encoder_seq_len, head_dim]
				data := make([]float32, numHeads*encoderSeqLen*headDim)
				inputs = append(inputs, backends.NamedTensor{
					Name: name, Shape: []int64{1, int64(numHeads), int64(encoderSeqLen), int64(headDim)}, Data: data,
				})
			} else {
				// Decoder self-attention KV: empty on first step
				inputs = append(inputs, backends.NamedTensor{
					Name: name, Shape: []int64{1, int64(numHeads), 0, int64(headDim)}, Data: []float32{},
				})
			}
		default:
			t.Logf("Skipping unknown decoder input: %s (shape=%v)", name, info.Shape)
		}
	}
	return inputs
}

type scoredToken struct {
	idx   int
	score float32
}

func topK(logits []float32, k int) []scoredToken {
	tokens := make([]scoredToken, len(logits))
	for i, s := range logits {
		tokens[i] = scoredToken{idx: i, score: s}
	}
	sort.Slice(tokens, func(i, j int) bool {
		return tokens[i].score > tokens[j].score
	})
	if k > len(tokens) {
		k = len(tokens)
	}
	return tokens[:k]
}

// compareTensors compares two named tensors element-by-element and reports statistics.
func compareTensors(t *testing.T, stage string, got, want backends.NamedTensor) {
	t.Helper()

	goData, ok := got.Data.([]float32)
	if !ok {
		t.Fatalf("%s: Go output is %T, not []float32", stage, got.Data)
	}
	xlaData, ok := want.Data.([]float32)
	if !ok {
		t.Fatalf("%s: XLA output is %T, not []float32", stage, want.Data)
	}

	if len(goData) != len(xlaData) {
		t.Fatalf("%s: length mismatch: Go=%d, XLA=%d", stage, len(goData), len(xlaData))
	}

	var maxAbsDiff, sumAbsDiff float64
	var maxRelDiff float64
	maxAbsIdx := 0
	aboveThreshold := 0
	nanCount := 0
	infCount := 0

	for i := range goData {
		if math.IsNaN(float64(goData[i])) {
			nanCount++
			continue
		}
		if math.IsInf(float64(goData[i]), 0) {
			infCount++
			continue
		}

		absDiff := math.Abs(float64(goData[i]) - float64(xlaData[i]))
		sumAbsDiff += absDiff
		if absDiff > maxAbsDiff {
			maxAbsDiff = absDiff
			maxAbsIdx = i
		}

		denom := math.Max(math.Abs(float64(xlaData[i])), 1e-7)
		relDiff := absDiff / denom
		if relDiff > maxRelDiff {
			maxRelDiff = relDiff
		}
		if relDiff > 0.01 {
			aboveThreshold++
		}
	}

	n := len(goData)
	meanAbsDiff := sumAbsDiff / float64(n)

	t.Logf("%s shape: Go=%v, XLA=%v, elements=%d", stage, got.Shape, want.Shape, n)
	if nanCount > 0 {
		t.Errorf("%s: Go output has %d NaN values!", stage, nanCount)
	}
	if infCount > 0 {
		t.Errorf("%s: Go output has %d Inf values!", stage, infCount)
	}
	t.Logf("%s max abs diff: %.6e at index %d (Go=%.6f, XLA=%.6f)",
		stage, maxAbsDiff, maxAbsIdx, goData[maxAbsIdx], xlaData[maxAbsIdx])
	t.Logf("%s mean abs diff: %.6e", stage, meanAbsDiff)
	t.Logf("%s max rel diff: %.6e", stage, maxRelDiff)
	t.Logf("%s elements >1%% relative error: %d / %d (%.2f%%)",
		stage, aboveThreshold, n, 100*float64(aboveThreshold)/float64(n))

	limit := min(5, n)
	t.Logf("%s first %d elements:", stage, limit)
	for i := range limit {
		t.Logf("  [%d] Go=%.8f  XLA=%.8f  diff=%.8f", i, goData[i], xlaData[i], goData[i]-xlaData[i])
	}

	t.Logf("%s elements around max diff (index %d):", stage, maxAbsIdx)
	start := max(0, maxAbsIdx-2)
	end := min(n, maxAbsIdx+3)
	for i := start; i < end; i++ {
		marker := "  "
		if i == maxAbsIdx {
			marker = ">>"
		}
		t.Logf("%s [%d] Go=%.8f  XLA=%.8f  diff=%.8f", marker, i, goData[i], xlaData[i], goData[i]-xlaData[i])
	}
}
