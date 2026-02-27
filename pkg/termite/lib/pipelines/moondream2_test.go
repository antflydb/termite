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

package pipelines

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestIsMoondream2Model_ValidStructure(t *testing.T) {
	// Create temp directory with Moondream2 structure
	tmpDir := t.TempDir()
	moondreamDir := filepath.Join(tmpDir, "vikhyatk-moondream2")
	require.NoError(t, os.MkdirAll(moondreamDir, 0755))

	// Create the three required ONNX files
	for _, f := range []string{"vision_encoder.onnx", "projection.onnx", "decoder_model.onnx"} {
		require.NoError(t, os.WriteFile(filepath.Join(moondreamDir, f), []byte("dummy"), 0644))
	}

	assert.True(t, IsMoondream2Model(moondreamDir))
}

func TestIsMoondream2Model_MissingProjection(t *testing.T) {
	tmpDir := t.TempDir()
	moondreamDir := filepath.Join(tmpDir, "moondream2")
	require.NoError(t, os.MkdirAll(moondreamDir, 0755))

	// Missing projection.onnx
	require.NoError(t, os.WriteFile(filepath.Join(moondreamDir, "vision_encoder.onnx"), []byte("dummy"), 0644))
	require.NoError(t, os.WriteFile(filepath.Join(moondreamDir, "decoder_model.onnx"), []byte("dummy"), 0644))

	assert.False(t, IsMoondream2Model(moondreamDir))
}

func TestIsMoondream2Model_NotMoondreamName(t *testing.T) {
	// Even with the right files, if path doesn't contain "moondream", it should return false
	// This distinguishes from Florence-2 which also has vision_encoder.onnx
	tmpDir := t.TempDir()
	otherDir := filepath.Join(tmpDir, "some-other-model")
	require.NoError(t, os.MkdirAll(otherDir, 0755))

	for _, f := range []string{"vision_encoder.onnx", "projection.onnx", "decoder_model.onnx"} {
		require.NoError(t, os.WriteFile(filepath.Join(otherDir, f), []byte("dummy"), 0644))
	}

	assert.False(t, IsMoondream2Model(otherDir))
}

func TestIsMoondream2Model_Florence2Structure(t *testing.T) {
	// Florence-2 has different structure - should not be detected as Moondream
	tmpDir := t.TempDir()
	florenceDir := filepath.Join(tmpDir, "florence-2")
	require.NoError(t, os.MkdirAll(florenceDir, 0755))

	// Florence-2 files (no projection.onnx)
	for _, f := range []string{"vision_encoder.onnx", "embed_tokens.onnx", "encoder_model.onnx", "decoder_model_merged.onnx"} {
		require.NoError(t, os.WriteFile(filepath.Join(florenceDir, f), []byte("dummy"), 0644))
	}

	assert.False(t, IsMoondream2Model(florenceDir))
}

func TestLoadMoondream2ModelConfig_ValidConfig(t *testing.T) {
	tmpDir := t.TempDir()
	moondreamDir := filepath.Join(tmpDir, "moondream2")
	require.NoError(t, os.MkdirAll(moondreamDir, 0755))

	// Create ONNX files
	for _, f := range []string{"vision_encoder.onnx", "projection.onnx", "decoder_model.onnx"} {
		require.NoError(t, os.WriteFile(filepath.Join(moondreamDir, f), []byte("dummy"), 0644))
	}

	// Create config.json
	config := map[string]any{
		"model_type":          "moondream",
		"hidden_size":         2048,
		"num_hidden_layers":   24,
		"num_attention_heads": 32,
		"vocab_size":          51200,
		"vision_hidden_size":  1152,
		"image_size":          378,
		"bos_token_id":        50256,
		"eos_token_id":        50256,
		"max_length":          2048,
	}
	configBytes, _ := json.Marshal(config)
	require.NoError(t, os.WriteFile(filepath.Join(moondreamDir, "config.json"), configBytes, 0644))

	cfg, err := LoadMoondream2ModelConfig(moondreamDir)
	require.NoError(t, err)

	assert.Equal(t, 2048, cfg.HiddenSize)
	assert.Equal(t, 24, cfg.NumLayers)
	assert.Equal(t, 32, cfg.NumHeads)
	assert.Equal(t, 64, cfg.HeadDim) // 2048 / 32
	assert.Equal(t, 1152, cfg.VisionHidden)

	assert.NotNil(t, cfg.DecoderConfig)
	assert.Equal(t, 51200, cfg.DecoderConfig.VocabSize)
	assert.Equal(t, int32(50256), cfg.DecoderConfig.EOSTokenID)

	assert.NotNil(t, cfg.ImageConfig)
	assert.Equal(t, 378, cfg.ImageConfig.Width)
	assert.Equal(t, 378, cfg.ImageConfig.Height)
}

func TestLoadMoondream2ModelConfig_DefaultValues(t *testing.T) {
	tmpDir := t.TempDir()
	moondreamDir := filepath.Join(tmpDir, "moondream2")
	require.NoError(t, os.MkdirAll(moondreamDir, 0755))

	// Create ONNX files
	for _, f := range []string{"vision_encoder.onnx", "projection.onnx", "decoder_model.onnx"} {
		require.NoError(t, os.WriteFile(filepath.Join(moondreamDir, f), []byte("dummy"), 0644))
	}

	// Create minimal config.json
	config := map[string]any{
		"model_type": "moondream",
	}
	configBytes, _ := json.Marshal(config)
	require.NoError(t, os.WriteFile(filepath.Join(moondreamDir, "config.json"), configBytes, 0644))

	cfg, err := LoadMoondream2ModelConfig(moondreamDir)
	require.NoError(t, err)

	// Should use defaults
	assert.Equal(t, 2048, cfg.HiddenSize)
	assert.Equal(t, 24, cfg.NumLayers)
	assert.Equal(t, 32, cfg.NumHeads)
	assert.Equal(t, 378, cfg.ImageConfig.Width)
}

func TestLoadMoondream2ModelConfig_WithPreprocessorConfig(t *testing.T) {
	tmpDir := t.TempDir()
	moondreamDir := filepath.Join(tmpDir, "moondream2")
	require.NoError(t, os.MkdirAll(moondreamDir, 0755))

	// Create ONNX files
	for _, f := range []string{"vision_encoder.onnx", "projection.onnx", "decoder_model.onnx"} {
		require.NoError(t, os.WriteFile(filepath.Join(moondreamDir, f), []byte("dummy"), 0644))
	}

	// Create config.json
	config := map[string]any{"model_type": "moondream"}
	configBytes, _ := json.Marshal(config)
	require.NoError(t, os.WriteFile(filepath.Join(moondreamDir, "config.json"), configBytes, 0644))

	// Create preprocessor_config.json with custom values
	preproc := map[string]any{
		"size":       384,
		"image_mean": []float64{0.48, 0.48, 0.48},
		"image_std":  []float64{0.26, 0.26, 0.26},
	}
	preprocBytes, _ := json.Marshal(preproc)
	require.NoError(t, os.WriteFile(filepath.Join(moondreamDir, "preprocessor_config.json"), preprocBytes, 0644))

	cfg, err := LoadMoondream2ModelConfig(moondreamDir)
	require.NoError(t, err)

	// Should use preprocessor values
	assert.Equal(t, 384, cfg.ImageConfig.Width)
	assert.Equal(t, 384, cfg.ImageConfig.Height)
	assert.InDelta(t, 0.48, cfg.ImageConfig.Mean[0], 0.01)
	assert.InDelta(t, 0.26, cfg.ImageConfig.Std[0], 0.01)
}

func TestLoadMoondream2ModelConfig_MissingConfigJSON(t *testing.T) {
	tmpDir := t.TempDir()
	moondreamDir := filepath.Join(tmpDir, "moondream2")
	require.NoError(t, os.MkdirAll(moondreamDir, 0755))

	// Create ONNX files but no config.json
	for _, f := range []string{"vision_encoder.onnx", "projection.onnx", "decoder_model.onnx"} {
		require.NoError(t, os.WriteFile(filepath.Join(moondreamDir, f), []byte("dummy"), 0644))
	}

	_, err := LoadMoondream2ModelConfig(moondreamDir)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "config.json")
}

func TestGetIntFromMoondreamConfig(t *testing.T) {
	config := map[string]any{
		"present":    float64(42),
		"zero":       float64(0),
		"wrong_type": "not a number",
	}

	assert.Equal(t, 42, getIntFromMoondreamConfig(config, "present", 0))
	assert.Equal(t, 0, getIntFromMoondreamConfig(config, "zero", 99))
	assert.Equal(t, 99, getIntFromMoondreamConfig(config, "missing", 99))
	assert.Equal(t, 99, getIntFromMoondreamConfig(config, "wrong_type", 99))
}
