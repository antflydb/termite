// Copyright 2025 Antfly, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//	http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
package cmd

import (
	"fmt"

	"github.com/antflydb/termite/pkg/termite/lib/cli"
	"github.com/antflydb/termite/pkg/termite/lib/modelregistry"
	"github.com/spf13/cobra"
)

var variants []string

var pullCmd = &cobra.Command{
	Use:   "pull <model-name> [model-name...]",
	Short: "Pull ONNX model(s) from the registry",
	Long: `Download one or more ONNX models from the Antfly model registry.

Models are downloaded to the appropriate directory based on their type:
  - Embedders:     models/embedders/<model-name>/
  - Chunkers:      models/chunkers/<model-name>/
  - Rerankers:     models/rerankers/<model-name>/
  - Recognizers:   models/recognizers/<model-name>/
  - Questionators: models/questionators/<model-name>/

Variants:
  f32     - FP32 baseline (default, highest accuracy)
  f16     - FP16 half precision (~50% smaller)
  i8      - INT8 dynamic quantization (smallest, fastest CPU)
  i8-st   - INT8 static quantization (calibrated)
  i4      - INT4 quantization

Examples:
  # Pull default FP32 model
  termite pull bge-small-en-v1.5

  # Pull only INT8 variant (smaller download)
  termite pull --variants i8 bge-small-en-v1.5

  # Pull multiple variants
  termite pull --variants f16,i8 bge-small-en-v1.5

  # Pull multiple models with same variant
  termite pull --variants i8 bge-small-en-v1.5 mxbai-rerank-base-v1

  # Pull recognizer model
  termite pull bert-base-ner

  # Pull to a custom directory
  termite pull --models-dir /opt/antfly/models bge-small-en-v1.5

  # Pull directly from HuggingFace
  termite pull hf:onnx-community/embeddinggemma-300m-ONNX --type embedder

  # Pull recognizer model from HuggingFace
  termite pull hf:dslim/bert-base-NER --type recognizer

  # Pull GLiNER model from HuggingFace
  termite pull hf:onnx-community/gliner_small-v2.1 --type recognizer

  # Pull questionator model from HuggingFace
  termite pull hf:onnx-community/gemma-3-270m-it-ONNX --type questionator`,
	Args: cobra.MinimumNArgs(1),
	RunE: runPull,
}

func init() {
	rootCmd.AddCommand(pullCmd)

	// Pull command flags
	pullCmd.Flags().StringSliceVar(&variants, "variants", nil,
		"Variant IDs to download (f32,f16,i8,i8-st,i4). Defaults to f32 if not specified.")
	pullCmd.Flags().String("type", "",
		"Model type (embedder, chunker, reranker, recognizer, questionator) - required for hf: pulls")
	pullCmd.Flags().String("hf-token", "",
		"HuggingFace API token for gated models (or use HF_TOKEN env var)")
	pullCmd.Flags().String("variant", "",
		"ONNX variant for HuggingFace models (fp16, q4, q4f16, quantized)")
}

func runPull(cmd *cobra.Command, args []string) error {
	modelTypeStr, _ := cmd.Flags().GetString("type")
	hfToken, _ := cmd.Flags().GetString("hf-token")
	variant, _ := cmd.Flags().GetString("variant")

	for _, modelRef := range args {
		fmt.Printf("\n=== Pulling %s ===\n", modelRef)

		// Check for hf: prefix to route to HuggingFace
		if repoID, isHF := modelregistry.ParseHuggingFaceRef(modelRef); isHF {
			if err := cli.PullFromHuggingFace(repoID, cli.HuggingFaceOptions{
				ModelsDir: modelsDir,
				ModelType: modelTypeStr,
				HFToken:   hfToken,
				Variant:   variant,
			}); err != nil {
				return fmt.Errorf("failed to pull %s: %w", modelRef, err)
			}
			continue
		}

		// Standard registry pull
		if err := cli.PullFromRegistry(modelRef, cli.PullOptions{
			RegistryURL: registryURL,
			ModelsDir:   modelsDir,
			Variants:    variants,
		}); err != nil {
			return fmt.Errorf("failed to pull %s: %w", modelRef, err)
		}
	}

	return nil
}
