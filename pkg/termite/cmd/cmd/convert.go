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

package cmd

import (
	"fmt"

	"github.com/antflydb/termite/pkg/termite/lib/tabular"
	"github.com/antflydb/termite/pkg/termite/lib/tabular/convert"
	"github.com/spf13/cobra"
)

var convertCmd = &cobra.Command{
	Use:   "convert",
	Short: "Convert ML models to Termite's tabular IR format",
	Long: `Convert traditional ML models (XGBoost, LightGBM, CatBoost) into
Termite's tabular intermediate representation for fast Go-native inference.

Produces tabular_model.json and model_manifest.json in the output directory.

Examples:
  # Convert XGBoost model
  termite convert xgboost model.json -o ./my-model

  # Convert LightGBM model (auto-detects text vs JSON)
  termite convert lightgbm model.txt -o ./my-model

  # Convert CatBoost JSON model
  termite convert catboost model.json -o ./my-model --name my-catboost`,
}

var xgboostCmd = &cobra.Command{
	Use:   "xgboost <model-path>",
	Short: "Convert an XGBoost JSON model",
	Long: `Convert an XGBoost model saved as JSON (xgb.save_model("model.json")).

Supports both modern (nodes array) and legacy (nested children) XGBoost JSON formats.`,
	Args: cobra.ExactArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		return runConvert(args[0], "xgboost", cmd)
	},
}

var lightgbmCmd = &cobra.Command{
	Use:   "lightgbm <model-path>",
	Short: "Convert a LightGBM model (text or JSON)",
	Long: `Convert a LightGBM model in either text format (lgb.save_model("model.txt"))
or JSON format (lgb.dump_model()).

The format is auto-detected from file contents.`,
	Args: cobra.ExactArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		return runConvert(args[0], "lightgbm", cmd)
	},
}

var catboostCmd = &cobra.Command{
	Use:   "catboost <model-path>",
	Short: "Convert a CatBoost JSON model",
	Long: `Convert a CatBoost model saved as JSON (model.save_model("model.json", format="json")).

CatBoost uses oblivious decision trees which are expanded into standard binary trees.`,
	Args: cobra.ExactArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		return runConvert(args[0], "catboost", cmd)
	},
}

func init() {
	rootCmd.AddCommand(convertCmd)
	convertCmd.AddCommand(xgboostCmd)
	convertCmd.AddCommand(lightgbmCmd)
	convertCmd.AddCommand(catboostCmd)

	// Shared flags on parent (inherited by subcommands)
	convertCmd.PersistentFlags().StringP("output", "o", "", "Output directory (required)")
	convertCmd.PersistentFlags().String("name", "", "Model name override (defaults to filename stem)")
	convertCmd.MarkPersistentFlagRequired("output")
}

func runConvert(modelPath, framework string, cmd *cobra.Command) error {
	output, _ := cmd.Flags().GetString("output")
	name, _ := cmd.Flags().GetString("name")

	var (
		model *tabular.TabularModel
		err   error
	)

	switch framework {
	case "xgboost":
		model, err = convert.ParseXGBoost(modelPath, name)
	case "lightgbm":
		model, err = convert.ParseLightGBM(modelPath, name)
	case "catboost":
		model, err = convert.ParseCatBoost(modelPath, name)
	default:
		return fmt.Errorf("unsupported framework: %s", framework)
	}

	if err != nil {
		return fmt.Errorf("parsing %s model: %w", framework, err)
	}

	// Validate the produced IR
	if err := model.Validate(); err != nil {
		return fmt.Errorf("validation failed: %w", err)
	}

	if err := convert.SaveModel(model, output); err != nil {
		return err
	}

	convert.PrintSummary(model, output)
	return nil
}
