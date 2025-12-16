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
	"context"
	"net/http"
	"os/signal"
	"syscall"
	"time"

	"github.com/antflydb/antfly-go/libaf/healthserver"
	"github.com/antflydb/antfly-go/libaf/logging"
	"github.com/antflydb/termite/pkg/termite"
	"github.com/spf13/cobra"
	"github.com/spf13/viper"
)

var healthPort int

var runCmd = &cobra.Command{
	Use:   "run",
	Short: "Run the termite server",
	Long:  `Start the termite server for ML operations (embeddings, chunking, reranking).`,
	RunE:  runServer,
}

func init() {
	rootCmd.AddCommand(runCmd)

	// Run command flags
	runCmd.Flags().IntVar(&healthPort, "health-port", 4200, "health/metrics server port")
	mustBindPFlag("health_port", runCmd.Flags().Lookup("health-port"))
}

func runServer(cmd *cobra.Command, args []string) error {
	ctx, cancel := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer cancel()

	// Create logger from config
	logger := logging.NewLogger(&logging.Config{
		Level: logging.Level(viper.GetString("log.level")),
		Style: logging.Style(viper.GetString("log.style")),
	})
	defer func() {
		_ = logger.Sync()
	}()

	logger.Info("Running as termite")

	// Build termite config from viper/env
	cfg := termite.Config{
		ApiUrl:          viper.GetString("api_url"),
		ModelsDir:       modelsDir, // Set from --models-dir flag (defaults to ~/.termite/models)
		Gpu:             viper.GetString("gpu"),
		KeepAlive:       viper.GetString("keep_alive"),
		MaxLoadedModels: viper.GetInt("max_loaded_models"),
		MaxMemoryMb:     viper.GetInt("max_memory_mb"),
		Preload:         viper.GetStringSlice("preload"),
	}

	// Parse model_strategies from config (map[string]string -> map[string]ConfigModelStrategies)
	if rawStrategies := viper.GetStringMapString("model_strategies"); len(rawStrategies) > 0 {
		cfg.ModelStrategies = make(map[string]termite.ConfigModelStrategies, len(rawStrategies))
		for model, strategy := range rawStrategies {
			cfg.ModelStrategies[model] = termite.ConfigModelStrategies(strategy)
		}
	}

	// Start health server with ready checker that queries termite's /readyz
	termiteURL := cfg.ApiUrl
	if termiteURL == "" {
		termiteURL = "http://localhost:11433"
	}
	readyChecker := makeTermiteReadyChecker(termiteURL)
	healthserver.Start(logger, viper.GetInt("health_port"), readyChecker)

	termite.RunAsTermite(ctx, logger, cfg)
	return nil
}

// makeTermiteReadyChecker returns a function that checks termite's readiness
// by querying its /readyz endpoint.
func makeTermiteReadyChecker(termiteURL string) func() bool {
	client := &http.Client{
		Timeout: 2 * time.Second,
	}
	return func() bool {
		resp, err := client.Get(termiteURL + "/readyz")
		if err != nil {
			return false
		}
		defer func() { _ = resp.Body.Close() }()
		return resp.StatusCode == http.StatusOK
	}
}
