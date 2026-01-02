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

//go:generate go tool oapi-codegen --config=cfg.yaml ./openapi.yaml
package termite

import (
	"context"
	"net/http"
	"net/url"
	"path/filepath"
	"time"

	"github.com/antflydb/antfly-go/libaf/s3"
	"github.com/antflydb/antfly-go/libaf/scraping"
	"github.com/antflydb/termite/pkg/termite/lib/hugot"
	"go.uber.org/zap"
)

type TermiteNode struct {
	logger *zap.Logger

	client *http.Client

	// Embedder registry (lazy loading with TTL-based unloading)
	embedderRegistry *EmbedderRegistry

	cachedChunker         *CachedChunker
	rerankerRegistry      RerankerRegistryInterface
	generatorRegistry     *GeneratorRegistry
	nerRegistry           NERRegistryInterface
	seq2seqRegistry       *Seq2SeqRegistry
	contentSecurityConfig *scraping.ContentSecurityConfig
	s3Credentials         *s3.Credentials

	// Request queue for backpressure control
	requestQueue *RequestQueue

	// Caches for embeddings, reranking, and NER
	embeddingCache *EmbeddingCache
	rerankingCache *RerankingCache
	nerCache       *NERCache
}

// corsMiddleware adds permissive CORS headers for the Termite API
func corsMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS, PATCH")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Requested-With, Accept, Origin")
		w.Header().Set("Access-Control-Max-Age", "3600")

		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}

		next.ServeHTTP(w, r)
	})
}

// DefaultShutdownTimeout is the default time to wait for graceful shutdown
const DefaultShutdownTimeout = 30 * time.Second

// RunAsTermite implements a leader node that monitors and manages the cluster.
// If readyC is non-nil, it will be closed when the server is ready to accept requests.
func RunAsTermite(ctx context.Context, zl *zap.Logger, config Config, readyC chan struct{}) {
	zl = zl.Named("termite")
	zl.Info("Starting termite node", zap.Any("config", config))

	u, err := url.Parse(config.ApiUrl)
	if err != nil {
		zl.Fatal("Invalid API URL", zap.String("url", config.ApiUrl), zap.Error(err))
	}

	// Parse backend priority (supports "backend" or "backend:device" format)
	var backendPriority []hugot.BackendSpec
	if len(config.BackendPriority) > 0 {
		var err error
		backendPriority, err = hugot.ParseBackendPriority(config.BackendPriority)
		if err != nil {
			zl.Fatal("Invalid backend_priority configuration", zap.Error(err))
		}
		// Also set global priority for backward compatibility
		globalPriority := make([]hugot.BackendType, 0, len(backendPriority))
		for _, spec := range backendPriority {
			globalPriority = append(globalPriority, spec.Backend)
		}
		hugot.SetPriority(globalPriority)
		zl.Info("Backend priority configured", zap.Any("priority", config.BackendPriority))
	}

	// Log available backends
	availableBackends := hugot.ListAvailable()
	backendNames := make([]string, 0, len(availableBackends))
	for _, b := range availableBackends {
		backendNames = append(backendNames, b.Name())
	}
	zl.Info("Available inference backends", zap.Strings("backends", backendNames))

	// Detect and log GPU info, set metrics
	gpuInfo := hugot.GetGPUInfo()
	zl.Info("GPU detection complete",
		zap.Bool("available", gpuInfo.Available),
		zap.String("type", gpuInfo.Type),
		zap.String("device", gpuInfo.DeviceName))

	// Parse keep_alive duration
	// Default to 5 minutes like Ollama - lazy loading is the default behavior.
	// Set keep_alive to "0" to explicitly enable eager loading (all models loaded at startup).
	const defaultKeepAlive = 5 * time.Minute
	var keepAlive time.Duration
	if config.KeepAlive == "0" {
		// Explicit eager loading
		keepAlive = 0
		zl.Info("Eager loading mode (all models loaded at startup)")
	} else if config.KeepAlive != "" {
		keepAlive, err = time.ParseDuration(config.KeepAlive)
		if err != nil {
			zl.Fatal("Invalid keep_alive duration", zap.String("keep_alive", config.KeepAlive), zap.Error(err))
		}
		zl.Info("Lazy loading enabled",
			zap.Duration("keep_alive", keepAlive),
			zap.Int("max_loaded_models", config.MaxLoadedModels))
	} else {
		// Default to lazy loading with 5 minute keep_alive (Ollama-compatible)
		keepAlive = defaultKeepAlive
		zl.Info("Lazy loading enabled (default)",
			zap.Duration("keep_alive", keepAlive),
			zap.Int("max_loaded_models", config.MaxLoadedModels))
	}

	// Compute model subdirectory paths from models_dir
	var embedderModelsDir, chunkerModelsDir, rerankerModelsDir, generatorModelsDir string
	if config.ModelsDir != "" {
		embedderModelsDir = filepath.Join(config.ModelsDir, "embedders")
		chunkerModelsDir = filepath.Join(config.ModelsDir, "chunkers")
		rerankerModelsDir = filepath.Join(config.ModelsDir, "rerankers")
		generatorModelsDir = filepath.Join(config.ModelsDir, "generators")
	}

	// Create session manager for multi-backend support
	// SessionManager handles backend selection per-model and manages sessions.
	// IMPORTANT: ONNX Runtime backend allows only ONE session at a time.
	// SessionManager enforces this by sharing sessions within each backend type.
	var sessionManager *hugot.SessionManager
	hasModels := config.ModelsDir != ""

	if hasModels {
		sessionManager = hugot.NewSessionManager()
		defer func() { _ = sessionManager.Close() }()

		// Configure session manager with backend priority (includes device preferences)
		if len(backendPriority) > 0 {
			sessionManager.SetPriority(backendPriority)
		}

		defaultBackend := hugot.GetDefaultBackend()
		if defaultBackend != nil {
			zl.Info("Session manager initialized",
				zap.String("default_backend", defaultBackend.Name()))
		} else {
			zl.Warn("No inference backends available")
		}
	}

	// Initialize chunker with optional model directory support
	// If models_dir is set in config, Termite will discover and load chunker models
	// If not set, Termite falls back to semantic-only chunking
	cachedChunker, err := NewCachedChunker(chunkerModelsDir, sessionManager, config.PoolSize, zl.Named("chunker"))
	if err != nil {
		zl.Fatal("Failed to initialize chunker", zap.Error(err))
	}
	defer func() { _ = cachedChunker.Close() }()

	// Initialize embedder registry (lazy loading with TTL-based unloading)
	embedderRegistry, err := NewEmbedderRegistry(
		EmbedderConfig{
			ModelsDir:       embedderModelsDir,
			KeepAlive:       keepAlive,
			MaxLoadedModels: uint64(config.MaxLoadedModels),
			PoolSize:        config.PoolSize,
		},
		sessionManager,
		zl.Named("embedder"),
	)
	if err != nil {
		zl.Fatal("Failed to initialize embedder registry", zap.Error(err))
	}
	defer func() { _ = embedderRegistry.Close() }()

	// Apply per-model loading strategies
	// Models with "eager" strategy are pinned (never evicted)
	if len(config.ModelStrategies) > 0 {
		var eagerModels []string
		for modelName, strategy := range config.ModelStrategies {
			if strategy == ConfigModelStrategiesEager {
				eagerModels = append(eagerModels, modelName)
			}
		}
		if len(eagerModels) > 0 {
			zl.Info("Pinning eager models (will not be evicted)",
				zap.Strings("models", eagerModels))
			for _, modelName := range eagerModels {
				if err := embedderRegistry.Pin(modelName); err != nil {
					zl.Warn("Failed to pin model",
						zap.String("model", modelName),
						zap.Error(err))
				}
			}
		}
	}

	// Preload specified models at startup (Ollama-compatible)
	// Note: This preloads models that aren't already pinned
	if len(config.Preload) > 0 {
		if err := embedderRegistry.Preload(config.Preload); err != nil {
			zl.Warn("Some models failed to preload", zap.Error(err))
		}
	}

	// Initialize reranker registry with lazy loading
	// Models are discovered at startup but only loaded on first request
	var rerankerRegistry *RerankerRegistry
	if rerankerModelsDir != "" {
		rerankerRegistry, err = NewRerankerRegistry(
			RerankerConfig{
				ModelsDir:       rerankerModelsDir,
				KeepAlive:       keepAlive,
				MaxLoadedModels: uint64(config.MaxLoadedModels),
				PoolSize:        config.PoolSize,
			},
			sessionManager,
			zl.Named("reranker"),
		)
		if err != nil {
			zl.Fatal("Failed to initialize reranker registry", zap.Error(err))
		}
		defer func() { _ = rerankerRegistry.Close() }()

		// If eager loading is requested, preload all models
		if keepAlive == 0 {
			if err := rerankerRegistry.PreloadAll(); err != nil {
				zl.Warn("Failed to preload some reranker models", zap.Error(err))
			}
		}
	}

	// Initialize generator registry with lazy loading
	// Models are discovered at startup but only loaded on first request
	var generatorRegistry *GeneratorRegistry
	if generatorModelsDir != "" {
		generatorRegistry, err = NewGeneratorRegistry(
			GeneratorConfig{
				ModelsDir:       generatorModelsDir,
				KeepAlive:       keepAlive,
				MaxLoadedModels: uint64(config.MaxLoadedModels),
			},
			sessionManager,
			zl.Named("generator"),
		)
		if err != nil {
			zl.Fatal("Failed to initialize generator registry", zap.Error(err))
		}
		defer func() { _ = generatorRegistry.Close() }()

		// If eager loading is requested, preload all models
		if keepAlive == 0 {
			if err := generatorRegistry.PreloadAll(); err != nil {
				zl.Warn("Failed to preload some generator models", zap.Error(err))
			}
		}
	}

	// Initialize NER registry with lazy loading
	// Models are discovered at startup but only loaded on first request
	var nerRegistry *NERRegistry
	var nerModelsDir string
	if config.ModelsDir != "" {
		nerModelsDir = filepath.Join(config.ModelsDir, "recognizers")
	}
	if nerModelsDir != "" {
		nerRegistry, err = NewNERRegistry(
			NERConfig{
				ModelsDir:       nerModelsDir,
				KeepAlive:       keepAlive,
				MaxLoadedModels: uint64(config.MaxLoadedModels),
				PoolSize:        config.PoolSize,
			},
			sessionManager,
			zl.Named("ner"),
		)
		if err != nil {
			zl.Fatal("Failed to initialize NER registry", zap.Error(err))
		}
		defer func() { _ = nerRegistry.Close() }()

		// If eager loading is requested, preload all models
		if keepAlive == 0 {
			if err := nerRegistry.PreloadAll(); err != nil {
				zl.Warn("Failed to preload some NER models", zap.Error(err))
			}
		}
	}

	// Initialize Seq2Seq registry with lazy loading
	// Models are discovered at startup but only loaded on first request
	var seq2seqRegistry *Seq2SeqRegistry
	var seq2seqModelsDir string
	if config.ModelsDir != "" {
		seq2seqModelsDir = filepath.Join(config.ModelsDir, "rewriters")
	}
	if seq2seqModelsDir != "" {
		seq2seqRegistry, err = NewSeq2SeqRegistry(
			Seq2SeqConfig{
				ModelsDir:       seq2seqModelsDir,
				KeepAlive:       keepAlive,
				MaxLoadedModels: uint64(config.MaxLoadedModels),
			},
			sessionManager,
			zl.Named("seq2seq"),
		)
		if err != nil {
			zl.Fatal("Failed to initialize Seq2Seq registry", zap.Error(err))
		}
		defer func() { _ = seq2seqRegistry.Close() }()

		// If eager loading is requested, preload all models
		if keepAlive == 0 {
			if err := seq2seqRegistry.PreloadAll(); err != nil {
				zl.Warn("Failed to preload some Seq2Seq models", zap.Error(err))
			}
		}
	}

	t := &http.Transport{
		MaxIdleConns:        100,
		MaxIdleConnsPerHost: 10,
		IdleConnTimeout:     6 * time.Minute,
		DisableKeepAlives:   false,
		ForceAttemptHTTP2:   true,
	}
	client := &http.Client{
		Timeout:   time.Second * 540,
		Transport: t,
	}
	// Build content security config - use config value or fall back to defaults
	var contentSecurityConfig *scraping.ContentSecurityConfig
	if config.ContentSecurity.MaxDownloadSizeBytes != 0 || config.ContentSecurity.DownloadTimeoutSeconds != 0 || len(config.ContentSecurity.AllowedHosts) > 0 {
		contentSecurityConfig = &config.ContentSecurity
	} else {
		// Default secure settings
		contentSecurityConfig = &scraping.ContentSecurityConfig{
			BlockPrivateIps:        true,
			MaxDownloadSizeBytes:   104857600, // 100MB
			DownloadTimeoutSeconds: 30,
		}
	}

	// Initialize request queue for backpressure control
	var requestTimeout time.Duration
	if config.RequestTimeout != "" && config.RequestTimeout != "0" {
		requestTimeout, err = time.ParseDuration(config.RequestTimeout)
		if err != nil {
			zl.Fatal("Invalid request_timeout duration", zap.String("request_timeout", config.RequestTimeout), zap.Error(err))
		}
	}

	requestQueue := NewRequestQueue(RequestQueueConfig{
		MaxConcurrentRequests: config.MaxConcurrentRequests,
		MaxQueueSize:          config.MaxQueueSize,
		RequestTimeout:        requestTimeout,
	}, zl.Named("queue"))

	// Initialize caches for embeddings, reranking, and NER
	embeddingCache := NewEmbeddingCache(zl.Named("embedding-cache"))
	defer embeddingCache.Close()

	rerankingCache := NewRerankingCache(zl.Named("reranking-cache"))
	defer rerankingCache.Close()

	nerCache := NewNERCache(zl.Named("ner-cache"))
	defer nerCache.Close()

	// Build S3 credentials from config (optional)
	var s3Creds *s3.Credentials
	if config.S3Credentials.Endpoint != "" {
		s3Creds = &config.S3Credentials
	}

	node := &TermiteNode{
		logger: zl,

		embedderRegistry:      embedderRegistry,
		cachedChunker:         cachedChunker,
		rerankerRegistry:      rerankerRegistry,
		generatorRegistry:     generatorRegistry,
		nerRegistry:           nerRegistry,
		seq2seqRegistry:       seq2seqRegistry,
		contentSecurityConfig: contentSecurityConfig,
		s3Credentials:         s3Creds,
		requestQueue:          requestQueue,
		embeddingCache:        embeddingCache,
		rerankingCache:        rerankingCache,
		nerCache:              nerCache,

		client: client,
	}

	// Create API handler using generated ServerInterface
	apiHandler := NewTermiteAPI(zl, node)

	// Create root mux with health endpoints and API handler
	rootMux := http.NewServeMux()

	// Health endpoints (outside /api prefix for k8s compatibility)
	rootMux.HandleFunc("GET /healthz", node.handleHealthz)
	rootMux.HandleFunc("GET /readyz", node.handleReadyz)

	// Generate endpoint (manually registered until OpenAPI codegen is updated)
	rootMux.HandleFunc("POST /api/generate", node.handleApiGenerate)

	// Mount the OpenAPI-generated API handler (includes /api/version)
	rootMux.Handle("/api/", apiHandler)

	// OpenAI-compatible API at /openai/v1/* for standard SDK compatibility
	node.RegisterOpenAIRoutes(rootMux)

	// Serve the embedded dashboard at root (SPA with fallback to index.html)
	addDashboardRoutes(rootMux)

	srv := &http.Server{
		Addr:        u.Host,
		Handler:     corsMiddleware(rootMux),
		ReadTimeout: 540 * time.Second,
	}

	// Start server in goroutine
	serverErr := make(chan error, 1)
	go func() {
		zl.Info("Termite's api server starting", zap.String("address", config.ApiUrl))
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			serverErr <- err
		}
		close(serverErr)
	}()

	// Signal readiness after server starts
	if readyC != nil {
		close(readyC)
	}

	// Wait for context cancellation or server error
	select {
	case err := <-serverErr:
		if err != nil {
			zl.Fatal("HTTP server error", zap.Error(err))
		}
	case <-ctx.Done():
		zl.Info("Shutdown signal received, starting graceful shutdown...")
	}

	// Graceful shutdown
	shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), DefaultShutdownTimeout)
	defer shutdownCancel()

	// Stop accepting new connections
	srv.SetKeepAlivesEnabled(false)

	// Attempt graceful shutdown
	if err := srv.Shutdown(shutdownCtx); err != nil {
		zl.Warn("Graceful shutdown failed, forcing close",
			zap.Error(err),
			zap.Duration("timeout", DefaultShutdownTimeout))
		_ = srv.Close()
	} else {
		zl.Info("Graceful shutdown completed successfully")
	}

	zl.Info("HTTP server stopped")
}
