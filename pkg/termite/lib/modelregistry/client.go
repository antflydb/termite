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

package modelregistry

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"go.uber.org/zap"
)

const (
	// DefaultRegistryURL is the default model registry URL
	DefaultRegistryURL = "https://registry.antfly.io/v1"

	// DefaultTimeout is the default HTTP timeout for metadata requests
	DefaultTimeout = 30 * time.Second

	// DefaultDownloadTimeout is the default timeout for downloading model files
	DefaultDownloadTimeout = 10 * time.Minute
)

// Client is an HTTP client for the model registry
type Client struct {
	baseURL         string
	httpClient      *http.Client
	downloadClient  *http.Client
	logger          *zap.Logger
	progressHandler ProgressHandler
}

// ProgressHandler is called to report download progress
type ProgressHandler func(downloaded, total int64, filename string)

// ClientOption configures the client
type ClientOption func(*Client)

// WithBaseURL sets the registry base URL
func WithBaseURL(url string) ClientOption {
	return func(c *Client) {
		c.baseURL = strings.TrimSuffix(url, "/")
	}
}

// WithLogger sets the logger
func WithLogger(logger *zap.Logger) ClientOption {
	return func(c *Client) {
		c.logger = logger
	}
}

// WithProgressHandler sets the progress handler for downloads
func WithProgressHandler(h ProgressHandler) ClientOption {
	return func(c *Client) {
		c.progressHandler = h
	}
}

// WithTimeout sets the HTTP timeout for metadata requests
func WithTimeout(timeout time.Duration) ClientOption {
	return func(c *Client) {
		c.httpClient.Timeout = timeout
	}
}

// NewClient creates a new registry client
func NewClient(opts ...ClientOption) *Client {
	c := &Client{
		baseURL: DefaultRegistryURL,
		httpClient: &http.Client{
			Timeout: DefaultTimeout,
		},
		downloadClient: &http.Client{
			Timeout: DefaultDownloadTimeout,
		},
		logger: zap.NewNop(),
	}

	for _, opt := range opts {
		opt(c)
	}

	return c
}

// FetchIndex fetches the registry index listing all available models
func (c *Client) FetchIndex(ctx context.Context) (*RegistryIndex, error) {
	url := c.baseURL + "/index.json"
	c.logger.Debug("Fetching registry index", zap.String("url", url))

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, fmt.Errorf("creating request: %w", err)
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("fetching index: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("registry returned status %d", resp.StatusCode)
	}

	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("reading index: %w", err)
	}

	return ParseRegistryIndex(data)
}

// FetchManifest fetches the manifest for a specific model
func (c *Client) FetchManifest(ctx context.Context, modelName string) (*ModelManifest, error) {
	url := fmt.Sprintf("%s/manifests/%s.json", c.baseURL, modelName)
	c.logger.Debug("Fetching model manifest", zap.String("url", url), zap.String("model", modelName))

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, fmt.Errorf("creating request: %w", err)
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("fetching manifest: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode == http.StatusNotFound {
		return nil, fmt.Errorf("model not found: %s", modelName)
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("registry returned status %d", resp.StatusCode)
	}

	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("reading manifest: %w", err)
	}

	return ParseManifest(data)
}

// PullModel downloads a model and its files to the specified directory.
// variants specifies which variant IDs to download (e.g., ["f32", "f16", "i8"]).
// Use "f32" for the default FP32 model (model.onnx).
// Supporting files (tokenizer, config, etc.) are always downloaded.
// If variants is empty, defaults to downloading only "f32".
func (c *Client) PullModel(ctx context.Context, manifest *ModelManifest, modelsDir string, variants []string) error {
	// Determine output directory based on model type (includes owner if present)
	modelDir := filepath.Join(modelsDir, manifest.Type.DirName(), manifest.FullName())

	// Default to f32 if no variants specified
	if len(variants) == 0 {
		variants = []string{VariantF32}
	}

	c.logger.Info("Pulling model",
		zap.String("name", manifest.Name),
		zap.String("type", string(manifest.Type)),
		zap.Strings("variants", variants),
		zap.String("destination", modelDir))

	// Create model directory
	if err := os.MkdirAll(modelDir, 0755); err != nil {
		return fmt.Errorf("creating model directory: %w", err)
	}

	// Build set of requested variants for quick lookup
	requestedVariants := make(map[string]bool)
	for _, v := range variants {
		requestedVariants[v] = true
	}

	// Download supporting files (non-ONNX) and f32 model files if requested
	for _, file := range manifest.Files {
		isONNX := strings.HasSuffix(file.Name, ".onnx")
		if isONNX {
			// Download all ONNX files in base manifest if f32 variant is requested
			// This supports both single-model (model.onnx) and multi-model (visual_model.onnx, text_model.onnx) cases
			if requestedVariants[VariantF32] {
				if err := c.downloadFile(ctx, file, modelDir); err != nil {
					return fmt.Errorf("downloading %s: %w", file.Name, err)
				}
			}
			continue
		}
		// Always download supporting files (tokenizer, config, etc.)
		if err := c.downloadFile(ctx, file, modelDir); err != nil {
			return fmt.Errorf("downloading %s: %w", file.Name, err)
		}
	}

	// Download requested variant files (non-f32)
	for _, variantID := range variants {
		if variantID == VariantF32 {
			continue // Already handled above
		}
		variantEntry, ok := manifest.Variants[variantID]
		if !ok {
			c.logger.Warn("Requested variant not available in manifest",
				zap.String("variant", variantID),
				zap.String("model", manifest.Name))
			continue
		}
		// Download all files in the variant (supports both single and multi-model variants)
		for _, variantFile := range variantEntry.Files {
			if err := c.downloadFile(ctx, variantFile, modelDir); err != nil {
				return fmt.Errorf("downloading variant %s file %s: %w", variantID, variantFile.Name, err)
			}
		}
	}

	c.logger.Info("Model pulled successfully",
		zap.String("name", manifest.Name),
		zap.String("location", modelDir))

	return nil
}

// downloadFile downloads a single file from the registry
func (c *Client) downloadFile(ctx context.Context, file ModelFile, destDir string) error {
	destPath := filepath.Join(destDir, file.Name)

	// Check if file already exists with correct hash
	if c.fileExistsWithHash(destPath, file.Digest) {
		c.logger.Debug("File already exists with correct hash, skipping",
			zap.String("file", file.Name))
		if c.progressHandler != nil {
			c.progressHandler(file.Size, file.Size, file.Name)
		}
		return nil
	}

	// Construct blob URL from digest
	url := fmt.Sprintf("%s/blobs/%s", c.baseURL, file.Digest)
	c.logger.Debug("Downloading file",
		zap.String("file", file.Name),
		zap.String("url", url),
		zap.Int64("size", file.Size))

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return fmt.Errorf("creating request: %w", err)
	}

	resp, err := c.downloadClient.Do(req)
	if err != nil {
		return fmt.Errorf("downloading: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("download returned status %d", resp.StatusCode)
	}

	// Create temp file for download
	tmpPath := destPath + ".tmp"
	tmpFile, err := os.Create(tmpPath)
	if err != nil {
		return fmt.Errorf("creating temp file: %w", err)
	}
	defer func() {
		_ = tmpFile.Close()
		_ = os.Remove(tmpPath) // Clean up on error
	}()

	// Download with progress and hash verification
	hasher := sha256.New()
	var downloaded int64

	reader := resp.Body
	if c.progressHandler != nil {
		reader = &progressReader{
			reader:   resp.Body,
			total:    file.Size,
			filename: file.Name,
			handler:  c.progressHandler,
		}
	}

	writer := io.MultiWriter(tmpFile, hasher)
	downloaded, err = io.Copy(writer, reader)
	if err != nil {
		return fmt.Errorf("writing file: %w", err)
	}

	// Verify size
	if file.Size > 0 && downloaded != file.Size {
		return fmt.Errorf("size mismatch: expected %d, got %d", file.Size, downloaded)
	}

	// Verify hash
	actualHash := "sha256:" + hex.EncodeToString(hasher.Sum(nil))
	if actualHash != file.Digest {
		return fmt.Errorf("hash mismatch: expected %s, got %s", file.Digest, actualHash)
	}

	// Close temp file before rename
	if err := tmpFile.Close(); err != nil {
		return fmt.Errorf("closing temp file: %w", err)
	}

	// Rename to final destination
	if err := os.Rename(tmpPath, destPath); err != nil {
		return fmt.Errorf("renaming file: %w", err)
	}

	c.logger.Debug("File downloaded successfully",
		zap.String("file", file.Name),
		zap.Int64("size", downloaded))

	return nil
}

// fileExistsWithHash checks if a file exists and has the expected hash
func (c *Client) fileExistsWithHash(path string, expectedDigest string) bool {
	f, err := os.Open(path)
	if err != nil {
		return false
	}
	defer func() { _ = f.Close() }()

	hasher := sha256.New()
	if _, err := io.Copy(hasher, f); err != nil {
		return false
	}

	actualHash := "sha256:" + hex.EncodeToString(hasher.Sum(nil))
	return actualHash == expectedDigest
}

// progressReader wraps a reader to report progress
type progressReader struct {
	reader     io.ReadCloser
	downloaded int64
	total      int64
	filename   string
	handler    ProgressHandler
}

func (pr *progressReader) Read(p []byte) (int, error) {
	n, err := pr.reader.Read(p)
	if n > 0 {
		pr.downloaded += int64(n)
		pr.handler(pr.downloaded, pr.total, pr.filename)
	}
	return n, err
}

func (pr *progressReader) Close() error {
	return pr.reader.Close()
}
