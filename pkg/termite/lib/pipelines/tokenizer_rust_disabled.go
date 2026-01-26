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

//go:build !onnx || !ORT

package pipelines

import (
	"github.com/gomlx/go-huggingface/tokenizers"
	"github.com/gomlx/go-huggingface/tokenizers/api"
)

// loadRustTokenizer is a stub when Rust tokenizer is not available.
func loadRustTokenizer(_ string, _ *api.Config) (tokenizers.Tokenizer, error) {
	return nil, nil // Return nil to signal fallback to Go tokenizer
}

// rustTokenizerAvailable returns false when the Rust tokenizer is not available.
func rustTokenizerAvailable() bool {
	return false
}
