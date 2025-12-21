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

package hugot

import (
	"github.com/knights-analytics/hugot"
	"github.com/knights-analytics/hugot/options"
)

func init() {
	RegisterBackend(&goBackend{})
}

// goBackend implements Backend using the pure Go goMLX runtime.
// This backend is always available as it has no external dependencies.
//
// Advantages:
//   - No CGO required
//   - Single static binary
//   - Easy cross-compilation
//   - Works everywhere
//
// Trade-offs:
//   - Slower inference compared to ONNX Runtime or XLA
type goBackend struct{}

func (b *goBackend) Type() BackendType {
	return BackendGo
}

func (b *goBackend) Name() string {
	return "goMLX (Pure Go)"
}

func (b *goBackend) Available() bool {
	// Pure Go backend is always available
	return true
}

func (b *goBackend) Priority() int {
	// Lowest priority (highest number) - fallback only
	return 100
}

func (b *goBackend) CreateSession(opts ...options.WithOption) (*hugot.Session, error) {
	return hugot.NewGoSession(opts...)
}
