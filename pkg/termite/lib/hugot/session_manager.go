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
	"fmt"
	"sync"

	"github.com/knights-analytics/hugot"
	"github.com/knights-analytics/hugot/options"
)

// SessionManager manages Hugot sessions across multiple backends.
// It maintains at most one session per backend type (lazy-created).
//
// IMPORTANT: ONNX Runtime allows only ONE active session at a time.
// SessionManager enforces this constraint by managing shared sessions.
//
// Usage:
//
//	manager := hugot.NewSessionManager()
//	defer manager.Close()
//
//	// Configure backend priority with device preferences
//	manager.SetPriority([]BackendSpec{
//	    {Backend: BackendONNX, Device: DeviceCUDA},
//	    {Backend: BackendXLA, Device: DeviceTPU},
//	    {Backend: BackendONNX, Device: DeviceCPU},
//	    {Backend: BackendGo, Device: DeviceCPU},
//	})
//
//	// Get session for a model with backend restrictions
//	session, backend, err := manager.GetSessionForModel([]string{"onnx", "xla"})
type SessionManager struct {
	sessions map[BackendType]*hugot.Session
	priority []BackendSpec // Configured priority with device preferences
	mu       sync.RWMutex
	closed   bool
}

// NewSessionManager creates a new session manager.
func NewSessionManager() *SessionManager {
	return &SessionManager{
		sessions: make(map[BackendType]*hugot.Session),
	}
}

// SetPriority configures the backend priority order with device preferences.
// This is used when selecting backends for model loading.
func (sm *SessionManager) SetPriority(priority []BackendSpec) {
	sm.mu.Lock()
	defer sm.mu.Unlock()
	sm.priority = make([]BackendSpec, len(priority))
	copy(sm.priority, priority)
}

// GetPriority returns the configured priority or the global default.
func (sm *SessionManager) getPriority() []BackendSpec {
	// If we have a configured priority, use it
	if len(sm.priority) > 0 {
		result := make([]BackendSpec, len(sm.priority))
		copy(result, sm.priority)
		return result
	}

	// Fall back to global priority (BackendType only, DeviceAuto)
	globalPriority := GetPriority()
	result := make([]BackendSpec, len(globalPriority))
	for i, bt := range globalPriority {
		result[i] = BackendSpec{Backend: bt, Device: DeviceAuto}
	}
	return result
}

// GetSession returns a session for the specified backend.
// Creates a new session if one doesn't exist for this backend.
// Returns an error if the backend is unavailable.
func (sm *SessionManager) GetSession(backend BackendType, opts ...options.WithOption) (*hugot.Session, error) {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	if sm.closed {
		return nil, fmt.Errorf("session manager is closed")
	}

	// Return existing session if available
	if session, ok := sm.sessions[backend]; ok {
		return session, nil
	}

	// Get backend
	b, ok := GetBackend(backend)
	if !ok {
		return nil, fmt.Errorf("backend %q not registered", backend)
	}

	if !b.Available() {
		return nil, fmt.Errorf("backend %q not available", backend)
	}

	// Create new session
	session, err := b.CreateSession(opts...)
	if err != nil {
		return nil, fmt.Errorf("create %s session: %w", backend, err)
	}

	sm.sessions[backend] = session
	return session, nil
}

// GetSessionWithFallback attempts to get a session for the preferred backend,
// falling back to alternatives if unavailable.
// Returns the session and the actual backend type used.
func (sm *SessionManager) GetSessionWithFallback(preferred BackendType, opts ...options.WithOption) (*hugot.Session, BackendType, error) {
	// Try preferred backend first
	session, err := sm.GetSession(preferred, opts...)
	if err == nil {
		return session, preferred, nil
	}

	// Get fallback backend
	b, actualType, err := GetBackendWithFallback(preferred)
	if err != nil {
		return nil, "", fmt.Errorf("no available backends (preferred: %s): %w", preferred, err)
	}

	// Get or create session for fallback backend
	session, err = sm.GetSession(actualType, opts...)
	if err != nil {
		return nil, "", fmt.Errorf("create fallback session (%s): %w", b.Name(), err)
	}

	return session, actualType, nil
}

// GetSessionForModel returns a session for a model, respecting its backend restrictions.
// If modelBackends is empty, the model supports all backends and the default priority is used.
// Returns the session and the backend type that was used.
func (sm *SessionManager) GetSessionForModel(modelBackends []string, opts ...options.WithOption) (*hugot.Session, BackendType, error) {
	sm.mu.RLock()
	priority := sm.getPriority()
	sm.mu.RUnlock()

	// Build model backend set for quick lookup
	modelBackendSet := make(map[BackendType]bool)
	for _, b := range modelBackends {
		modelBackendSet[BackendType(b)] = true
	}

	// Try each backend spec in priority order
	for _, spec := range priority {
		// Skip if model doesn't support this backend (unless model has no restrictions)
		if len(modelBackends) > 0 && !modelBackendSet[spec.Backend] {
			continue
		}

		// Apply device setting before creating session
		SetGPUMode(spec.Device.ToGPUMode())

		session, err := sm.GetSession(spec.Backend, opts...)
		if err == nil {
			return session, spec.Backend, nil
		}
	}

	if len(modelBackends) > 0 {
		return nil, "", fmt.Errorf("no available backends matching model requirements %v", modelBackends)
	}
	return nil, "", fmt.Errorf("no available backends")
}

// GetDefaultSession returns a session using the default backend (first available by priority).
func (sm *SessionManager) GetDefaultSession(opts ...options.WithOption) (*hugot.Session, BackendType, error) {
	// Use GetSessionForModel with empty model backends to get first available
	return sm.GetSessionForModel(nil, opts...)
}

// HasSession returns true if a session exists for the given backend.
func (sm *SessionManager) HasSession(backend BackendType) bool {
	sm.mu.RLock()
	defer sm.mu.RUnlock()
	_, ok := sm.sessions[backend]
	return ok
}

// ActiveBackends returns the list of backends with active sessions.
func (sm *SessionManager) ActiveBackends() []BackendType {
	sm.mu.RLock()
	defer sm.mu.RUnlock()

	backends := make([]BackendType, 0, len(sm.sessions))
	for t := range sm.sessions {
		backends = append(backends, t)
	}
	return backends
}

// Close destroys all managed sessions and releases resources.
// After Close, the SessionManager cannot be reused.
func (sm *SessionManager) Close() error {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	if sm.closed {
		return nil
	}

	var firstErr error
	for backend, session := range sm.sessions {
		if session != nil {
			if err := session.Destroy(); err != nil && firstErr == nil {
				firstErr = fmt.Errorf("destroy %s session: %w", backend, err)
			}
		}
	}

	sm.sessions = nil
	sm.closed = true

	return firstErr
}
