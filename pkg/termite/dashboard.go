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

package termite

import (
	"embed"
	"io/fs"
	"net/http"
	"os"
)

//go:embed dashboard
var dashboardFS embed.FS

// spaFileSystem is a custom file system that serves 'index.html' for any path
// that doesn't correspond to an existing file. This is needed for Single-Page
// Applications where routing is handled on the client side.
type spaFileSystem struct {
	root http.FileSystem
}

func (fs spaFileSystem) Open(name string) (http.File, error) {
	f, err := fs.root.Open(name)
	if os.IsNotExist(err) {
		// If the file is not found, serve index.html instead.
		return fs.root.Open("index.html")
	}
	return f, err
}

func addDashboardRoutes(mux *http.ServeMux) {
	// Get a filesystem rooted at the "dashboard" directory.
	subFS, err := fs.Sub(dashboardFS, "dashboard")
	if err != nil {
		panic("could not find dashboard directory in embedded files")
	}

	// Serve the static files for the dashboard application.
	mux.Handle("/", http.FileServer(spaFileSystem{http.FS(subFS)}))
}
