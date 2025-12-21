# Termite Operator Makefile
# For managing the Termite TPU ML inference operator

# Image URLs for building/pushing
OPERATOR_IMG ?= ghcr.io/antflydb/termite-operator:latest
PROXY_IMG ?= ghcr.io/antflydb/termite-proxy:latest
TERMITE_IMG ?= ghcr.io/antflydb/termite:latest
VERSION ?= latest

# Get the currently used golang install path
ifeq (,$(shell go env GOBIN))
GOBIN=$(shell go env GOPATH)/bin
else
GOBIN=$(shell go env GOBIN)
endif

# Local bin directory for tools
LOCALBIN ?= $(shell pwd)/bin
$(LOCALBIN):
	mkdir -p $(LOCALBIN)

# Tool versions
ENVTEST_K8S_VERSION ?= 1.31.0

# Tool binaries
ENVTEST ?= $(LOCALBIN)/setup-envtest

# Setting SHELL to bash allows bash commands to be executed by recipes.
SHELL = /usr/bin/env bash -o pipefail
.SHELLFLAGS = -ec

.PHONY: all
all: build

##@ General

.PHONY: help
help: ## Display this help.
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Development

.PHONY: build-docs
build-docs: ## Bundle and lint OpenAPI specification.
	npx @redocly/cli@latest bundle pkg/termite/openapi.yaml -o openapi.yaml
	npx @redocly/cli@latest lint openapi.yaml

.PHONY: generate
generate: build-docs ## Generate CRDs, DeepCopy methods, and RBAC.
	@echo "Generating manifests..."
	cd pkg/client && go generate ./...
	cd pkg/operator && go generate ./...
	cd pkg/proxy && go generate ./...
	cd pkg/termite && go generate ./...
	@echo "Generated CRDs in pkg/operator/manifests/crd/"
	@echo "Generated RBAC in pkg/operator/manifests/rbac/"

.PHONY: manifests
manifests: generate ## Alias for generate.

.PHONY: fmt
fmt: ## Run go fmt against code.
	cd pkg/client && go fmt ./...
	cd pkg/operator && go fmt ./...
	cd pkg/proxy && go fmt ./...
	cd pkg/termite && go fmt ./...

.PHONY: vet
vet: ## Run go vet against code.
	cd pkg/client && go vet ./...
	cd pkg/operator && go vet ./...
	cd pkg/proxy && go vet ./...
	cd pkg/termite && go vet ./...

.PHONY: test
test: generate fmt vet envtest ## Run tests.
	cd pkg/client && go test -v ./...
	cd pkg/proxy && go test -v ./...
	cd pkg/termite && go test -v ./...
	cd pkg/operator && KUBEBUILDER_ASSETS="$(shell $(ENVTEST) use $(ENVTEST_K8S_VERSION) --bin-dir $(LOCALBIN) -p path)" go test -v ./...

.PHONY: lint
lint: ## Run linters against code (modernize, golangci-lint, testifylint).
	cd pkg/client && go run golang.org/x/tools/gopls/internal/analysis/modernize/cmd/modernize@latest -fix -test ./...
	cd pkg/operator && go run golang.org/x/tools/gopls/internal/analysis/modernize/cmd/modernize@latest -fix -test ./...
	cd pkg/proxy && go run golang.org/x/tools/gopls/internal/analysis/modernize/cmd/modernize@latest -fix -test ./...
	cd pkg/termite && go run golang.org/x/tools/gopls/internal/analysis/modernize/cmd/modernize@latest -fix -test ./...
	cd pkg/client && go run github.com/golangci/golangci-lint/v2/cmd/golangci-lint@latest run --fix ./...
	cd pkg/operator && go run github.com/golangci/golangci-lint/v2/cmd/golangci-lint@latest run --fix ./...
	cd pkg/proxy && go run github.com/golangci/golangci-lint/v2/cmd/golangci-lint@latest run --fix ./...
	cd pkg/termite && go run github.com/golangci/golangci-lint/v2/cmd/golangci-lint@latest run --fix ./...
	cd pkg/client && go run github.com/Antonboom/testifylint@latest --fix ./...
	cd pkg/operator && go run github.com/Antonboom/testifylint@latest --fix ./...
	cd pkg/proxy && go run github.com/Antonboom/testifylint@latest --fix ./...
	cd pkg/termite && go run github.com/Antonboom/testifylint@latest --fix ./...

.PHONY: update-deps
update-deps: ## Update Go dependencies to latest versions.
	cd pkg/client && go get -u ./... && go mod tidy
	cd pkg/operator && go get -u ./... && go mod tidy
	cd pkg/proxy && go get -u ./... && go mod tidy
	cd pkg/termite && go get -u ./... && go mod tidy

##@ Build

.PHONY: build
build: generate fmt vet build-operator build-proxy build-termite ## Build all binaries.

.PHONY: build-operator
build-operator: ## Build operator binary.
	@echo "Building operator..."
	go build -o bin/manager ./pkg/operator/cmd/termite-operator

.PHONY: build-proxy
build-proxy: ## Build proxy binary.
	@echo "Building proxy..."
	go build -o bin/termite-proxy ./pkg/proxy/cmd/termite-proxy

.PHONY: build-termite
build-termite: ## Build termite binary.
	@echo "Building termite..."
	go build -o bin/termite ./pkg/termite/cmd

.PHONY: run-operator
run-operator: generate fmt vet ## Run the operator locally.
	go run ./pkg/operator/cmd/termite-operator

##@ Docker

.PHONY: docker-build
docker-build: docker-build-operator docker-build-proxy docker-build-termite ## Build all docker images.

.PHONY: docker-build-operator
docker-build-operator: ## Build operator docker image.
	@echo "Building operator Docker image..."
	docker build -f Dockerfile.operator -t ${OPERATOR_IMG} .

.PHONY: docker-build-proxy
docker-build-proxy: ## Build proxy docker image.
	@echo "Building proxy Docker image..."
	docker build -f Dockerfile.proxy -t ${PROXY_IMG} .

.PHONY: docker-build-termite
docker-build-termite: ## Build termite docker image.
	@echo "Building termite Docker image..."
	docker build -f Dockerfile.termite -t ${TERMITE_IMG} .

.PHONY: docker-push
docker-push: docker-push-operator docker-push-proxy docker-push-termite ## Push all docker images.

.PHONY: docker-push-operator
docker-push-operator: ## Push operator docker image.
	@echo "Pushing operator Docker image..."
	docker push ${OPERATOR_IMG}

.PHONY: docker-push-proxy
docker-push-proxy: ## Push proxy docker image.
	@echo "Pushing proxy Docker image..."
	docker push ${PROXY_IMG}

.PHONY: docker-push-termite
docker-push-termite: ## Push termite docker image.
	@echo "Pushing termite Docker image..."
	docker push ${TERMITE_IMG}

##@ Multi-arch Docker (for release)

.PHONY: docker-buildx-operator
docker-buildx-operator: ## Build and push multi-arch operator image.
	@echo "Building multi-arch operator image..."
	docker buildx build --platform linux/amd64,linux/arm64 \
		-f Dockerfile.operator -t ${OPERATOR_IMG} --push .

.PHONY: docker-buildx-proxy
docker-buildx-proxy: ## Build and push multi-arch proxy image.
	@echo "Building multi-arch proxy image..."
	docker buildx build --platform linux/amd64,linux/arm64 \
		-f Dockerfile.proxy -t ${PROXY_IMG} --push .

##@ Deployment

.PHONY: deploy-samples
deploy-samples: ## Deploy sample TermitePools.
	@echo "Deploying sample pools..."
	kubectl apply -f config/samples/

.PHONY: undeploy-samples
undeploy-samples: ## Remove sample TermitePools.
	@echo "Removing sample pools..."
	kubectl delete -f config/samples/ --ignore-not-found=true

##@ Release

.PHONY: release
release: test docker-build ## Prepare release artifacts.
	@echo "Preparing release artifacts..."
	@echo "  Tests passed"
	@echo "  Docker images built"
	@echo ""
	@echo "Release ready! Don't forget to:"
	@echo "  1. Tag the release: git tag v${VERSION}"
	@echo "  2. Push the tag: git push origin v${VERSION}"
	@echo "  3. Push the images: make docker-push"

.PHONY: release-all
release-all: release docker-push ## Complete release including pushing images.
	@echo "Release complete!"

##@ Utilities

.PHONY: clean
clean: ## Clean build artifacts.
	@echo "Cleaning build artifacts..."
	rm -rf bin/
	rm -f cover.out

.PHONY: verify
verify: test lint ## Verify code quality.
	@echo "Code verification complete"

.PHONY: dev-setup
dev-setup: envtest ## Set up development environment.
	@echo "Setting up development environment..."
	go install sigs.k8s.io/controller-tools/cmd/controller-gen@latest
	@echo "Development environment ready"

.PHONY: envtest
envtest: $(ENVTEST) ## Download setup-envtest locally if necessary.
$(ENVTEST): $(LOCALBIN)
	$(call go-install-tool,$(ENVTEST),sigs.k8s.io/controller-runtime/tools/setup-envtest,latest)

# go-install-tool will 'go install' any package with custom target and target directory
define go-install-tool
@[ -f $(1) ] || { \
set -e; \
package=$(2)@$(3) ;\
echo "Downloading $${package}" ;\
GOBIN=$(LOCALBIN) go install $${package} ;\
}
endef

##@ Local Development

.PHONY: kind-create
kind-create: ## Create a local kind cluster for testing.
	@echo "Creating kind cluster..."
	kind create cluster --name termite-test

.PHONY: kind-delete
kind-delete: ## Delete the local kind cluster.
	@echo "Deleting kind cluster..."
	kind delete cluster --name termite-test

.PHONY: kind-deploy
kind-deploy: docker-build-operator kind-load deploy ## Build, load to kind, and deploy.
	@echo "Deployed to kind cluster"

.PHONY: kind-load
kind-load: ## Load operator image into kind cluster.
	@echo "Loading image into kind cluster..."
	kind load docker-image ${OPERATOR_IMG} --name termite-test

##@ GKE TPU Development

.PHONY: gke-deploy
gke-deploy: deploy deploy-samples ## Deploy operator and sample pools to GKE.
	@echo "Deployed to GKE cluster"
	@echo ""
	@echo "Monitor pools:"
	@echo "  kubectl get termitepools -n termite-operator-namespace --watch"

.PHONY: gke-logs
gke-logs: ## View operator logs on GKE.
	kubectl logs -f deployment/termite-operator -n termite-operator-namespace

.PHONY: gke-status
gke-status: ## Show status of Termite deployment on GKE.
	@echo "=== Namespace ==="
	kubectl get namespace termite-operator-namespace
	@echo ""
	@echo "=== CRDs ==="
	kubectl get crd | grep termite
	@echo ""
	@echo "=== Operator ==="
	kubectl get deployment termite-operator -n termite-operator-namespace
	@echo ""
	@echo "=== Pools ==="
	kubectl get termitepools -n termite-operator-namespace
	@echo ""
	@echo "=== Routes ==="
	kubectl get termiteroutes -n termite-operator-namespace
	@echo ""
	@echo "=== StatefulSets ==="
	kubectl get statefulsets -n termite-operator-namespace
	@echo ""
	@echo "=== Pods ==="
	kubectl get pods -n termite-operator-namespace

##@ Documentation

.PHONY: docs
docs: ## Show documentation locations.
	@echo "Documentation available:"
	@echo "  - CLAUDE.md: Developer guide"
	@echo "  - devops/comprehensive/README.md: Deployment guide"
	@echo "  - devops/comprehensive/04-pools.yaml: Pool examples"
	@echo "  - deploy/kubernetes/routes.yaml: Route examples"

.PHONY: examples
examples: ## Show example usage.
	@echo "Example usage:"
	@echo ""
	@echo "1. Deploy the operator:"
	@echo "   make deploy"
	@echo ""
	@echo "2. Create a TermitePool:"
	@echo "   kubectl apply -f config/samples/"
	@echo ""
	@echo "3. Check pool status:"
	@echo "   kubectl get termitepools -n termite-operator-namespace"
	@echo ""
	@echo "4. View pool details:"
	@echo "   kubectl describe termitepool read-heavy-embedders -n termite-operator-namespace"

##@ E2E Testing

# Detect OS and architecture for library paths
UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)
ifeq ($(UNAME_S),Darwin)
    ifeq ($(UNAME_M),arm64)
        PLATFORM := darwin-arm64
    else
        PLATFORM := darwin-amd64
    endif
else
    ifeq ($(UNAME_M),aarch64)
        PLATFORM := linux-arm64
    else
        PLATFORM := linux-amd64
    endif
endif

.PHONY: e2e-deps
e2e-deps: ## Download ONNX Runtime and PJRT for omni builds.
	@echo "Downloading ONNX Runtime..."
	./scripts/download-onnxruntime.sh
	@echo "Downloading PJRT..."
	./scripts/download-pjrt.sh

.PHONY: e2e
e2e: e2e-deps ## Run E2E tests with omni build (ONNX + XLA).
	@echo "Running E2E tests with omni build..."
	@echo "This will download the CLIP model (~500MB) on first run."
	@echo "Platform: $(PLATFORM)"
	export ONNXRUNTIME_ROOT=$$(pwd)/onnxruntime && \
	export PJRT_ROOT=$$(pwd)/pjrt && \
	export CGO_ENABLED=1 && \
	export LIBRARY_PATH=$$(pwd)/onnxruntime/$(PLATFORM)/lib:$$LIBRARY_PATH && \
	export LD_LIBRARY_PATH=$$(pwd)/onnxruntime/$(PLATFORM)/lib:$$LD_LIBRARY_PATH && \
	export DYLD_LIBRARY_PATH=$$(pwd)/onnxruntime/$(PLATFORM)/lib:$$DYLD_LIBRARY_PATH && \
	cd e2e && go mod tidy && \
	go test -v -tags="onnx,ORT,xla,XLA" -timeout 15m ./...

.PHONY: e2e-onnx
e2e-onnx: ## Run E2E tests with ONNX only (no XLA).
	@echo "Running E2E tests with ONNX build..."
	@echo "Platform: $(PLATFORM)"
	./scripts/download-onnxruntime.sh
	export ONNXRUNTIME_ROOT=$$(pwd)/onnxruntime && \
	export CGO_ENABLED=1 && \
	export LIBRARY_PATH=$$(pwd)/onnxruntime/$(PLATFORM)/lib:$$LIBRARY_PATH && \
	export LD_LIBRARY_PATH=$$(pwd)/onnxruntime/$(PLATFORM)/lib:$$LD_LIBRARY_PATH && \
	export DYLD_LIBRARY_PATH=$$(pwd)/onnxruntime/$(PLATFORM)/lib:$$DYLD_LIBRARY_PATH && \
	cd e2e && go mod tidy && \
	go test -v -tags="onnx,ORT" -timeout 15m ./...
