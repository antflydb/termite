#!/bin/sh
# Termite install script
# Based on the Antfly install script approach
set -eu

status() { echo ">>> $*" >&2; }
error() { echo "ERROR: $*" >&2; exit 1; }
warning() { echo "WARNING: $*" >&2; }

TEMP_DIR=$(mktemp -d)
cleanup() { rm -rf "$TEMP_DIR"; }
trap cleanup EXIT

available() { command -v "$1" >/dev/null; }

require() {
    local MISSING=''
    for TOOL in "$@"; do
        if ! available "$TOOL"; then
            MISSING="$MISSING $TOOL"
        fi
    done

    if [ -n "$MISSING" ]; then
        error "Missing required tools:$MISSING. Please install them and try again."
    fi
}

# Detect OS and architecture
detect_platform() {
    OS=$(uname -s | tr '[:upper:]' '[:lower:]')
    ARCH=$(uname -m)

    case "$OS" in
        linux) OS="Linux" ;;
        darwin) OS="Darwin" ;;
        *) error "Unsupported operating system: $OS" ;;
    esac

    case "$ARCH" in
        x86_64|amd64) ARCH="x86_64" ;;
        aarch64|arm64) ARCH="arm64" ;;
        *) error "Unsupported architecture: $ARCH" ;;
    esac

    # Darwin only supports arm64 (Apple Silicon)
    if [ "$OS" = "Darwin" ] && [ "$ARCH" = "x86_64" ]; then
        error "macOS x86_64 is not supported. Apple Silicon (arm64) is required."
    fi

    echo "$OS $ARCH"
}

# Download and install termite
install_termite() {
    require curl tar

    status "Detecting platform..."
    read -r OS ARCH <<EOF
$(detect_platform)
EOF
    status "Detected platform: $OS $ARCH"

    VERSION="${1:-latest}"

    # Handle 'latest' version
    if [ "$VERSION" = "latest" ]; then
        status "Fetching latest version..."
        # Try to get version from metadata
        LATEST_URL="https://releases.antfly.io/termite/latest/metadata.json"
        if VERSION_INFO=$(curl -fsSL "$LATEST_URL" 2>/dev/null); then
            VERSION=$(echo "$VERSION_INFO" | grep -o '"tag":"[^"]*"' | head -1 | cut -d'"' -f4)
        fi
        if [ -z "$VERSION" ] || [ "$VERSION" = "latest" ]; then
            error "Could not determine latest version. Please specify a version explicitly."
        fi
    fi

    # Normalize version: TAG has termite-v prefix, VERSION_NUM does not
    # GoReleaser uses .Tag (with prefix) for paths and .Version (without prefix) for filenames
    case "$VERSION" in
        termite-v*) TAG="$VERSION"; VERSION_NUM="${VERSION#termite-v}" ;;
        v*) TAG="termite-$VERSION"; VERSION_NUM="${VERSION#v}" ;;
        *)  TAG="termite-v$VERSION"; VERSION_NUM="$VERSION" ;;
    esac

    status "Installing Termite $EDITION version $TAG..."

    # Construct download URL
    # Default: termite_VERSION_OS_ARCH.tar.gz
    # Omni:   termite-omni_VERSION_OS_ARCH.tar.gz
    if [ "$EDITION" = "omni" ]; then
        ARCHIVE_NAME="termite-omni_${VERSION_NUM}_${OS}_${ARCH}.tar.gz"
    else
        ARCHIVE_NAME="termite_${VERSION_NUM}_${OS}_${ARCH}.tar.gz"
    fi
    DOWNLOAD_URL="https://releases.antfly.io/termite/${TAG}/${ARCHIVE_NAME}"

    status "Downloading from $DOWNLOAD_URL..."
    if ! curl -fsSL "$DOWNLOAD_URL" -o "$TEMP_DIR/$ARCHIVE_NAME"; then
        error "Failed to download Termite. Please check your internet connection and the version number."
    fi

    status "Extracting archive..."
    tar -xzf "$TEMP_DIR/$ARCHIVE_NAME" -C "$TEMP_DIR"

    # Determine install location
    if [ "$(id -u)" -eq 0 ]; then
        # Running as root
        INSTALL_DIR="/usr/local/bin"
        LIB_DIR="/usr/local/lib/termite"
    else
        # Running as regular user
        INSTALL_DIR="$HOME/.local/bin"
        LIB_DIR="$HOME/.local/lib/termite"
        mkdir -p "$INSTALL_DIR"
    fi

    status "Installing binary to $INSTALL_DIR..."

    # Install termite
    if [ -f "$TEMP_DIR/termite" ]; then
        install_binary "$TEMP_DIR/termite" "$INSTALL_DIR/termite"
    else
        error "termite binary not found in archive"
    fi

    # Install bundled libraries (omni edition)
    if [ -d "$TEMP_DIR/lib" ]; then
        status "Installing bundled libraries to $LIB_DIR..."
        if [ -w "$(dirname "$LIB_DIR")" ] || [ "$(id -u)" -eq 0 ]; then
            mkdir -p "$LIB_DIR"
            cp -r "$TEMP_DIR/lib/"* "$LIB_DIR/"
        else
            sudo mkdir -p "$LIB_DIR"
            sudo cp -r "$TEMP_DIR/lib/"* "$LIB_DIR/"
        fi
        status "Installed bundled libraries to $LIB_DIR"
    fi

    status "Termite installation complete!"
    status ""
    status "Run 'termite --help' to get started"

    if [ "$EDITION" = "omni" ]; then
        status ""
        status "Omni edition includes ONNX Runtime and XLA backends."
        status "Libraries installed to $LIB_DIR"
    fi

    # Check if install dir is in PATH
    case ":$PATH:" in
        *":$INSTALL_DIR:"*) ;;
        *)
            warning "$INSTALL_DIR is not in your PATH"
            warning "Add the following to your shell profile (~/.bashrc, ~/.zshrc, etc.):"
            warning "  export PATH=\"\$PATH:$INSTALL_DIR\""
            ;;
    esac
}

# Install a single binary to the target path
install_binary() {
    SRC="$1"
    DST="$2"
    if [ -w "$(dirname "$DST")" ] || [ "$(id -u)" -eq 0 ]; then
        mv "$SRC" "$DST"
        chmod +x "$DST"
    else
        status "Need sudo permission to install to $(dirname "$DST")"
        sudo mv "$SRC" "$DST"
        sudo chmod +x "$DST"
    fi
    status "Installed $(basename "$DST") to $DST"
}

# Main execution
main() {
    EDITION="default"
    VERSION=""

    while [ $# -gt 0 ]; do
        case "$1" in
            -h|--help)
                cat <<EOF
Termite Installer

Usage:
  curl -fsSL https://releases.antfly.io/termite/latest/install.sh | sh
  curl -fsSL https://releases.antfly.io/termite/latest/install.sh | sh -s -- --omni
  curl -fsSL https://releases.antfly.io/termite/latest/install.sh | sh -s -- --omni termite-v0.1.0

Options:
  -h, --help    Show this help message
  --omni        Install the omni edition (includes ONNX Runtime + XLA backends)
  [version]     Install a specific version (e.g., termite-v0.1.0, v0.1.0, or 0.1.0)

Environment:
  This script will automatically detect your OS and architecture,
  download the appropriate binary, and install it.

  By default, it installs to:
    - /usr/local/bin (if running as root)
    - ~/.local/bin (if running as regular user)

Termite is a standalone ML inference service for embeddings, chunking, and reranking.

For more information, visit: https://docs.antfly.io/docs/guides/termite
EOF
                exit 0
                ;;
            --omni)
                EDITION="omni"
                shift
                ;;
            *)
                VERSION="$1"
                shift
                ;;
        esac
    done

    export EDITION
    install_termite "${VERSION:-latest}"
}

main "$@"
