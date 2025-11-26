#!/bin/bash

export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"  # This loads nvm
[ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"  # This loads nvm bash_completion


# Check whether nvm is already installed
check_nvm_installed() {
    if [ -s "$NVM_DIR/nvm.sh" ]; then
        echo "nvm is already installed"
        return 0
    else
        echo "nvm is not installed"
        return 1
    fi
}

# Install nvm on Linux
install_nvm_linux() {
    curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.1/install.sh | bash
    source "$NVM_DIR/nvm.sh"
}

# Install nvm on macOS
install_nvm_mac() {
    curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.1/install.sh | bash
    source "$NVM_DIR/nvm.sh"
}

# Install nvm on Windows
install_nvm_windows() {
    echo "Downloading nvm for Windows..."
    curl -o nvm-setup.exe -L https://github.com/coreybutler/nvm/releases/download/1.1.12/nvm-setup.exe
    echo "Installing nvm for Windows..."
    ./nvm-setup.exe
    echo "nvm version:"
    nvm -v
    echo "nvm installation for Windows completed."
    rm -f nvm-setup.exe
}

# Install Node.js 10
install_nodejs() {
    nvm install 10
    nvm use 10
}

# Verify Node.js installation
check_node() {
    node_version=$(node -v)
    echo "Installed Node.js version: $node_version"
    if [[ "$node_version" == v10* ]]; then
        echo "Node.js 10 is installed successfully."
    else
        echo "Node.js 10 is not installed."
        exit 1
    fi
}

# Detect the operating system and install nvm if needed
case "$OSTYPE" in
  linux-gnu*)
    if ! check_nvm_installed; then
        install_nvm_linux
    fi
    ;;
  darwin*)
    if ! check_nvm_installed; then
        install_nvm_mac
    fi
    ;;
  cygwin*|msys*|mingw*|bccwin*|wsl*)
    if ! check_nvm_installed; then
        install_nvm_windows
    fi
    ;;
  *)
    echo "Unsupported OS, You could install nvm manually"
    exit 1
    ;;
esac

# Install Node.js 10
install_nodejs

check_node
