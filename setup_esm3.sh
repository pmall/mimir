#!/bin/bash
set -e

# Ensure we are in the project root
if [ ! -f "pyproject.toml" ]; then
    echo "Error: pyproject.toml not found. Please run this script from the project root."
    exit 1
fi

echo "Ensuring Python 3.11 is installed..."
uv python install 3.11

echo "Installing dependencies..."
uv sync

echo "Checking Hugging Face authentication..."

# Check for 'hf' command, otherwise fall back to 'huggingface-cli'
# Always use uv run hf to use the modern CLI within the project environment
AUTH_CMD="uv run hf auth login"
WHOAMI_CMD="uv run hf auth whoami"

# Check if logged in
if ! $WHOAMI_CMD &> /dev/null; then
    echo "Not logged in to Hugging Face. Please log in now."
    $AUTH_CMD
else
    echo "Already logged in to Hugging Face."
fi

echo "Downloading ESM-3 weights..."
uv run python scripts/download_weights.py

echo "Setup complete!"
