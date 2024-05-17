#!/bin/bash

# Update and install system dependencies
sudo apt-get update
sudo apt-get install -y pciutils

# Install Ollama
curl https://ollama.ai/install.sh | sh

# Install Python dependencies
pip install -r requirements.txt
