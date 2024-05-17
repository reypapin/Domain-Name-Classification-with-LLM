# Domain Name Classification with Contextual Learning

This repository contains code for training models to classify domain names into two categories: Domain Generation Algorithms (DGAs) and normal domain names, using contextual learning techniques. Python is the primary programming language used for development.

## Introduction

Domain names play a crucial role in various internet-related applications. However, distinguishing between legitimate domain names and those generated by malicious algorithms (DGAs) can be challenging. This repository addresses this challenge by leveraging contextual learning methods to develop accurate classifiers.

## Models

The primary models trained in this repository are Language Models (LMs), particularly Large Language Models (LLMs), such as LLama 3 8B. These models are trained on large datasets of domain names to learn contextual representations and patterns that distinguish between DGA-generated and normal domain names.

## Usage

To use the models trained in this repository:

1. **Clone the repository to your local machine.**
    ```sh
    git clone https://github.com/reypapin/Domain-Name-Classification-with-Contextual-Learning.git
    cd Domain-Name-Classification-with-Contextual-Learning
    ```

2. **Install the necessary dependencies listed in `requirements.txt`.**
    ```sh
    chmod +x install.sh
    ./install.sh
    ```

3. **Run the provided script to start Ollama and the model:**
    ```python
    import os
    import threading
    import subprocess
    import requests
    import json

    def ollama():
        os.environ['OLLAMA_HOST'] = '0.0.0.0:11434'
        os.environ['OLLAMA_ORIGINS'] = '*'
        subprocess.Popen(["ollama", "serve"])

    ollama_thread = threading.Thread(target=ollama)
    ollama_thread.start()

    subprocess.run(["ollama", "run", "llama3"])
    ```
## Contributing

Contributions to this repository are welcome! If you have ideas for improvements, new features, or bug fixes, feel free to open an issue or submit a pull request.

1. **Fork the repository.**
2. **Create a new branch for your feature or bugfix.**
3. **Make your changes and commit them with descriptive messages.**
4. **Push your changes to your fork.**
5. **Submit a pull request to the main repository.**





