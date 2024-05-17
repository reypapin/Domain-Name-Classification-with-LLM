# Models Folder README

This folder is intended to provide information on the various models from Ollama that can be utilized in the code. By simply changing the model name in the code, you can switch between different models available in the Ollama library. Below is a table of example models that can be downloaded and used.

## Model Library

Ollama supports a list of models available on [ollama.com/library](https://ollama.com/library).

Here are some example models that can be downloaded:

| Model                  | Parameters | Size  | Download Command                   |
|------------------------|------------|-------|------------------------------------|
| Llama 3                | 8B         | 4.7GB | `ollama run llama3`                |
| Llama 3                | 70B        | 40GB  | `ollama run llama3:70b`            |
| Phi-3                  | 3.8B       | 2.3GB | `ollama run phi3`                  |
| Mistral                | 7B         | 4.1GB | `ollama run mistral`               |
| Neural Chat            | 7B         | 4.1GB | `ollama run neural-chat`           |
| Starling               | 7B         | 4.1GB | `ollama run starling-lm`           |
| Code Llama             | 7B         | 3.8GB | `ollama run codellama`             |
| Llama 2 Uncensored     | 7B         | 3.8GB | `ollama run llama2-uncensored`     |
| LLaVA                  | 7B         | 4.5GB | `ollama run llava`                 |
| Gemma                  | 2B         | 1.4GB | `ollama run gemma:2b`              |
| Gemma                  | 7B         | 4.8GB | `ollama run gemma:7b`              |
| Solar                  | 10.7B      | 6.1GB | `ollama run solar`                 |

### Note
You should have at least 8 GB of RAM available to run the 7B models, 16 GB to run the 13B models, and 32 GB to run the 33B models.

## Usage

To use a different model in your code, simply change the model name in the appropriate section of your code to one of the models listed above. For example, to use the "Mistral" model, you would use the command:

ollama run mistral


Additionally, you need to change the model name in the payload section of your code. For example, if you are currently using the "Mistral" model, your payload might look like this:

```python
payload = {
    "model": "mistral",
    "temperature": 0.6,
    "stream": False,
    "messages": [
        {"role": "system", "content": "You are an AI assistant!"},
        {"role": "user", "content": question}
    ]
}
```
To switch to another model, such as "Llama 3", you would change the "model" value:
```
payload = {
    "model": "llama3",
    "temperature": 0.6,
    "stream": False,
    "messages": [
        {"role": "system", "content": "You are an AI assistant!"},
        {"role": "user", "content": question}
    ]
}
```
These models can be used in the same manner as described in the notebooks, allowing you to train, classify, and evaluate domain names with different model architectures and sizes.


