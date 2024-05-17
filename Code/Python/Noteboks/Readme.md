Notebooks Folder README

This folder contains three Jupyter notebooks, each designed to train and evaluate models
using datasets of domain names. The notebooks were created in Google Colab, where I upload 
the datasets to my Google Drive and access them from Colab. Each notebook installs the 
necessary dependencies for using Ollama models, processes labeled domains through prompts 
to train the models, and evaluates the models' performance on a test dataset.

Notebooks

    ollama_Llama3_8B_prompting.ipynb
        Description: Utilizes the Llama3 8B model from Ollama. This notebook uploads the datasets 
        to Google Drive, installs the required dependencies, and trains the model with labeled 
        domain names. It then classifies domains as either DGA or normal and evaluates the model's 
        performance using test metrics.

    ollama_solar_prompting.ipynb
        Description: Uses the Solar model from Ollama. The workflow is similar to the other notebooks:
        datasets are uploaded to Google Drive, dependencies are installed, and the model is trained 
        with labeled domains. The model's performance is evaluated at the end using test metrics.

    ollama_mistral_prompting.ipynb
        Description: Employs the Mistral model from Ollama. This notebook follows the same procedure: 
        uploading datasets to Google Drive, installing dependencies, training the model with labeled 
        domains, and evaluating the model's performance on a test dataset with relevant metrics.

Workflow

    Upload Datasets: Datasets are uploaded to Google Drive.
    Access Datasets in Colab: Access the datasets from Google Drive within Colab.
    Install Dependencies: Install all necessary dependencies for the Ollama models.
    Train Model: Train the model with labeled domain names through prompts.
    Classify Domains: Classify a domain as DGA or normal using the trained model.
    Evaluate Model: Evaluate the model's performance on a test dataset and display the metrics.

These notebooks provide a comprehensive approach to training, classifying, and evaluating domain names
using different Ollama models. Each notebook's name indicates the specific model used.
