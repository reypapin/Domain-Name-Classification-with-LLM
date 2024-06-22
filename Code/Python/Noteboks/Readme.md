# Notebooks Folder README

This folder contains Jupyter notebooks designed to train and evaluate models using datasets of domain names. The notebooks were created in Google Colab, where datasets are uploaded to Google Drive and accessed from Colab. Each notebook installs the necessary dependencies for using Ollama models, processes labeled domains through prompts to train the models, and evaluates the models' performance on a test dataset.

## Notebooks

1. **ollama_Llama3_8B_prompting.ipynb**
   - **Description:** Utilizes the Llama3 8B model from Ollama. This notebook uploads the datasets to Google Drive, installs the required dependencies, and trains the model with labeled domain names. It then classifies domains as either DGA or normal and evaluates the model's performance using test metrics.

2. **ollama_solar_prompting.ipynb**
   - **Description:** Uses the Solar model from Ollama. The workflow is similar to the other notebooks: datasets are uploaded to Google Drive, dependencies are installed, and the model is trained with labeled domains. The model's performance is evaluated at the end using test metrics.

3. **ollama_mistral_prompting.ipynb**
   - **Description:** Employs the Mistral model from Ollama. This notebook follows the same procedure: uploading datasets to Google Drive, installing dependencies, training the model with labeled domains, and evaluating the model's performance on a test dataset with relevant metrics.

4. **ollama_Llama3_8B_prompting_n_size.ipynb**
   - **Description:** Similar to `ollama_Llama3_8B_prompting.ipynb`, but specifically evaluates the model's performance for different training batch sizes.

5. **ollama_openhermes_prompting_n_size.ipynb**
   - **Description:** Similar to `ollama_mistral_prompting.ipynb`, but specifically evaluates the model's performance for different training batch sizes.

6. **ollama_Llama3_8B_prompting_n_size_family.ipynb**
   - **Description:** This notebook experiments with training the Llama3 model from Ollama using three different training batch sizes, excluding the 'ramnit' family. The resulting models are then evaluated on a test dataset that contains only 'ramnit' family domains and normal domains. The objective is to observe how the model performs when encountering a specific family that was not seen during training.

7. **ollama_Llama3_8B_prompting_n_size_feature_comparation.ipynb**
   - **Description:** This notebook compares the performance of the Llama3 8B model using two types of prompts: one with domain features such as length, number of vowels, consonants, digits, hyphens, presence of numbers, and top-level domain, and another without these features. The models are evaluated with training batch sizes of 100, 200, and 500. The results indicate that the model without domain features performs better than the one with features.

8. **ollama_mistral_test_2_prompt_reason.ipynb**
   - **Description:** Introduces a new notebook where the Mistral model from Ollama undergoes testing using a two-prompt approach. In the first prompt, the model is presented with a set number of labeled domains to learn from, followed by 10 unlabeled domains for classification and a brief reasoning of its decision. The second prompt incorporates the question and response from the first prompt, along with the labels of the 10 classified domains, aiming to refine the model's responses and analyze the validity of its reasoning. Additionally, the model is tasked with classifying a new domain.

9. **ollama_mistral_8B_prompting_n_size.ipynb**
   - **Description:** In this notebook, the Mistral model is evaluated with different training batch sizes: 100, 1000, and 1500. The notebook uses a prompt where labeled sample domains are provided, and the model is asked to evaluate a new domain name. The purpose of this notebook is to compare the results with those from another notebook that includes reasoning in its prompt. Ultimately, the results show that the prompt with reasoning performs better than the one without reasoning.

10. **ollama_Llama3_8B_prompting_n_size_family1.ipynb**
    - **Description:** This notebook experiments with training the Llama3 model from Ollama using three different training batch sizes, excluding the 'fobber' family. The resulting models are then evaluated on a test dataset that contains only 'ramnit' family domains and normal domains. The objective is to observe how the model performs when encountering a specific family that was not seen during training.

11. **Llama3_eval_JS.ipynb**
    - **Description:** This notebook evaluates the Llama3_8B model from Ollama using different families of DGAs. We select 30 sets, each consisting of 100 domains, composed of 50 DGA domains from a specific family and 50 normal domains. These 30 batches are evaluated by the model, and we calculate the mean and standard deviation for the metrics. For each family, the files containing the predictions and the actual labels are saved.

12. **Test_Llama3_FineTuning.ipynb**
    - **Description:** This notebook evaluates the fine-tuning performance of the Llama3 8B model using specific adapter loading techniques. It assesses the model's capability to classify domain names as either DGA or normal across various DGA families. Each family is segmented into chunks of 50 domains, which are matched with blocks of legitimate domains. The first 30 matched blocks are evaluated, and the deviation in metrics between these blocks is calculated. The notebook provides flexibility to run evaluations for all families across 30 iterations or for a selected number of families. Additionally, it allows for iterative evaluations where initial runs cover a specified number of chunks per family, followed by continuation. Evaluation metrics are printed during execution, and results for each family are saved as files for future reference. This notebook was executed on Google Colab using a T4 GPU.



## Workflow

1. **Upload Datasets:** Datasets are uploaded to Google Drive.
2. **Access Datasets in Colab:** Access the datasets from Google Drive within Colab.
3. **Install Dependencies:** Install all necessary dependencies for the Ollama models.
4. **Train Model:** Train the model with labeled domain names through prompts.
5. **Classify Domains:** Classify a domain as DGA or normal using the trained model.
6. **Evaluate Model:** Evaluate the model's performance on a test dataset and display the metrics.

These notebooks provide a comprehensive approach to training, classifying, and evaluating domain names using different Ollama models. Each notebook's name indicates the specific model used. The new notebooks, `ollama_Llama3_8B_prompting_n_size.ipynb` and `ollama_openhermes_prompting_n_size.ipynb`, specifically focus on evaluating the models' performance for different training batch sizes. The `ollama_Llama3_8B_prompting_n_size_family.ipynb` notebook evaluates how the Llama3 model performs when trained without the 'ramnit' family and tested on a dataset including only 'ramnit' family domains and normal domains.


