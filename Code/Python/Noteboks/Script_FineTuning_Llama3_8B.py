import os
import pandas as pd
import torch
from datasets import Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from sklearn.model_selection import train_test_split
from huggingface_hub import login

# Set environment variables to disable WandB and set notebook name
os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_NOTEBOOK_NAME"] = "LlamaFineTuningNotebook"

# Login to Hugging Face Hub using a specific token
login(token="hf_Uhi...")

# Load the CSV file into a Pandas DataFrame
df = pd.read_csv('train_2M.csv')

# Bits and Bytes configuration for quantization
compute_dtype = getattr(torch, "float16")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=True,
)

# Load the model from the Hugging Face model hub and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B",
    quantization_config=bnb_config,
    device_map={"": 0},
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B", trust_remote_code=True)

# Prepare the tokenizer and resize token embeddings
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model.resize_token_embeddings(len(tokenizer))

# PEFT configuration for Partial Execution Fine-Tuning
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.5,
    r=4,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj", "k_proj"]
)

# Disable cache in model config and apply PEFT modifications
model.config.use_cache = False
model = get_peft_model(model, peft_config)
print(model.print_trainable_parameters())  # Print trainable parameters

# Training arguments for the Trainer
training_arguments = TrainingArguments(
    output_dir='./results',
    per_device_train_batch_size=32,
    gradient_accumulation_steps=12,
    num_train_epochs=3,
    optim='adamw_bnb_8bit',
    save_steps=1000000,
    eval_steps=1000000,
    fp16=True,
    learning_rate=2e-5,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    evaluation_strategy="steps"
)

# Clean and split the data into training and validation sets
df_clean = df.dropna(subset=['domain', 'Labels'])
train_df, test_df = train_test_split(df_clean, test_size=0.01, random_state=42)
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(test_df)

# Function to format prompts with domain and labels
def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['domain'])):
        text = f"#domain: {example['domain'][i]}\n#label: {example['Labels'][i]}"
        output_texts.append(text)
    return output_texts

# Data collator for completion LM tasks
collator = DataCollatorForCompletionOnlyLM(
    response_template="#label:", 
    tokenizer=tokenizer, 
    mlm=False
)

# Initialize the SFTTrainer for model training
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    peft_config=peft_config,
    formatting_func=formatting_prompts_func,
    max_seq_length=256,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=False,
    data_collator=collator,
)

# Start training the model
trainer.train()

# Save the trained model in the current directory
output_model_dir = "./"
model.save_pretrained(output_model_dir)
