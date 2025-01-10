import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import logging
from datetime import datetime

def setup_logging():
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/training_{timestamp}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return log_file

def train_model():
    # Setup
    log_file = setup_logging()
    logging.info("Starting training process")
    
    # Load data
    df = pd.read_csv("data/training_data.csv")
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Create datasets
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=2
    )
    
    # Tokenize function
    def tokenize_and_format(examples):
        tokenized = tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors=None
        )
        # Add labels to the tokenized output
        tokenized['labels'] = examples['label']
        return tokenized
    
    # Prepare datasets
    train_dataset = train_dataset.map(
        tokenize_and_format,
        batched=True,
        remove_columns=train_dataset.column_names
    )
    val_dataset = val_dataset.map(
        tokenize_and_format,
        batched=True,
        remove_columns=val_dataset.column_names
    )
    
    # Set format for pytorch
    columns = ['input_ids', 'attention_mask', 'labels']
    train_dataset.set_format(type='torch', columns=columns)
    val_dataset.set_format(type='torch', columns=columns)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="models/fine_tuned_bert",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_dir="logs",
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
        report_to="none"
    )
    
    # Metrics function
    def compute_metrics(eval_pred):
        predictions = np.argmax(eval_pred.predictions, axis=1)
        return {"accuracy": (predictions == eval_pred.label_ids).astype(np.float32).mean().item()}
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    
    # Train
    logging.info("Starting training...")
    trainer.train()
    
    # Save
    logging.info("Saving model...")
    trainer.save_model("models/fine_tuned_bert")
    tokenizer.save_pretrained("models/fine_tuned_bert")
    logging.info(f"Training complete. Logs saved to: {log_file}")

if __name__ == "__main__":
    os.makedirs("models/fine_tuned_bert", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    train_model() 