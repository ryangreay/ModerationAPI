import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

def load_data_in_chunks(file_path, chunk_size=1000):
    """Load and process data in chunks"""
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        train_chunk, val_chunk = train_test_split(chunk, test_size=0.2, random_state=42)
        yield train_chunk, val_chunk

def train_model():
    # Load tokenizer and model
    print("Loading model and tokenizer...")
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
        tokenized['labels'] = examples['label']
        return tokenized

    # Training arguments
    training_args = TrainingArguments(
        output_dir="models/fine_tuned_bert",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        warmup_ratio=0.1,
        weight_decay=0.01,
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        load_best_model_at_end=True,
        push_to_hub=False,
        report_to="none",
        # Add gradient accumulation to handle larger effective batch sizes
        gradient_accumulation_steps=4
    )
    
    # Metrics function
    def compute_metrics(eval_pred):
        predictions = np.argmax(eval_pred.predictions, axis=1)
        return {"accuracy": (predictions == eval_pred.label_ids).astype(np.float32).mean().item()}

    # Process and train in chunks
    chunk_size = 1000  # Adjust based on your memory constraints
    chunk_number = 0
    
    for train_chunk, val_chunk in load_data_in_chunks("data/training_data.csv", chunk_size):
        chunk_number += 1
        print(f"\nProcessing chunk {chunk_number}...")
        
        # Create datasets from chunks
        train_dataset = Dataset.from_pandas(train_chunk)
        val_dataset = Dataset.from_pandas(val_chunk)
        
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
        
        # Initialize trainer for this chunk
        trainer = Trainer(
            model=model,  # Use the same model across chunks
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
        )
        
        # Train on this chunk
        print(f"Training on chunk {chunk_number}...")
        trainer.train()
        
        # Evaluate after each chunk
        metrics = trainer.evaluate()
        print(f"Chunk {chunk_number} metrics:", metrics)
    
    print("Saving final model...")
    trainer.save_model("models/fine_tuned_bert")
    tokenizer.save_pretrained("models/fine_tuned_bert")
    print("Training complete!")

if __name__ == "__main__":
    os.makedirs("models/fine_tuned_bert", exist_ok=True)
    train_model() 