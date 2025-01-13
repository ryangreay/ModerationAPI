import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from datasets import Dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import logging
import os
from typing import Dict, List, Generator, Tuple

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data_in_chunks(
    file_path: str,
    chunk_size: int = 1000,
    text_column: str = "text",
    label_column: str = "label"
) -> Generator[Tuple[pd.DataFrame, pd.DataFrame], None, None]:
    """Load and process data in chunks with validation split"""
    logger.info(f"Starting to load data in chunks of size {chunk_size}")
    
    for chunk_num, chunk in enumerate(pd.read_csv(file_path, chunksize=chunk_size)):
        logger.info(f"Processing chunk {chunk_num + 1}")
        
        # Split chunk into train and validation
        try:
            train_chunk, eval_chunk = train_test_split(
                chunk,
                test_size=0.2,
                stratify=chunk[label_column],
                random_state=42
            )
            
            yield train_chunk, eval_chunk
            
        except ValueError as e:
            logger.error(f"Error splitting chunk {chunk_num + 1}: {str(e)}")
            continue

def compute_metrics(pred):
    """Compute metrics for evaluation"""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    # Check for empty classes
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        logger.warning(f"Only found classes {unique_labels} in evaluation set!")
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, 
        preds, 
        average='binary',
        zero_division=0
    )
    acc = accuracy_score(labels, preds)
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'num_label_0': np.sum(labels == 0),
        'num_label_1': np.sum(labels == 1)
    }

def train_model(
    data_path: str,
    model_name: str = "bert-base-uncased",
    output_dir: str = "models/fine_tuned_bert",
    chunk_size: int = 1000,
    batch_size: int = 16,
    learning_rate: float = 3e-5,
    num_epochs_per_chunk: int = 5
):
    """Train the model on chunks of data"""
    
    # Initialize tokenizer and model
    logger.info(f"Initializing model and tokenizer from {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    def tokenize_function(examples):
        tokenized = tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=512
        )
        # Add labels to the tokenized output
        tokenized["labels"] = examples["label"]
        return tokenized
    
    # Calculate steps for warmup and evaluation
    total_steps = (chunk_size * num_epochs_per_chunk) // batch_size
    warmup_steps = total_steps // 10  # 10% of total steps
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs_per_chunk,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=warmup_steps,
        learning_rate=learning_rate,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        save_total_limit=2,
        gradient_accumulation_steps=2,
        fp16=torch.cuda.is_available(),
        greater_is_better=True,
        remove_unused_columns=True,
    )
    
    # Process and train on chunks
    chunk_metrics = []
    for chunk_num, (train_chunk, eval_chunk) in enumerate(load_data_in_chunks(data_path, chunk_size)):
        logger.info(f"\nTraining on chunk {chunk_num + 1}")
        
        # Create datasets from chunks
        train_dataset = Dataset.from_pandas(train_chunk)
        eval_dataset = Dataset.from_pandas(eval_chunk)
        
        # Tokenize datasets
        train_dataset = train_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=train_dataset.column_names
        )
        eval_dataset = eval_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=eval_dataset.column_names
        )
        
        # Set format for pytorch
        train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        eval_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        
        # Initialize trainer for this chunk
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        # Train on this chunk
        trainer.train()
        
        # Evaluate after chunk
        metrics = trainer.evaluate()
        chunk_metrics.append(metrics)
        logger.info(f"Chunk {chunk_num + 1} metrics: {metrics}")
        
        # Log class distribution
        train_labels = train_chunk['label'].value_counts()
        eval_labels = eval_chunk['label'].value_counts()
        logger.info(f"Train set class distribution: {train_labels}")
        logger.info(f"Eval set class distribution: {eval_labels}")
    
    # Save final model and metrics
    logger.info("Saving final model...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Calculate and log average metrics
    avg_metrics = {
        key: np.mean([m[key] for m in chunk_metrics])
        for key in chunk_metrics[0].keys()
    }
    logger.info(f"Average metrics across all chunks: {avg_metrics}")
    
    return avg_metrics

if __name__ == "__main__":
    os.makedirs("models/fine_tuned_bert", exist_ok=True)
    
    # Train model with chunks
    final_metrics = train_model(
        data_path="data/training_set.csv",
        model_name="bert-base-uncased",
        output_dir="models/fine_tuned_bert",
        chunk_size=2000,
        batch_size=16,
        learning_rate=3e-5,
        num_epochs_per_chunk=5
    )
    
    logger.info("Training completed successfully!")
    logger.info(f"Final average metrics: {final_metrics}") 