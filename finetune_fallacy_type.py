import numpy as np
from datasets import load_dataset, DatasetDict, ClassLabel
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report

# The 8 fallacy types the model will learn to distinguish
FALLACY_TYPES = [
    "authority",
    "blackwhite",
    "hasty_generalization",
    "natural",
    "population",
    "slippery_slope",
    "tradition",
    "worse_problems",
]
LABEL2ID = {label: idx for idx, label in enumerate(FALLACY_TYPES)}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    # Calculate Precision, Recall, and F1 (Macro) across all 8 fallacy types
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='macro', zero_division=0
    )
    acc = accuracy_score(labels, predictions)

    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def main():
    # 1. Load the dataset from JSONL
    dataset = load_dataset('json', data_files='touchefallacy_2026_train.jsonl')

    # 2. Prepare splits (80/10/10)
    # Encode labels first so 'label' exists when stratify_by_column is called
    # Filter out rows where fallacy_type is missing (not a fallacy example)
    def encode_label(examples):
        examples['label'] = LABEL2ID[examples['fallacy_type']]
        return examples

    filtered = dataset['train'].filter(lambda x: x['fallacy_type'] is not None)
    encoded = filtered.map(encode_label)

    # Cast 'label' to ClassLabel so stratify_by_column works
    encoded = encoded.cast_column('label', ClassLabel(names=FALLACY_TYPES))

    # First split into 80% train and 20% temp
    train_test_split = encoded.train_test_split(test_size=0.2, seed=42, stratify_by_column='label')
    # Split the 20% temp into 50% validation and 50% test (results in 10% each of total)
    test_val_split = train_test_split['test'].train_test_split(test_size=0.5, seed=42, stratify_by_column='label')

    raw_datasets = DatasetDict({
        'train': train_test_split['train'],
        'validation': test_val_split['train'],
        'test': test_val_split['test']
    })

    # 3. Tokenization
    model_checkpoint = "roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    def tokenize_function(examples):
        # Using 'text_base' as input and 'fallacy_type' as label
        return tokenizer(examples['text_base'], truncation=True, padding=True)

    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

    # Remove all original columns except 'label' (which we just added)
    cols_to_remove = [c for c in raw_datasets['train'].column_names if c != 'label']
    tokenized_datasets = tokenized_datasets.remove_columns(cols_to_remove)

    # 4. Initialize Model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint,
        num_labels=len(FALLACY_TYPES),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    # 5. Training Arguments
    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        eval_accumulation_steps=8,
    )

    # 6. Trainer
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # 7. Fine-tune
    print("Starting training...")
    trainer.train()

    # 8. Evaluation on Test Set
    print("\nEvaluating on Test Set...")
    test_results = trainer.evaluate(tokenized_datasets["test"])

    print(f"\nFinal Test Results:")
    print(f"Macro Precision: {test_results['eval_precision']:.4f}")
    print(f"Macro Recall:    {test_results['eval_recall']:.4f}")
    print(f"Macro F1 Score:  {test_results['eval_f1']:.4f}")
    print(f"Accuracy:        {test_results['eval_accuracy']:.4f}")

    # Print a per-class breakdown so you can see which fallacy types are hardest to classify
    test_output = trainer.predict(tokenized_datasets["test"])
    predictions = np.argmax(test_output.predictions, axis=-1)
    print("\nPer-class breakdown:")
    print(classification_report(
        test_output.label_ids, predictions,
        target_names=FALLACY_TYPES,
        zero_division=0,
    ))

    # 9. Save the best model explicitly
    trainer.save_model("./final_model")
    tokenizer.save_pretrained("./final_model")
    print("\nBest model saved to ./final_model")


if __name__ == "__main__":
    main()