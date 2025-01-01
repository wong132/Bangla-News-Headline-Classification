from transformers import TrainingArguments, Trainer
from dataset import load_and_preprocess_data, split_and_tokenize_data
from model import load_model_and_tokenizer
from utils import compute_metrics

def main():
    # File path to the dataset
    file_path = "Bangla-News-Headlines.csv"
    checkpoint = "csebuetnlp/banglabert"

    # Load and preprocess data
    df, label_encoder = load_and_preprocess_data(file_path)

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(checkpoint, num_labels=len(label_encoder.classes_))

    # Tokenize and split datasets
    tokenized_datasets = split_and_tokenize_data(df, tokenizer)

    # Define training arguments
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
        metric_for_best_model="accuracy",
        push_to_hub=False,
        report_to="none",
    )

    # Train the model
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()

    trainer.save_model("bangla-bert-news-headline")

if __name__ == "__main__":
    main()
