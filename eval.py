from transformers import Trainer
from dataset import load_and_preprocess_data, split_and_tokenize_data
from model import load_model_and_tokenizer
from utils import compute_metrics

def main():
    # File path to the dataset
    file_path = "Bangla-News-Headlines.csv"
    checkpoint = "bangla-bert-news-headline"

    # Load and preprocess data
    df, label_encoder = load_and_preprocess_data(file_path)

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(checkpoint, num_labels=len(label_encoder.classes_))

    # Tokenize and split datasets
    tokenized_datasets = split_and_tokenize_data(df, tokenizer)

    # Evaluate the model
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    results = trainer.evaluate(eval_dataset=tokenized_datasets["test"])
    print("Evaluation Results:", results)

if __name__ == "__main__":
    main()
