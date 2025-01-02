import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset, DatasetDict
import bkit

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)

    # Normalize text
    normalizer = bkit.transform.Normalizer(
        normalize_characters=True,
        normalize_zw_characters=True,
        normalize_halant=True,
        normalize_vowel_kar=True,
        normalize_punctuation_spaces=True
    )
    df['Headline'] = df['Headline'].apply(normalizer)

    # Encode labels
    label_encoder = LabelEncoder()
    df['NewsType'] = label_encoder.fit_transform(df['NewsType'])

    return df, label_encoder

def split_and_tokenize_data(df, tokenizer, max_length=128):
    train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df['NewsType'], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['NewsType'], random_state=42)

    datasets = DatasetDict({
        "train": Dataset.from_pandas(train_df),
        "validation": Dataset.from_pandas(val_df),
        "test": Dataset.from_pandas(test_df)
    })

    def tokenize_function(example):
        return tokenizer(example["Headline"], padding="max_length", truncation=True, max_length=max_length)

    tokenized_datasets = datasets.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.rename_column("NewsType", "labels")
    tokenized_datasets = tokenized_datasets.remove_columns(["Headline", "__index_level_0__"])
    tokenized_datasets.set_format("torch")

    return tokenized_datasets
