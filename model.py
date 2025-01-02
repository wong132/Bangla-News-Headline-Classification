from transformers import AutoTokenizer, AutoModelForSequenceClassification

def load_model_and_tokenizer(checkpoint, num_labels):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=num_labels)
    return model, tokenizer