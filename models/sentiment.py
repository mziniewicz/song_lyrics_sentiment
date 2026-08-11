import numpy as np
import torch
from transformers import AutoTokenizer,AutoModelForSequenceClassification

SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"

# Tokenizer i model
tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL)

model = AutoModelForSequenceClassification.from_pretrained(
    SENTIMENT_MODEL
)

model.eval()

# Maksymalna długość wejścia modelu
MAX_LENGTH = 512

# Identyfikatory klas pobierane z konfiguracji modelu
label_to_id = {
    label.lower(): int(label_id)
    for label_id, label in model.config.id2label.items()
}

NEGATIVE_ID = label_to_id["negative"]
NEUTRAL_ID = label_to_id["neutral"]
POSITIVE_ID = label_to_id["positive"]

def _tokenize_text(text):

    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
        return_overflowing_tokens=True,
        return_special_tokens_mask=True,
        padding=True,
        stride=0
    )

    # Liczba rzeczywistych tokenów tekstu w każdym fragmencie, bez paddingu i tokenów specjalnych
    content_lengths = (
        encoded["attention_mask"].bool()
        & ~encoded["special_tokens_mask"].bool()
    ).sum(dim=1).numpy()

    # Do modelu przekazujemy tylko wymagane wejścia
    model_inputs = {
        key: value
        for key, value in encoded.items()
        if key in tokenizer.model_input_names
    }

    return model_inputs, content_lengths


def sentiment(text):

    model_inputs, chunk_lengths = _tokenize_text(text)

    with torch.inference_mode():
        outputs = model(**model_inputs)

    probabilities = torch.softmax(
        outputs.logits,
        dim=-1
    ).numpy()

    # Średnia ważona długością fragmentów
    averaged_probabilities = np.average(
        probabilities,
        axis=0,
        weights=chunk_lengths
    )

    negative = averaged_probabilities[NEGATIVE_ID]
    neutral = averaged_probabilities[NEUTRAL_ID]
    positive = averaged_probabilities[POSITIVE_ID]

    score = positive - negative

    return {
        "negative": float(negative),
        "neutral": float(neutral),
        "positive": float(positive),
        "sentiment": float(score)
    }