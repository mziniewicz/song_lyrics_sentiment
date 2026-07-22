from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from data import split_into_chunks

SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"

tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL)
model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL)

def sentiment(text):

    # tekst poniżej 510 tokenów
    if len(tokenizer.tokenize(text)) <= 510:

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        with torch.no_grad():
            outputs = model(**inputs)

        probs = torch.softmax(outputs.logits, dim=1)

        return probs[0][2].item() - probs[0][0].item()

    # tekst powyżej 510 tokenów
    chunks = split_into_chunks(text, tokenizer)

    scores = []

    for chunk in chunks:

        inputs = tokenizer(
            chunk,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        with torch.no_grad():
            outputs = model(**inputs)

        probs = torch.softmax(outputs.logits, dim=1)

        scores.append(
            probs[0][2].item() - probs[0][0].item()
        )

    return sum(scores) / len(scores)