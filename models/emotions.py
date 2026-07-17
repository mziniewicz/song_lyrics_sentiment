from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from data import split_into_chunks

EMOTION_MODEL = "j-hartmann/emotion-english-distilroberta-base"

emotion_tokenizer = AutoTokenizer.from_pretrained(EMOTION_MODEL)
emotion_model = AutoModelForSequenceClassification.from_pretrained(EMOTION_MODEL)

def emotion(text):

    chunks = split_into_chunks(
        text,
        emotion_tokenizer,
        max_tokens=510,
        overlap=50
    )

    labels = emotion_model.config.id2label

    all_probs = []

    for chunk in chunks:

        inputs = emotion_tokenizer(
            chunk,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        with torch.no_grad():
            outputs = emotion_model(**inputs)

        probs = torch.softmax(outputs.logits, dim=1)[0]

        all_probs.append(probs)

    mean_probs = torch.stack(all_probs).mean(dim=0)

    return {
        labels[i]: mean_probs[i].item()
        for i in range(len(labels))
    }