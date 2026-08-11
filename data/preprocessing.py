import pandas as pd
import re

df = pd.read_csv('data/topSongsLyrics1950_2019.csv') # wczytanie danych z pliku CSV do DataFrame

df = df.drop([24, 28, 88, 120, 140, 245, 275, 313, 329, 362, 617, 620, 634, 639, 640, 660, 662, 665, 666, 672, 676]) # usunięcie wierszy z brakującymi lub niepoprawnymi danymi

df["lyrics"] = (
    df["lyrics"]
    .fillna("") # wypełnienie brakujących wartości pustym stringiem
    .astype("string") # konwersja kolumny z tekstami utworów do typu string
)

df["lyrics"] = (
    df["lyrics"]
    .str.replace("|", " ", regex=False)      # zamiana symbolu "|" na spację
    .str.replace(r"\s+", " ", regex=True)    # usunięcię nadmiarowych spacji
    .str.strip()                             # usunięcie spacji z początku i końca
)

# funkcja dzieląca tekst na mniejsze fragmenty o określonej liczbie tokenów

def split_into_chunks(text, tokenizer, max_tokens=510):
    token_ids = tokenizer.encode(
        text,
        add_special_tokens=False
    )

    if not token_ids:
        return []

    chunks = []

    for start in range(0, len(token_ids), max_tokens):
        chunk_ids = token_ids[start:start + max_tokens]

        chunk = tokenizer.decode(
            chunk_ids,
            skip_special_tokens=True
        )

        chunks.append(chunk)

    return chunks