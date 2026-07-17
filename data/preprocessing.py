import pandas as pd

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

def split_into_chunks(text, tokenizer,
                      max_tokens=510,
                      overlap=50):

    token_ids = tokenizer.encode(
        text,
        add_special_tokens=False
    )

    chunks = []

    step = max_tokens - overlap

    for i in range(0, len(token_ids), step):

        chunk = token_ids[i:i + max_tokens]

        if len(chunk) == 0:
            break

        chunks.append(
            tokenizer.decode(
                chunk,
                skip_special_tokens=True
            )
        )

        if i + max_tokens >= len(token_ids):
            break

    return chunks