import pandas as pd

df = pd.read_csv('data/topSongsLyrics1950_2019.csv') # wczytanie danych z pliku CSV do DataFrame

EXCLUDED_ROWS = [
    1,
    24, 28,
    55, 56, 69, 71, 86, 88, 94,
    102, 120, 140, 153, 156, 157, 189,
    245, 267, 275, 292, 298,
    313, 348, 362,
    396, 397,
    433, 463, 477, 481,
    577, 578, 595,
    605, 608, 617, 620, 630, 634, 636, 639, 640, 649,
    660, 662, 665, 666, 672, 673, 676, 682, 686
]

df = df.drop(EXCLUDED_ROWS) # usunięcie wierszy z niepoprawnymi lub nieodpowiednimi danymi

df["lyrics"] = (
    df["lyrics"]
    .fillna("") # wypełnienie brakujących wartości pustym stringiem
    .astype("string") # konwersja kolumny z tekstami utworów do typu string
    .str.replace("|", " ", regex=False) # zamiana symbolu "|" na spację
    .str.replace("You might also like", " ", regex=False) # usunięcie artefaktu pochodzącego ze strony źródłowej
    .str.replace(r"\s+", " ", regex=True) # usunięcie nadmiarowych spacji
    .str.strip() # usunięcie spacji z początku i końca
)