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