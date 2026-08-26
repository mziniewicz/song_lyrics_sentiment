import numpy as np
import re
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from hdbscan import HDBSCAN
from umap import UMAP

def remove_repeated_words(text):
    return re.sub(
        r"\b(\w+)(?:\s+\1\b)+",
        r"\1",
        text,
        flags=re.IGNORECASE
    )

class TopicAnalyzer:

    def __init__(
        self,
        min_cluster_size=20,
        min_samples=1,
        random_state=42,
        batch_size=32,
        cluster_selection_method="eom"
    ):

        # Model do tworzenia embeddingów tekstu
        self.embedding_model = SentenceTransformer(
            "all-mpnet-base-v2"
        )

        self.tokenizer = self.embedding_model.tokenizer
        self.max_seq_length = self.embedding_model.max_seq_length
        self.batch_size = batch_size

        special_tokens = self.tokenizer.num_special_tokens_to_add(
            pair=False
        )

        self.chunk_size = (
            self.max_seq_length - special_tokens
        )

        custom_stopwords = list(ENGLISH_STOP_WORDS) + [
            "oh", "ooh", "yeah", "ah", "uh",
            "hey", "la", "na", "whoa", "woah",
            "woo", "ha", "doo", "dum", "bum",
            "sha", "mmm", "mm", "ohh", "da",
            "bop", "lo", "umm", "nah", "hoo",
            "ayy", "di", "yah", "ba", "ya", "ha", "eh",

            # Fragmenty kontrakcji
            "don", "doesn", "didn", "isn", "aren",
            "wasn", "weren", "won", "wouldn",
            "shouldn", "couldn", "haven", "hasn",
            "hadn", "mustn", "mightn", "needn",
            "shan", "ain", "ll", "ve", "re"
        ]

        vectorizer_model = CountVectorizer(
            stop_words=custom_stopwords,
            preprocessor=remove_repeated_words,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.5
        )

        ctfidf_model = ClassTfidfTransformer(
            reduce_frequent_words=True
        )

        # Reprodukowalna redukcja wymiarów
        umap_model = UMAP(
            n_neighbors=15,
            n_components=5,
            min_dist=0.0,
            metric="cosine",
            random_state=random_state
        )

        # Klasteryzacja
        hdbscan_model = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="euclidean",
            cluster_selection_method=cluster_selection_method,
            prediction_data=True
        )

        # Główny model BERTopic
        self.model = BERTopic(
            embedding_model=self.embedding_model,
            vectorizer_model=vectorizer_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            ctfidf_model=ctfidf_model,
            calculate_probabilities=True,
            verbose=True,
            top_n_words=20
        )

        self.document_embeddings_ = None

    def _split_into_chunks(self, text):

        token_ids = self.tokenizer.encode(
            text,
            add_special_tokens=False,
            truncation=False,
            verbose=False
        )

        if len(token_ids) == 0:
            return [], []

        chunks = []
        chunk_lengths = []

        for start in range(
            0,
            len(token_ids),
            self.chunk_size
        ):

            chunk_ids = token_ids[
                start:start + self.chunk_size
            ]

            chunk = self.tokenizer.decode(
                chunk_ids,
                skip_special_tokens=True
            )

            chunks.append(chunk)
            chunk_lengths.append(len(chunk_ids))

        return chunks, chunk_lengths

    def create_embeddings(self, texts):

        texts = list(texts)

        all_chunks = []
        document_chunk_lengths = []

        for index, text in enumerate(texts):

            if not isinstance(text, str):
                text = str(text)

            chunks, lengths = self._split_into_chunks(text)

            if not chunks:
                raise ValueError(
                    f"Empty text at position {index}."
                )

            all_chunks.extend(chunks)
            document_chunk_lengths.append(lengths)

        # Tworzenie embeddingów wszystkich fragmentów
        chunk_embeddings = self.embedding_model.encode(
            all_chunks,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        document_embeddings = []

        position = 0

        # Łączenie embeddingów fragmentów do jednego embeddingu utworu
        for lengths in document_chunk_lengths:

            number_of_chunks = len(lengths)

            embeddings = chunk_embeddings[
                position:position + number_of_chunks
            ]

            weights = np.asarray(
                lengths,
                dtype=np.float32
            )

            document_embedding = np.average(
                embeddings,
                axis=0,
                weights=weights
            )

            # Normalizacja embeddingu całego dokumentu
            norm = np.linalg.norm(document_embedding)

            if norm > 0:
                document_embedding = (
                    document_embedding / norm
                )

            document_embeddings.append(
                document_embedding
            )

            position += number_of_chunks

        return np.vstack(document_embeddings)


    # Trenowanie modelu na tekstach
    def fit(self, texts):

        texts = list(texts)

        embeddings = self.create_embeddings(texts)

        self.document_embeddings_ = embeddings

        topics, probabilities = self.model.fit_transform(
            texts,
            embeddings=embeddings
        )

        return topics, probabilities

    # Przypisanie tematów nowym tekstom
    def transform(self, texts):

        texts = list(texts)

        embeddings = self.create_embeddings(texts)

        topics, probabilities = self.model.transform(
            texts,
            embeddings=embeddings
        )

        return topics, probabilities

    # Tabela wszystkich tematów
    def get_topic_info(self):

        return self.model.get_topic_info()

    # Słowa opisujące konkretny temat
    def get_topic(self, topic_id):

        return self.model.get_topic(topic_id)

    # Wizualizacja tematów
    def visualize_topics(self):

        return self.model.visualize_topics()

    # Wizualizacja słów w tematach
    def visualize_barchart(self):

        return self.model.visualize_barchart()

    # Zapis modelu
    def save(self, path):

        self.model.save(path)

    # Wczytanie wcześniej zapisanego modelu
    @classmethod
    def load(cls, path):

        analyzer = cls()

        analyzer.model = BERTopic.load(
            path,
            embedding_model=analyzer.embedding_model
        )

        return analyzer