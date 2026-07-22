from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

from hdbscan import HDBSCAN

class TopicAnalyzer:

    def __init__(self):

        # Model do tworzenia embeddingów tekstu
        self.embedding_model = SentenceTransformer(
            "all-mpnet-base-v2"
        )

        custom_stopwords = list(ENGLISH_STOP_WORDS) + [
            "oh", "ooh", "yeah", "ah", "uh",
            "hey", "la", "na", "whoa", "woah",
            "woo", "ha", "doo", "dum", "bum",
            "sha", "mmm", "mm", "ohh", "da",
            "bop", "lo", "umm", "nah", "hoo"
        ]

        vectorizer_model = CountVectorizer(
            stop_words=custom_stopwords,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.5
        )

        hdbscan_model = HDBSCAN(
            min_cluster_size=20,
            min_samples=1,
            metric="euclidean",
            prediction_data=True
        )

        # Główny model BERTopic
        self.model = BERTopic(
            embedding_model=self.embedding_model,
            vectorizer_model=vectorizer_model,
            hdbscan_model=hdbscan_model,
            calculate_probabilities=True,
            min_topic_size=15,
            verbose=True
        )

    # Trenowanie modelu na tekstach
    def fit(self, texts):

        topics, probabilities = self.model.fit_transform(texts)

        return topics, probabilities

    # Przypisanie tematów nowym tekstom
    def transform(self, texts):

        topics, probabilities = self.model.transform(texts)

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
        analyzer.model = BERTopic.load(path)

        return analyzer