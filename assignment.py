# Import necessary libraries
import json
import pandas as pd
import os
from pathlib import Path
import logging
from typing import List, Dict, Tuple, Optional
import warnings
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import re
from nltk.tokenize import word_tokenize
from unidecode import unidecode
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score, classification_report
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.multioutput import MultiOutputClassifier

# Suppress warnings
warnings.filterwarnings('ignore')

class ConfigManager:
    """Configuration management class."""
    
    def __init__(self, config_path: Path = Path("config") / "config.json"):
        self.config_path = config_path
        self.config = None
    
    def load_config(self) -> Dict:
        """Load configuration from file."""
        with open(self.config_path, "r") as f:
            self.config = json.load(f)
        return self.config

class DataManager:
    """Data management class for loading and saving data."""
    
    def __init__(self, data_path: Path = Path("data") / "data.csv", config: Dict = None):
        self.data_path = data_path
        self.config = config
    
    def load_data(self) -> pd.DataFrame:
        """Load data from CSV file."""
        df = pd.read_csv(self.data_path)
        return df
    
    def save_data(self, df: pd.DataFrame, output_path: Path) -> None:
        """Save DataFrame to CSV file."""
        df.to_csv(output_path, index=False)
    
    def save_json(self, data: Dict, output_path: Path) -> None:
        """Save dictionary to JSON file."""
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

class LoggingManager:
    """Logging management class."""
    
    def __init__(self, log_path: Path = Path("analysis") / "logs" / "logs.log"):
        self.log_path = log_path
        self.logger = None
        self._setup_logging()
    
    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        if self.log_path.exists():
            self.log_path.unlink()
        logging.basicConfig(
            level=logging.INFO, 
            filename=self.log_path,
            filemode="w", 
            format="%(levelname)s - %(message)s"
        )
        self.logger = logging.getLogger(__name__)
    
    def get_logger(self) -> logging.Logger:
        """Get logger instance."""
        return self.logger

class NLTKManager:
    """NLTK resource management class."""
    
    @staticmethod
    def download_required_resources() -> None:
        """Download required NLTK resources."""
        resources = [
            ('tokenizers/punkt', 'punkt'),
            ('corpora/stopwords', 'stopwords'),
            ('corpora/wordnet', 'wordnet'),
            ('vader_lexicon', 'vader_lexicon')
        ]
        print("=" * 100)
        for resource_path, resource_name in resources:
            try:
                nltk.data.find(resource_path)
            except LookupError:
                nltk.download(resource_name)

class DataSampler:
    """Class for sampling texts from the dataset by groups."""
    
    def __init__(self, config: Dict, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.n_samples = config['sampling']['n_samples']
        self.group_names = config['group_name']
    
    def sample_group_data(self, df: pd.DataFrame, group_name: str) -> pd.DataFrame:
        """Sample n_samples texts from a specific group."""
        if group_name not in df.columns:
            self.logger.warning(f"Group {group_name} not found in data columns.")
            return pd.DataFrame()
        
        group_df = df[df[group_name].notna()].copy()
        
        if len(group_df) == 0:
            self.logger.warning(f"No data found for group {group_name}.")
            return pd.DataFrame()
        
        if self.n_samples == 'all':
            sampled_df = group_df.copy()
            self.logger.info(f"Sampling all {len(sampled_df)} texts from group {group_name}.")
        else:
            try:
                n_samples = int(self.n_samples)
                if n_samples > len(group_df):
                    self.logger.warning(f"Requested {n_samples} samples for {group_name}, but has only {len(group_df)} texts. Using all texts.")
                    sampled_df = group_df.copy()
                else:
                    sampled_df = group_df.sample(n=n_samples, random_state=self.config.get('random_state', 42))
                    self.logger.info(f"Sampled {n_samples} texts from group {group_name}.")
            except ValueError:
                self.logger.error(f"Invalid n_samples value: {self.n_samples}. Using all texts for {group_name}.")
                sampled_df = group_df.copy()
        
        return sampled_df
    
    def split_non_overlapping_groups(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Split data into non-overlapping groups based on tagging columns."""
        self.logger.info("SPLITTING DATA INTO NON-OVERLAPPING GROUPS")
        self.logger.info(f"Starting with {len(df)} total samples")
        self.logger.info(f"Configured to sample {self.n_samples} samples per group")
        
        groups = {}
        remaining_df = df.copy()
        
        for group_name in self.group_names:
            if group_name in df.columns:
                group_data = remaining_df[remaining_df[group_name].notna()]
                if len(group_data) > 0:
                    self.logger.info(f"Group {group_name}: {len(group_data)} available samples")
                    self.logger.info("-" * 100)
                    sampled_group = self.sample_group_data(group_data, group_name)
                    if not sampled_group.empty:
                        groups[group_name] = sampled_group
                        remaining_df = remaining_df.drop(sampled_group.index)
                        self.logger.info(f"Group {group_name}: sampled {len(sampled_group)} samples")
                        self.logger.info("-" * 100)
                    else:
                        self.logger.warning(f"Group {group_name}: no samples after sampling")
                else:
                    self.logger.warning(f"Group {group_name}: no available samples")
            else:
                self.logger.warning(f"Group {group_name}: column not found in data")
        
        total_sampled = sum(len(group_df) for group_df in groups.values())
        self.logger.info("=" * 100)
        self.logger.info(f"SAMPLING COMPLETE: {total_sampled} total samples from {len(groups)} groups")
        self.logger.info("=" * 100)
        
        return groups

class DataAnalyzer:
    """Data analyzer class."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def get_data_stats(self, df: pd.DataFrame) -> None:
        """Log data statistics for all columns except 'text'."""
        for col in df.columns:
            if col == "text":
                continue
            self.logger.info(f"Column: {col}")
            self.logger.info(f"Unique values: {df[col].unique()}")
            self.logger.info(f"Value counts: {df[col].value_counts()}")
            self.logger.info(f"Missing values: {df[col].isnull().sum()}")
            self.logger.info("-" * 100 + '\n\n')
    
    def analyze_text_stats(self, texts: List[str]) -> None:
        """Analyze and log text statistics."""
        if isinstance(texts, str):
            texts = [texts]
        texts_lengths = []
        texts_vocab = []
        for text in texts:
            if pd.isna(text) or not isinstance(text, str):
                continue
            words = text.split()
            texts_lengths.append(len(words))
            texts_vocab.append(words)
        texts_vocab = set(word for sublist in texts_vocab for word in sublist)
        texts_lengths = pd.DataFrame(texts_lengths, columns=['texts_length'])
        self.logger.info("Texts Sentence Length Statistics:")
        self.logger.info(texts_lengths.describe().to_dict())
        self.logger.info(f"Texts Vocabulary Statistics: {len(texts_vocab)}")

class TextPreprocessor:
    """Text preprocessing class with configurable options."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.lemmatizer = WordNetLemmatizer() if self.config['preprocessing']['lemmatization'] else None
        self.stop_words = set(stopwords.words('english')) if self.config['preprocessing']['remove_stopwords'] else set()
    
    def remove_diacritics(self, text: str) -> str:
        """Remove diacritics from text."""
        return unidecode(text)
        
    def to_lowercase(self, text: str) -> str:
        """Convert text to lowercase."""
        return text.lower() if self.config['preprocessing']['lowercase'] else text
    
    def remove_punctuation(self, text: str) -> str:
        """Remove punctuation from text."""
        if self.config['preprocessing']['remove_punctuation']:
            return text.translate(str.maketrans('', '', string.punctuation))
        return text
    
    def apply_lemmatization(self, text: str) -> str:
        """Apply lemmatization to text."""
        if self.config['preprocessing']['lemmatization'] and self.lemmatizer:
            tokens = word_tokenize(text)
            return ' '.join([self.lemmatizer.lemmatize(token) for token in tokens])
        return text
    
    def remove_stopwords(self, text: str) -> str:
        """Remove stopwords from text."""
        if self.config['preprocessing']['remove_stopwords']:
            tokens = word_tokenize(text)
            return ' '.join([token for token in tokens if token.lower() not in self.stop_words])
        return text
    
    def preprocess_text(self, text: str) -> str:
        """Apply all preprocessing steps to text."""
        if pd.isna(text) or not isinstance(text, str):
            return ""
        text = self.remove_diacritics(text)
        text = self.to_lowercase(text)
        text = self.remove_punctuation(text)
        text = self.remove_stopwords(text)
        text = self.apply_lemmatization(text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def process_dataframe(self, df: pd.DataFrame, text_column: str = 'text', 
                         output_column: str = 'processed_text') -> pd.DataFrame:
        """Process text in DataFrame and save to new column."""
        df_copy = df.copy()
        df_copy[output_column] = df_copy[text_column].apply(self.preprocess_text)
        return df_copy

class FeatureEngineer:
    """Feature engineering class for TF-IDF vectorization."""
    
    def __init__(self, config: Dict, logger: logging.Logger):
        self.config = config['feature_engineering']
        self.logger = logger
        self.vectorizer = None
        self.feature_matrix = None
        self.feature_names = None
        
    def _normalize_ngram_range(self, ngram_range: List[int]) -> Tuple[int, int]:
        """Normalize ngram_range to always return a tuple with 2 elements."""
        if len(ngram_range) == 1:
            return (ngram_range[0], ngram_range[0])
        elif len(ngram_range) == 2:
            return (ngram_range[0], ngram_range[1])
        else:
            return (1, 1)
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """Fit and transform texts to TF-IDF features."""
        normalized_ngram_range = self._normalize_ngram_range(self.config['ngram_range'])
        self.vectorizer = TfidfVectorizer(
            max_features=self.config['max_features'],
            ngram_range=normalized_ngram_range,
            min_df=self.config['min_df'],
            max_df=self.config['max_df']
        )
        self.feature_matrix = self.vectorizer.fit_transform(texts)
        # Handle both old and new scikit-learn versions
        try:
            self.feature_names = self.vectorizer.get_feature_names_out()
        except AttributeError:
            self.feature_names = self.vectorizer.get_feature_names()
        joblib.dump(self.vectorizer, self.config['vectorizer_model'])
        self.logger.info(f"Feature matrix shape: {self.feature_matrix.shape}")
        return self.feature_matrix
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """Transform texts to TF-IDF features using fitted vectorizer."""
        if self.vectorizer is None:
            raise ValueError("Vectorizer not fitted. Call fit_transform first.")
        self.feature_matrix = self.vectorizer.transform(texts)
        return self.feature_matrix
    
    def get_feature_dataframe(self) -> pd.DataFrame:
        """Return feature matrix as DataFrame."""
        return pd.DataFrame(
            self.feature_matrix.toarray(),
            columns=self.feature_names
        )

class TextClusterer:
    """Text clustering class using K-means."""
    
    def __init__(self, config: Dict, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.kmeans = None
        self.optimal_clusters = None
        self.plot_path = Path(self.config['analysis_path']) / 'plots'
        self.plots = self.config['clustering']['plots']
        self.cluster_names = self.config['clustering'].get('cluster_names', {})

    def find_optimal_clusters(self, feature_matrix: np.ndarray) -> int:
        """Find optimal number of clusters using elbow method and silhouette score."""
        inertias = []
        silhouette_scores = []
        k_range = range(2, min(self.config['clustering']['max_clusters'], feature_matrix.shape[0]//2))
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=self.config['clustering']['random_state'])
            kmeans.fit(feature_matrix)
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(feature_matrix, kmeans.labels_))
        optimal_k = k_range[np.argmax(silhouette_scores)]
        self.logger.info(f"Optimal number of clusters: {optimal_k}")
        self.logger.info(f"Best silhouette score: {max(silhouette_scores):.3f}")
        if self.plots:
            self.plot_cluster_optimization(k_range, inertias, silhouette_scores, optimal_k)
        return optimal_k

    def plot_cluster_optimization(self, k_range, inertias, silhouette_scores, optimal_k):
        """Create and save cluster optimization plots."""
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        ax1.plot(k_range, inertias, 'bo-')
        ax1.set_xlabel('Number of Clusters (k)')
        ax1.set_ylabel('Inertia')
        ax1.set_title('Elbow Method for Optimal k')
        ax1.grid(True)
        ax2.plot(k_range, silhouette_scores, 'ro-')
        ax2.axvline(x=optimal_k, color='green', linestyle='--', label=f'Optimal k={optimal_k}')
        ax2.set_xlabel('Number of Clusters (k)')
        ax2.set_ylabel('Silhouette Score')
        ax2.set_title('Silhouette Score for Different k')
        ax2.legend()
        ax2.grid(True)
        plt.tight_layout()
        path = self.plot_path / "cluster_optimization.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info("-" * 100)
        self.logger.info(f"Cluster optimization plot saved to {path}")
        self.logger.info("-" * 100)

    def plot_cluster_analysis(self, cluster_labels, feature_names=None, features=None):
        """Create and save cluster analysis plots."""
        if not self.plots:
            return
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 2, 1)
        cluster_counts = Counter(cluster_labels)
        plt.bar(cluster_counts.keys(), cluster_counts.values(), color='skyblue')
        plt.xlabel('Cluster ID')
        plt.ylabel('Number of Cases')
        plt.title('Distribution of Cases Across Clusters')
        plt.grid(True, alpha=0.3)
        if feature_names is not None and features is not None:
            plt.subplot(2, 2, 2)
            largest_cluster = max(cluster_counts, key=cluster_counts.get)
            cluster_indices = [i for i, label in enumerate(cluster_labels) if label == largest_cluster]
            cluster_features = features[cluster_indices]
            n_top_terms = 10
            mean_tfidf = np.mean(cluster_features.toarray(), axis=0)
            top_indices = np.argsort(mean_tfidf)[-n_top_terms:]
            top_terms = [feature_names[i] for i in top_indices]
            top_scores = [mean_tfidf[i] for i in top_indices]
            plt.barh(top_terms, top_scores, color='lightcoral')
            plt.xlabel('Average TF-IDF Score')
            plt.title(f'Top Terms in Largest Cluster ({largest_cluster})')
            plt.gca().invert_yaxis()
        plt.tight_layout()
        path = self.plot_path / "cluster_analysis.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info("-" * 100)
        self.logger.info(f"Cluster analysis plot saved to {path}")
        self.logger.info("-" * 100)

    def get_cluster_name(self, cluster_id: int) -> str:
        """Get the name for a given cluster ID."""
        return self.cluster_names.get(str(cluster_id), f"Cluster {cluster_id}")
    
    def extract_cluster_vocabularies(self, processed_data: List[Dict], print_details: bool = True) -> Dict:
        """Extract and optionally print vocabulary for each cluster."""
        cluster_data = {}
        for item in processed_data:
            cluster_id = item['cluster_id']
            if cluster_id not in cluster_data:
                cluster_data[cluster_id] = []
            cluster_data[cluster_id].append(item)
        cluster_vocabularies = {}
        for cluster_id in sorted(cluster_data.keys()):
            cluster_items = cluster_data[cluster_id]
            cluster_name = self.get_cluster_name(cluster_id)
            all_text = [item['processed_summary'].split() for item in cluster_items]
            vocabulary = sorted(set(word for sublist in all_text for word in sublist))
            cluster_vocabularies[cluster_id] = {
                'cluster_name': cluster_name,
                'vocabulary': vocabulary,
                'vocabulary_size': len(vocabulary),
                'case_count': len(cluster_items)
            }
            if print_details:
                self.logger.info(f"Cluster {cluster_id} - {cluster_name} ({len(cluster_items)} cases):")
                self.logger.info(f"Vocabulary size: {len(vocabulary)}")
                self.logger.info(f"Vocabulary: {vocabulary}")
                self.logger.info("-" * 80)
        if print_details:
            self.logger.info(f"Total clusters: {len(cluster_data)}")
        return cluster_vocabularies
    
    def fit_predict(self, feature_matrix: np.ndarray, texts: List[str], feature_names: np.ndarray = None) -> np.ndarray:
        """Fit clustering model and predict clusters."""
        self.optimal_clusters = self.find_optimal_clusters(feature_matrix)
        self.kmeans = KMeans(
            n_clusters=self.optimal_clusters,
            random_state=self.config['clustering']['random_state']
        )
        clusters = self.kmeans.fit_predict(feature_matrix)
        joblib.dump(self.kmeans, self.config['clustering']['model_name'])
        self.plot_cluster_analysis(clusters, feature_names, feature_matrix)
        return clusters
    
    def predict_clusters(self, feature_matrix: np.ndarray) -> np.ndarray:
        """Predict clusters for new texts using fitted model."""
        if self.kmeans is None:
            raise ValueError("Model not fitted. Call fit_predict first.")
        return self.kmeans.predict(feature_matrix)
    
    def predict_single_text_cluster(self, text: str, feature_engineer: 'FeatureEngineer', 
                                   text_preprocessor: 'TextPreprocessor') -> int:
        """Predict cluster for a single text."""
        if isinstance(text, str):
            texts = [text]
        else:
            texts = text
        
        # Preprocess text
        processed_texts = [text_preprocessor.preprocess_text(t) for t in texts]
        
        # Transform to features
        feature_matrix = feature_engineer.transform(processed_texts)
        
        # Predict cluster
        clusters = self.predict_clusters(feature_matrix)
        
        return clusters[0] if len(clusters) == 1 else clusters
    
    def predict_text_clusters(self, texts: List[str], feature_engineer: 'FeatureEngineer', 
                             text_preprocessor: 'TextPreprocessor') -> List[int]:
        """Predict clusters for multiple texts."""
        if isinstance(texts, str):
            return [self.predict_single_text_cluster(texts, feature_engineer, text_preprocessor)]
        
        # Preprocess texts
        processed_texts = [text_preprocessor.preprocess_text(text) for text in texts]
        
        # Transform to features
        feature_matrix = feature_engineer.transform(processed_texts)
        
        # Predict clusters
        clusters = self.predict_clusters(feature_matrix)
        
        return clusters.tolist()

class FeaturePreparationMixin:
    """Mixin class for common feature preparation functionality."""
    
    def prepare_features_for_prediction(self, texts: List[str]) -> np.ndarray:
        """Prepare features for prediction."""
        if isinstance(texts, str):
            texts = [texts]
        
        # Preprocess texts
        processed_texts = [self.text_preprocessor.preprocess_text(text) for text in texts]
        
        # Transform to TF-IDF features
        feature_matrix = self.feature_engineer.transform(processed_texts)
        
        # Get cluster predictions
        clusters = self.text_clusterer.predict_text_clusters(texts, self.feature_engineer, self.text_preprocessor)
        cluster_features = np.array(clusters).reshape(-1, 1)
        
        # Combine features
        combined_features = np.hstack([feature_matrix.toarray(), cluster_features])
        
        return combined_features

class TextClassificationPipeline(FeaturePreparationMixin):
    """Pipeline for text classification with logistic regression and clustering."""
    
    def __init__(self, config: Dict, logger: logging.Logger, feature_engineer: FeatureEngineer, 
                 text_clusterer: TextClusterer, text_preprocessor: TextPreprocessor, group_name: str):
        self.config = config
        self.logger = logger
        self.feature_engineer = feature_engineer
        self.text_clusterer = text_clusterer
        self.text_preprocessor = text_preprocessor
        self.group_name = group_name
        self.model_path = Path(self.config['classification']['model_path']) / f"{group_name}_logistic_model.joblib"
        self.model = None
        self.train_indices = None
        self.test_indices = None
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Transform text to TF-IDF features and include cluster IDs."""
        # Transform text to TF-IDF
        feature_matrix = self.feature_engineer.transform(df['processed_text'].tolist())
        # Add cluster IDs as features
        cluster_features = df[['main_cluster']].to_numpy()
        combined_features = np.hstack([feature_matrix.toarray(), cluster_features])
        # Get target variable
        target_col = self.group_name
        y = df[target_col].to_numpy()
        return combined_features, y
    
    def split_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data into train and test sets with stratification when possible."""
        # Check if stratification is possible (all classes must have at least 2 samples)
        unique_classes, class_counts = np.unique(y, return_counts=True)
        min_class_count = np.min(class_counts)
        
        if min_class_count < 2:
            self.logger.warning(f"Stratification not possible for {self.group_name} - some classes have only {min_class_count} sample(s). Using random split.")
            # Use random split without stratification
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=self.config['classification']['test_size'],
                random_state=self.config['classification']['random_state']
            )
        else:
            # Use stratified split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=self.config['classification']['test_size'],
                stratify=y,
                random_state=self.config['classification']['random_state']
            )
        
        self.logger.info(f"Train set size for {self.group_name}: {len(X_train)}")
        self.logger.info(f"Test set size for {self.group_name}: {len(X_test)}")
        self.logger.info(f"Class distribution in dataset for {self.group_name}: {dict(zip(unique_classes, class_counts))}")
        return X_train, X_test, y_train, y_test
    
    def compute_class_weights(self, y: np.ndarray) -> Dict:
        """Compute inverse frequency class weights."""
        classes = np.unique(y)
        class_weights = compute_class_weight('balanced', classes=classes, y=y)
        return dict(zip(classes, class_weights))
    
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train logistic regression model with cross-validation."""
        class_weights = self.compute_class_weights(y_train)
        self.model = LogisticRegression(
            class_weight=class_weights,
            random_state=self.config['classification']['random_state'],
            max_iter=1000
        )
        # Perform cross-validation
        cv_scores = cross_val_score(
            self.model, X_train, y_train,
            cv=self.config['classification']['cv_folds'],
            scoring='f1_weighted'
        )
        self.logger.info(f"Cross-validation scores for {self.group_name}: {cv_scores}")
        self.logger.info(f"Mean CV score for {self.group_name}: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")
        # Train on full training data
        self.model.fit(X_train, y_train)
        joblib.dump(self.model, self.model_path)
        self.logger.info(f"Model for {self.group_name} saved to {self.model_path}")
    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> None:
        """Evaluate model and log classification report."""
        y_pred = self.model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=False)
        self.logger.info(f"Classification report for {self.group_name} on test set:")
        self.logger.info("\n" + report)
    
    def predict_probabilities(self, texts: List[str]) -> np.ndarray:
        """Predict class probabilities for new texts."""
        if self.model is None:
            raise ValueError("Model not trained. Call run() first.")
        
        X = self.prepare_features_for_prediction(texts)
        return self.model.predict_proba(X)
    
    def predict_classes(self, texts: List[str]) -> np.ndarray:
        """Predict classes for new texts."""
        if self.model is None:
            raise ValueError("Model not trained. Call run() first.")
        
        X = self.prepare_features_for_prediction(texts)
        return self.model.predict(X)
    
    def predict_single_text(self, text: str) -> Dict:
        """Predict class and probability for a single text."""
        probabilities = self.predict_probabilities([text])
        prediction = self.predict_classes([text])
        cluster = self.text_clusterer.predict_single_text_cluster(text, self.feature_engineer, self.text_preprocessor)
        
        return {
            'predicted_class': prediction[0],
            'probabilities': probabilities[0],
            'cluster': cluster
        }
    
    def run(self, df: pd.DataFrame) -> None:
        """Run the classification pipeline."""
        self.logger.info(f"Starting classification pipeline for {self.group_name}")
        X, y = self.prepare_features(df)
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        self.train_model(X_train, y_train)
        self.evaluate_model(X_test, y_test)
        self.logger.info(f"Completed classification pipeline for {self.group_name}")

class NullValuesFiller(FeaturePreparationMixin):
    """Class to fill null values in target columns using trained models."""
    
    def __init__(self, config: Dict, logger: logging.Logger, feature_engineer: FeatureEngineer,
                 text_clusterer: TextClusterer, text_preprocessor: TextPreprocessor):
        self.config = config
        self.logger = logger
        self.feature_engineer = feature_engineer
        self.text_clusterer = text_clusterer
        self.text_preprocessor = text_preprocessor
        self.confidence_threshold = config.get('null_filling', {}).get('confidence_threshold', 0.6)
        self.null_fill_value = config.get('null_filling', {}).get('null_fill_value', -100)
        self.classifiers = {}
        self.group_names = config.get('group_names', ['is_spam', 'type', 'queue', 'priority'])
    
    def load_trained_models(self) -> None:
        """Load all trained classification models."""
        model_path = Path(self.config['classification']['model_path'])
        
        for group_name in self.group_names:
            model_file = model_path / f"{group_name}_logistic_model.joblib"
            if model_file.exists():
                try:
                    model = joblib.load(model_file)
                    self.classifiers[group_name] = model
                    self.logger.info(f"Loaded model for {group_name}")
                except Exception as e:
                    self.logger.error(f"Failed to load model for {group_name}: {e}")
            else:
                self.logger.warning(f"Model file not found for {group_name}: {model_file}")
    
    def predict_with_confidence(self, texts: List[str], group_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Predict values with confidence scores."""
        if group_name not in self.classifiers:
            raise ValueError(f"No trained model found for {group_name}")
        
        model = self.classifiers[group_name]
        X = self.prepare_features_for_prediction(texts)
        
        # Get predictions and probabilities
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)
        
        # Get confidence scores (max probability)
        confidence_scores = np.max(probabilities, axis=1)
        
        return predictions, confidence_scores
    
    def fill_null_values_for_group(self, df: pd.DataFrame, group_name: str) -> pd.DataFrame:
        """Fill null values for a specific group."""
        if group_name not in df.columns:
            self.logger.warning(f"Column {group_name} not found in dataframe")
            return df
        
        # Find rows with null values
        null_mask = df[group_name].isnull()
        
        if not null_mask.any():
            self.logger.info(f"No null values found in {group_name}")
            return df
        
        # Get texts for null rows
        null_texts = df.loc[null_mask, 'text'].tolist()
        
        if not null_texts:
            return df
        
        # Predict values
        predictions, confidence_scores = self.predict_with_confidence(null_texts, group_name)
        
        # Apply confidence threshold
        high_confidence_mask = confidence_scores >= self.confidence_threshold
        
        # Fill values
        df_copy = df.copy()
        null_indices = df_copy[null_mask].index
        
        for i, idx in enumerate(null_indices):
            if high_confidence_mask[i]:
                df_copy.loc[idx, group_name] = predictions[i]
            else:
                df_copy.loc[idx, group_name] = self.null_fill_value
        
        return df_copy
    
    def fill_all_null_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill null values for all groups."""
        self.load_trained_models()
        
        # Process text if not already processed
        if 'processed_text' not in df.columns:
            df = self.text_preprocessor.process_dataframe(df)
        
        df_filled = df.copy()
        
        for group_name in self.group_names:
            if group_name in self.classifiers:
                self.logger.info(f"Processing null values for {group_name}")
                df_filled = self.fill_null_values_for_group(df_filled, group_name)
            else:
                self.logger.warning(f"No model available for {group_name}")
        
        # Log summary
        for group_name in self.group_names:
            if group_name in df_filled.columns:
                null_count = df_filled[group_name].isnull().sum()
                filled_count = (df_filled[group_name] == self.null_fill_value).sum()
                self.logger.info(f"{group_name}: {null_count} nulls remaining, {filled_count} filled with {self.null_fill_value}")
        
        return df_filled

class MultiLabelMLPPredictor(FeaturePreparationMixin):
    """Multi-label MLP predictor for all target variables simultaneously."""
    
    def __init__(self, config: Dict, logger: logging.Logger, feature_engineer: FeatureEngineer,
                 text_clusterer: TextClusterer, text_preprocessor: TextPreprocessor):
        self.config = config
        self.logger = logger
        self.feature_engineer = feature_engineer
        self.text_clusterer = text_clusterer
        self.text_preprocessor = text_preprocessor
        self.model = None
        self.label_encoders = {}
        self.group_names = config.get('group_names', ['is_spam', 'type', 'queue', 'priority'])
        self.model_path = Path(self.config.get('mlp_model_path', 'models/mlp_models'))
        self.model_path.mkdir(parents=True, exist_ok=True)
        self.model_file = self.model_path / "multilabel_mlp_model.joblib"
        self.encoders_file = self.model_path / "label_encoders.joblib"
    
    def prepare_multilabel_targets(self, df: pd.DataFrame) -> Tuple[np.ndarray, Dict]:
        """Prepare multi-label targets with label encoding."""
        
        # Filter out rows with null values or -100 values for any target
        valid_mask = pd.Series([True] * len(df))
        for group_name in self.group_names:
            if group_name in df.columns:
                valid_mask &= (df[group_name].notna()) & (df[group_name] != -100)
        
        if not valid_mask.any():
            self.logger.warning("No valid data found for multi-label training")
            return np.array([]), {}
        
        valid_df = df[valid_mask].copy()
        
        # Encode each target variable
        encoded_targets = []
        for group_name in self.group_names:
            if group_name in valid_df.columns:
                if group_name not in self.label_encoders:
                    self.label_encoders[group_name] = LabelEncoder()
                    encoded_target = self.label_encoders[group_name].fit_transform(valid_df[group_name])
                else:
                    encoded_target = self.label_encoders[group_name].transform(valid_df[group_name])
                encoded_targets.append(encoded_target)
        
        # Stack targets as columns
        if encoded_targets:
            target_matrix = np.column_stack(encoded_targets)
        else:
            target_matrix = np.array([])
        
        return target_matrix, valid_df
    
    def create_multilabel_model(self, input_size: int, output_size: int) -> MLPClassifier:
        """Create multi-label MLP model."""
        # Configure hidden layers based on input and output sizes
        hidden_layer_sizes = (
            min(512, max(input_size, output_size * 10)),
            min(256, max(input_size // 2, output_size * 5)),
            min(128, max(input_size // 4, output_size * 2))
        )
        
        model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation='relu',
            solver='adam',
            alpha=0.001,
            batch_size='auto',
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=1000,
            shuffle=True,
            random_state=self.config.get('random_state', 42),
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            tol=1e-4
        )
        
        return model
    
    def train_multilabel_model(self, df: pd.DataFrame) -> Dict:
        """Train multi-label MLP model for all targets simultaneously."""
        # Prepare targets
        y, valid_df = self.prepare_multilabel_targets(df)
        
        if y.size == 0:
            self.logger.warning("No valid data for multi-label training")
            return {}
        
        # Prepare features
        X = self.prepare_features_for_prediction(valid_df['text'].tolist())
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.config.get('random_state', 42)
        )
        
        # Create model
        self.model = self.create_multilabel_model(X.shape[1], y.shape[1])
        
        self.logger.info(f"Training multi-label MLP model with {y.shape[1]} targets...")
        self.logger.info(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")
        
        # Train model with multi-output wrapper
        multi_output_model = MultiOutputClassifier(self.model)
        multi_output_model.fit(X_train, y_train)
        
        # Evaluate model on both train and test sets
        y_train_pred = multi_output_model.predict(X_train)
        y_test_pred = multi_output_model.predict(X_test)
        
        # Calculate metrics for each target
        metrics = {}
        for i, group_name in enumerate(self.group_names):
            if i < y_test.shape[1]:
                # Test set metrics
                test_metrics = {
                    'accuracy': accuracy_score(y_test[:, i], y_test_pred[:, i]),
                    'precision': precision_score(y_test[:, i], y_test_pred[:, i], average='weighted', zero_division=0),
                    'recall': recall_score(y_test[:, i], y_test_pred[:, i], average='weighted', zero_division=0),
                    'f1': f1_score(y_test[:, i], y_test_pred[:, i], average='weighted', zero_division=0)
                }
                
                # Train set metrics
                train_metrics = {
                    'accuracy': accuracy_score(y_train[:, i], y_train_pred[:, i]),
                    'precision': precision_score(y_train[:, i], y_train_pred[:, i], average='weighted', zero_division=0),
                    'recall': recall_score(y_train[:, i], y_train_pred[:, i], average='weighted', zero_division=0),
                    'f1': f1_score(y_train[:, i], y_train_pred[:, i], average='weighted', zero_division=0)
                }
                
                metrics[group_name] = {
                    'test': test_metrics,
                    'train': train_metrics
                }
                
                # Log metrics
                self.logger.info(f"Multi-label MLP metrics for {group_name}:")
                self.logger.info(f"  Train metrics:")
                for metric, value in train_metrics.items():
                    self.logger.info(f"    {metric}: {value:.4f}")
                self.logger.info(f"  Test metrics:")
                for metric, value in test_metrics.items():
                    self.logger.info(f"    {metric}: {value:.4f}")
                
                # Generate classification reports
                self.logger.info(f"Classification Report for {group_name} (Train set):")
                try:
                    # Get original class names for better report readability
                    class_names = self.label_encoders[group_name].classes_
                    train_report = classification_report(
                        y_train[:, i], y_train_pred[:, i], 
                        target_names=[str(name) for name in class_names],
                        zero_division=0
                    )
                    self.logger.info(f"\n{train_report}")
                except Exception as e:
                    self.logger.warning(f"Could not generate train classification report for {group_name}: {e}")
                
                self.logger.info(f"Classification Report for {group_name} (Test set):")
                try:
                    test_report = classification_report(
                        y_test[:, i], y_test_pred[:, i], 
                        target_names=[str(name) for name in class_names],
                        zero_division=0
                    )
                    self.logger.info(f"\n{test_report}")
                except Exception as e:
                    self.logger.warning(f"Could not generate test classification report for {group_name}: {e}")
                
                self.logger.info("-" * 80)
        
        # Save model and encoders
        self.model = multi_output_model
        joblib.dump(self.model, self.model_file)
        joblib.dump(self.label_encoders, self.encoders_file)
        
        self.logger.info(f"Multi-label MLP model saved to {self.model_file}")
        self.logger.info(f"Label encoders saved to {self.encoders_file}")
        
        return metrics
    
    def load_trained_model(self) -> None:
        """Load trained multi-label MLP model."""
        if self.model_file.exists() and self.encoders_file.exists():
            try:
                self.model = joblib.load(self.model_file)
                self.label_encoders = joblib.load(self.encoders_file)
                self.logger.info("Loaded multi-label MLP model and encoders")
            except Exception as e:
                self.logger.error(f"Failed to load multi-label MLP model: {e}")
        else:
            self.logger.warning("Multi-label MLP model or encoders not found")
    
    def predict_single_text(self, text: str) -> Dict:
        """Predict all targets for a single text."""
        if self.model is None:
            self.load_trained_model()
        
        if self.model is None:
            raise ValueError("No trained multi-label MLP model available")
        
        X = self.prepare_features_for_prediction([text])
        
        # Get predictions
        predictions = self.model.predict(X)[0]
        
        # Get probabilities for each target
        try:
            probabilities = self.model.predict_proba(X)
        except:
            probabilities = None
        
        # Decode predictions
        results = {}
        for i, group_name in enumerate(self.group_names):
            if i < len(predictions) and group_name in self.label_encoders:
                decoded_prediction = self.label_encoders[group_name].inverse_transform([predictions[i]])[0]
                result = {
                    'predicted_value': decoded_prediction,
                    'encoded_value': predictions[i]
                }
                
                # Add probability if available
                if probabilities is not None and i < len(probabilities):
                    try:
                        prob_array = probabilities[i][0] if isinstance(probabilities[i], list) else probabilities[i]
                        result['probabilities'] = prob_array
                        result['confidence'] = np.max(prob_array)
                    except:
                        result['confidence'] = 0.0
                
                results[group_name] = result
        
        return results
    
    def predict_multiple_texts(self, texts: List[str]) -> List[Dict]:
        """Predict all targets for multiple texts."""
        if self.model is None:
            self.load_trained_model()
        
        if self.model is None:
            raise ValueError("No trained multi-label MLP model available")
        
        X = self.prepare_features_for_prediction(texts)
        predictions = self.model.predict(X)
        
        results = []
        for i, text in enumerate(texts):
            text_results = {'text': text}
            
            # Decode predictions for each target
            for j, group_name in enumerate(self.group_names):
                if j < predictions.shape[1] and group_name in self.label_encoders:
                    decoded_prediction = self.label_encoders[group_name].inverse_transform([predictions[i, j]])[0]
                    text_results[group_name] = {
                        'predicted_value': decoded_prediction,
                        'encoded_value': predictions[i, j]
                    }
            
            results.append(text_results)
        
        return results

class MainPipeline:
    """Main pipeline class that orchestrates text clustering and classification."""
    
    def __init__(self, config_path: Path = Path("config") / "config.json", 
                 data_path: Path = Path("data") / "data.csv"):
        self.config_path = config_path
        self.data_path = data_path
        self.config_manager = ConfigManager(config_path)
        self.data_manager = None
        self.logging_manager = LoggingManager()
        self.nltk_manager = NLTKManager()
        self.logger = self.logging_manager.get_logger()
        self.config = None
        self.data_sampler = None
        self.data_analyzer = None
        self.text_preprocessor = None
        self.feature_engineer = None
        self.text_clusterer = None
        self.null_values_filler = None
        self.mlp_predictor = None
    
    def setup_directories(self) -> None:
        """Setup necessary directories."""
        analysis_path = Path("analysis")
        analysis_path.mkdir(parents=True, exist_ok=True)
        (analysis_path / "logs").mkdir(parents=True, exist_ok=True)
        Path("models").mkdir(parents=True, exist_ok=True)
        
        # Create classification model directory with default if not in config
        classification_config = self.config.get('classification', {})
        model_path = classification_config.get('model_path', 'models/classification')
        Path(model_path).mkdir(parents=True, exist_ok=True)
        
        # Create MLP model directory
        mlp_path = self.config.get('mlp_model_path', 'models/mlp_models')
        Path(mlp_path).mkdir(parents=True, exist_ok=True)
    
    def initialize_components(self) -> None:
        """Initialize all pipeline components."""
        self.config = self.config_manager.load_config()
        
        self.data_manager = DataManager(self.data_path, self.config)
        self.data_sampler = DataSampler(self.config, self.logger)
        self.data_analyzer = DataAnalyzer(self.logger)
        self.text_preprocessor = TextPreprocessor(self.config)
        self.feature_engineer = FeatureEngineer(self.config, self.logger)
        self.text_clusterer = TextClusterer(self.config, self.logger)
        self.null_values_filler = NullValuesFiller(self.config, self.logger, self.feature_engineer, self.text_clusterer, self.text_preprocessor)
        self.mlp_predictor = MultiLabelMLPPredictor(self.config, self.logger, self.feature_engineer, self.text_clusterer, self.text_preprocessor)
    
    def run_data_analysis(self, df: pd.DataFrame) -> None:
        """Run data analysis and logging."""
        self.logger.info("=" * 100)
        self.logger.info("Raw Data stats:")
        self.logger.info(df.info())
        self.logger.info("=" * 100)
        self.data_analyzer.get_data_stats(df)
    
    def run_text_preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run text preprocessing pipeline."""
        self.logger.info("Text Stats Before Preprocessing:")
        self.data_analyzer.analyze_text_stats(df['text'])
        df = self.text_preprocessor.process_dataframe(df)
        self.logger.info("Text Stats After Preprocessing:")
        self.data_analyzer.analyze_text_stats(df['processed_text'])
        return df
    
    def run_feature_engineering(self, df: pd.DataFrame) -> np.ndarray:
        """Run feature engineering pipeline."""
        self.logger.info("=" * 100)
        self.logger.info("Feature Engineering:")
        self.logger.info("=" * 100)
        feature_matrix = self.feature_engineer.fit_transform(df['processed_text'].tolist())
        return feature_matrix
    
    def run_clustering(self, df: pd.DataFrame, feature_matrix: np.ndarray) -> pd.DataFrame:
        """Run clustering pipeline."""
        self.logger.info("=" * 80)
        self.logger.info("Clustering:")
        clusters = self.text_clusterer.fit_predict(feature_matrix, df['processed_text'].tolist(), self.feature_engineer.feature_names)
        df['main_cluster'] = clusters + 1
        return df
    
    def extract_and_save_vocabularies(self, df: pd.DataFrame) -> None:
        """Extract and save cluster vocabularies."""
        processed_data = []
        for _, row in df.iterrows():
            processed_data.append({
                'cluster_id': row['main_cluster'] - 1,
                'processed_summary': row['processed_text']
            })
        cluster_vocabularies = self.text_clusterer.extract_cluster_vocabularies(processed_data, print_details=True)
        vocabulary_output = {}
        for cluster_id, vocab_data in cluster_vocabularies.items():
            vocabulary_output[str(cluster_id + 1)] = {
                'cluster_name': vocab_data['cluster_name'],
                'vocabulary': vocab_data['vocabulary'],
                'vocabulary_size': vocab_data['vocabulary_size'],
                'case_count': vocab_data['case_count']
            }
        vocabulary_file = self.config['clustering']['vocabulary_file']
        self.data_manager.save_json(vocabulary_output, vocabulary_file)
        self.logger.info("=" * 100)
        self.logger.info(f"Cluster vocabularies saved to {vocabulary_file}")
        self.logger.info("=" * 100)
    
    def run_classification(self, group_dfs: Dict[str, pd.DataFrame]) -> None:
        """Run classification pipeline for each group."""
        for group_name, df in group_dfs.items():
            if not df.empty:
                classifier = TextClassificationPipeline(self.config, self.logger, self.feature_engineer, 
                                                      self.text_clusterer, self.text_preprocessor, group_name)
                classifier.run(df)
    
    def save_results(self, df: pd.DataFrame) -> None:
        """Save final results."""
        output_file = self.data_path.parent / "data_with_clusters.csv"
        self.data_manager.save_data(df, output_file)
        self.logger.info("=" * 100)
        self.logger.info(f"Data with clusters saved to {output_file}")
        self.logger.info(f"Total clusters created: {self.text_clusterer.optimal_clusters}")
        self.logger.info("=" * 100)
    
    def log_cluster_distribution(self, df: pd.DataFrame) -> None:
        """Log cluster distribution as a dictionary once."""
        cluster_dist = df['main_cluster'].value_counts().sort_index().to_dict()
        self.logger.info("=" * 100)
        self.logger.info(f"Cluster distribution: {cluster_dist}")
        self.logger.info("=" * 100)
    
    def run(self) -> None:
        """Run the complete text clustering and classification pipeline."""
        self.nltk_manager.download_required_resources()
        self.initialize_components()
        self.setup_directories()
        
        # Load and sample data
        df = self.data_manager.load_data()
        
        # Split into non-overlapping groups
        group_dfs = self.data_sampler.split_non_overlapping_groups(df)
        
        # Combine all sampled data for training TF-IDF and K-means models
        combined_sampled_data = []
        for group_name, group_df in group_dfs.items():
            if not group_df.empty:
                combined_sampled_data.append(group_df)
        
        if not combined_sampled_data:
            self.logger.error("No sampled data available for training.")
            return
            
        # Combine all sampled data into one dataframe
        combined_df = pd.concat(combined_sampled_data, ignore_index=True)
        self.logger.info(f"Combined sampled data: {len(combined_df)} total samples")
        
        # Train TF-IDF vectorizer and K-means model ONCE on combined data
        self.logger.info("Training TF-IDF vectorizer and K-means model on combined sampled data...")
        combined_df = self.run_text_preprocessing(combined_df)
        combined_feature_matrix = self.run_feature_engineering(combined_df)
        combined_df = self.run_clustering(combined_df, combined_feature_matrix)
        
        # Log cluster distribution once at the beginning
        self.log_cluster_distribution(combined_df)
        
        # Extract and save vocabularies from combined data
        self.extract_and_save_vocabularies(combined_df)
        
        # Process each group using the trained models
        processed_group_dfs = {}
        for group_name, group_df in group_dfs.items():
            if group_df.empty:
                self.logger.info(f"No data for {group_name} group. Skipping.")
                processed_group_dfs[group_name] = group_df
                continue
                
            self.logger.info(f"Processing {group_name} group with {len(group_df)} samples using trained models...")
            
            # Preprocess text for this group
            processed_group_df = self.run_text_preprocessing(group_df.copy())
            
            # Transform using trained TF-IDF vectorizer (not fit_transform)
            group_feature_matrix = self.feature_engineer.transform(processed_group_df['processed_text'].tolist())
            
            # Predict clusters using trained K-means model (not fit_predict)
            group_clusters = self.text_clusterer.predict_clusters(group_feature_matrix)
            processed_group_df['main_cluster'] = group_clusters + 1
            
            # Store the processed dataframe
            processed_group_dfs[group_name] = processed_group_df
        
        # Train classification models separately for each group
        self.run_classification(processed_group_dfs)
        
        # Fill null values using trained models
        self.logger.info("=" * 100)
        self.logger.info("FILLING NULL VALUES USING TRAINED MODELS")
        self.logger.info("=" * 100)
        df_filled = self.fill_null_values_with_trained_models(combined_df)
        
        # Train MLP on complete data with all valid labels
        self.logger.info("=" * 100)
        self.logger.info("TRAINING MLP ON COMPLETE DATA")
        self.logger.info("=" * 100)
        self.train_final_mlp(df_filled)
        
        # Save final results
        self.save_results(df_filled)
    
    def fill_null_values_with_trained_models(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill null values using trained classification models."""
        # Process text if not already processed
        if 'processed_text' not in df.columns:
            df = self.text_preprocessor.process_dataframe(df)
        
        # Fill null values using trained models
        df_filled = self.null_values_filler.fill_all_null_values(df)
        
        # Save filled data
        filled_data_path = self.data_path.parent / "data_filled_sampled.csv"
        self.data_manager.save_data(df_filled, filled_data_path)
        self.logger.info(f"Filled data saved to {filled_data_path}")
        
        return df_filled
    
    def train_final_mlp(self, df: pd.DataFrame) -> Dict:
        """Train final MLP model on complete data with all valid labels."""
        # Retrain TF-IDF on all processed texts
        self.logger.info("Retraining TF-IDF vectorizer on all processed texts...")
        feature_matrix = self.feature_engineer.fit_transform(df['processed_text'].tolist())
        
        # Train MLP model
        self.logger.info("Training MLP model on complete data...")
        mlp_metrics = self.mlp_predictor.train_multilabel_model(df)
        
        return mlp_metrics
    
    def _models_ready(self) -> bool:
        """Check if models are ready for prediction."""
        try:
            # Check if vectorizer is fitted
            if self.feature_engineer.vectorizer is None:
                return False
            
            # Check if clustering model is trained
            if self.text_clusterer.kmeans is None:
                return False
            
            return True
        except Exception:
            return False
    

    

    
    def predict_multiple_texts(self, texts: List[str]) -> List[Dict]:
        """Predict all target values for multiple texts using batch prediction."""
        # Convert single string to list for unified processing
        if isinstance(texts, str):
            texts = [texts]
        
        if not hasattr(self, 'config') or self.config is None:
            self.initialize_components()
        
        # Check if models are trained/loaded, if not, run the pipeline first
        if not self._models_ready():
            self.logger.info("Models not ready. Running training pipeline first...")
            self.run()
        
        results = []
        
        # Get cluster predictions for all texts
        try:
            cluster_predictions = self.text_clusterer.predict_text_clusters(texts, self.feature_engineer, self.text_preprocessor)
        except Exception as e:
            self.logger.error(f"Cluster prediction failed: {e}")
            cluster_predictions = [0] * len(texts)
        
        # Get MLP predictions for all texts at once
        try:
            mlp_results = self.mlp_predictor.predict_multiple_texts(texts)
        except Exception as e:
            self.logger.error(f"MLP prediction failed: {e}")
            mlp_results = [{'text': text} for text in texts]
        
        # Get classification results for each text and group
        for i, text in enumerate(texts):
            # Extract MLP predictions for this text
            text_mlp_predictions = {}
            if i < len(mlp_results) and isinstance(mlp_results[i], dict):
                text_mlp_predictions = {k: v for k, v in mlp_results[i].items() if k != 'text'}
            
            result = {
                'text': text,
                'cluster': cluster_predictions[i] if i < len(cluster_predictions) else 0,
                'mlp_predictions': text_mlp_predictions
            }
            
            results.append(result)
        
        return results

if __name__ == "__main__":
    pipeline = MainPipeline()
    pipeline.initialize_components()

    if pipeline.config['run_pipeline']:
        pipeline.run()
    else:
        # Models will be trained automatically if not ready
        print("Testing single text prediction:")
        results = pipeline.predict_multiple_texts("Hello, how are you?")
        for result in results:
            for key, value in result.items():
                print(f"{key}: {value}")
                print("-" * 100)
        
        print("\nTesting multiple texts prediction:")
        results = pipeline.predict_multiple_texts(["Hello, how are you?", "I'm fine, thank you!"])
        for result in results:
            for key, value in result.items():
                if key == "mlp_predictions":
                    for k, v in value.items():
                        print(f"{k}: {v}")
                else:
                    print(f"{key}: {value}")
                print("-" * 100)