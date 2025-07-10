# MatrixDNA - Text Analysis Pipeline

A comprehensive text analysis pipeline for clustering, classification, and multi-label prediction on text data.

## Features

- **Text Preprocessing**: Configurable text cleaning with lemmatization, stopword removal, and normalization
- **TF-IDF Vectorization**: Feature extraction with customizable n-grams and frequency filters
- **K-means Clustering**: Automatic optimal cluster detection using silhouette analysis
- **Multi-class Classification**: Logistic regression models for individual target variables
- **Multi-label MLP**: Neural network for simultaneous prediction of all target variables
- **Null Value Filling**: Smart imputation using trained classification models
- **Web Interface**: Interactive Streamlit demo for easy testing and visualization
- **Modular Architecture**: Clean separation of concerns with reusable components

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. For the web demo, ensure Streamlit is installed:
```bash
pip install streamlit plotly
```

## Usage

### Run Full Pipeline
```python
from claude_work import MainPipeline

pipeline = MainPipeline()
pipeline.run()  # Trains models and processes data
```

### Predict New Text
```python
# Single text prediction
result = pipeline.predict_multiple_texts("Your text here")

# Multiple texts prediction
results = pipeline.predict_multiple_texts(["Text 1", "Text 2"])
```

### Interactive Web Demo
Launch the Streamlit web interface for easy interaction:
```bash
streamlit run demo.py
```

The demo provides:
- **Single Email Classification**: Real-time text analysis with confidence scores
- **Batch Processing**: Upload CSV files for bulk classification
- **Analytics Dashboard**: Visualizations of classification results
- **Sample Emails**: Pre-loaded examples to test the system
- **Model Status**: Training progress and model readiness indicators

## Configuration

Edit `config/config.json` to customize:
- Sampling strategy and group definitions
- Preprocessing options (lemmatization, stopwords, etc.)
- TF-IDF parameters (n-grams, max features)
- Clustering settings (max clusters, plots)
- Classification parameters (test size, CV folds)

## Project Structure

```
MatrixDNA/
├── claude_work.py       # Main pipeline implementation
├── demo.py              # Streamlit web interface
├── config/
│   └── config.json      # Configuration settings
├── data/
│   └── data.csv         # Input data
├── models/              # Trained models (auto-created)
├── analysis/            # Logs and plots (auto-created)
└── requirements.txt     # Dependencies
```

## Output

The pipeline generates:
- Clustered data with assigned cluster IDs
- Trained classification models for each target variable
- Multi-label MLP model for simultaneous prediction
- Cluster vocabulary analysis
- Performance metrics and visualizations

## Key Components

- **ConfigManager**: Handles configuration loading
- **DataManager**: Data I/O operations
- **TextPreprocessor**: Configurable text cleaning
- **FeatureEngineer**: TF-IDF vectorization
- **TextClusterer**: K-means clustering with optimization
- **TextClassificationPipeline**: Individual classifiers
- **MultiLabelMLPPredictor**: Neural network for multi-label prediction
- **MainPipeline**: Orchestrates the entire workflow
