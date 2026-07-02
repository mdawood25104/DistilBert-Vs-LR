# Comparative Analysis: DistilBert

A machine learning project comparing different approaches to tweet classification and sentiment analysis, featuring implementations of DistilBert transformers and Logistic Regression models.

## Project Overview

This project analyzes tweets using multiple machine learning approaches, comparing the performance of:
- **DistilBert** - A lightweight transformer-based model
- **Logistic Regression** - A classical machine learning baseline
- **FastAPI Endpoint** - API implementation for model serving

## Project Structure

```
Tweets.csv
├── Raw tweet data for analysis

Data Preparation For Analysis/
├── clean_tweets.csv      - Preprocessed and cleaned tweet data
└── Code.ipynb            - Data cleaning and preparation notebook

DistilBert/
├── Code.ipynb            - DistilBert model implementation and training
└── saved_model.pb        - Trained DistilBert model

Logistic Regression Implementation/
└── Code.ipynb            - Logistic regression model implementation

FastAPI Endpoint/
├── main.py               - FastAPI server for model serving
└── saved_model.pb        - Model for endpoint deployment
```

## Quick Start

### 1. Data Preparation
- Start with `Tweets.csv` containing raw tweet data
- Run the notebook in `Data Preparation For Analysis/Code.ipynb` to clean and prepare the data
- Output: `clean_tweets.csv`

### 2. Model Development
- **DistilBert Model**: See `DistilBert/Code.ipynb`
- **Logistic Regression**: See `Logistic Regression Implementation/Code.ipynb`
- Both notebooks include training, evaluation, and model saving

### 3. Model Deployment
- Start the FastAPI endpoint: `python FastAPI Endpoint/main.py`
- Use the pre-trained model from `FastAPI Endpoint/saved_model.pb`

## Requirements

- Python 3.7+
- transformers
- scikit-learn
- pandas
- numpy
- fastapi
- uvicorn

## Usage

### Running Notebooks
Open any `.ipynb` file in Jupyter Notebook or JupyterLab to explore the analysis and training processes.

### Starting the API
```bash
cd "FastAPI Endpoint"
python main.py
```

## Results

Compare model performance metrics including accuracy, precision, recall, and F1-score across both approaches.

## Author

Muhammad Dawood Jaffar
