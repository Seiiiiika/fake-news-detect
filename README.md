# Fake News Detection Using Machine Learning

## Overview

This project aims to build an automated system for detecting fake news using machine learning techniques. The solution includes data preprocessing, feature extraction, model training, evaluation, and deployment as a web application.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Web Application](#web-application)
- [Results](#results)
- [License](#license)

## Project Structure

```
prj_gr/
│
├── data/
│   └── raw/
│       ├── True.csv
│       └── Fake.csv
├── models/
│   ├── model.pkl
│   └── vectorization.pkl
├── src/
│   └── preprocess.ipynb
├── webapp/
│   └── app.py
└── README.md
```

## Dataset

- **True.csv**: Contains real news articles.
- **Fake.csv**: Contains fake news articles.
- Source: [Kaggle Fake News Dataset](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset)

## Installation

1. **Clone the repository:**

   ```sh
   git clone <repository_url>
   cd prj_gr
   ```

2. **Install dependencies:**

   ```sh
   pip install -r requirements.txt
   ```

   _(If `requirements.txt` is not available, install manually: pandas, numpy, scikit-learn, flask, joblib, matplotlib, seaborn)_

3. **(Optional) Create a virtual environment:**
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Mac/Linux
   venv\Scripts\activate     # On Windows
   ```

## Usage

### 1. Data Preprocessing & Model Training

- Open and run `src/preprocess.ipynb` to:
  - Load and clean the data
  - Extract features using TF-IDF
  - Train and evaluate multiple models (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting)
  - Save the best model and vectorizer to the `models/` directory

### 2. Running the Web Application

- Start the Flask web server:
  ```sh
  python webapp/app.py
  ```
- By default, the app runs at [http://127.0.0.1:5000](http://127.0.0.1:5000) or another port if specified.

- Enter a news article in the web interface to receive a prediction (Fake/Real).

## Model Training

- Four models are trained and compared:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - Gradient Boosting
- TF-IDF is used for feature extraction.
- The best-performing model is saved for deployment.

## Web Application

- Built with Flask.
- Loads the trained model and vectorizer.
- Provides a simple web interface for users to input news text and get predictions.

## Results

- **Accuracy:** XX% (replace with your actual result)
- **Precision, Recall, F1-score:** See `src/preprocess.ipynb` for detailed metrics.
- Decision Tree was selected for deployment due to its balance of performance and interpretability.

This project is licensed under the MIT License. See [LICENSE](LICENSE) for
