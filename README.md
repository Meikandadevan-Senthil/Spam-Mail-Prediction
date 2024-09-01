# Spam Mail Classification Project

## Overview

This project involves spam Mail classification using various machine learning models. The goal is to transform text data into TF-IDF feature vectors and use these vectors to make predictions with different models, including RandomForest, Logistic Regression, and ExtraTrees.

## Table of Contents

1. [Requirements](#requirements)
2. [Installation](#installation)
3. [Data](#data)
4. [Usage](#usage)
5. [Results](#results)
6. [License](#license)

## Requirements

The following Python libraries are required for this project:

- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `seaborn`

You can install the required libraries using pip:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

## Installation

Clone the repository and navigate to the project directory:

```bash
git https://github.com/Meikandadevan-Senthil/Spam-Mail-Prediction
cd Spam-Mail-Prediction
```

## Data

- Ensure your data is in a format suitable for text classification (e.g., CSV file with a column for text and a column for labels).
- Update the data loading section of the code to read your dataset.

## Usage

1. **Load the Data**:
   Update the `data_loading` section in the code to load your dataset.

2. **Preprocess the Data**:
   Perform any necessary text preprocessing (e.g., tokenization, lowercasing).

3. **Feature Extraction**:
   Convert text data into TF-IDF features:
   ```python
   from sklearn.feature_extraction.text import TfidfVectorizer

   feature_extraction = TfidfVectorizer(min_df=1, stop_words='english')
   X_train_features = feature_extraction.fit_transform(X_train)
   X_test_features = feature_extraction.transform(X_test)
   ```

4. **Train Models**:
   Train your models (RandomForest, Logistic Regression, ExtraTrees) and make predictions:
   ```python
   from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
   from sklearn.linear_model import LogisticRegression

   # Initialize models
   rf_model = RandomForestClassifier()
   log_model = LogisticRegression()
   et_model = ExtraTreesClassifier()

   # Train models
   rf_model.fit(X_train_features, y_train)
   log_model.fit(X_train_features, y_train)
   et_model.fit(X_train_features, y_train)

   # Make predictions
   prediction1 = log_model.predict(X_test_features)
   prediction2 = rf_model.predict(X_test_features)
   prediction3 = et_model.predict(X_test_features)
   ```

5. **View Predictions**:
   Create a table to compare predictions from different models:
   ```python
   import pandas as pd

   prediction_table = {'Models': ['RandomForest', 'Logistic', 'ExtraTree'],
                       'Prediction': [prediction2, prediction1, prediction3]}
   prediction_table_df = pd.DataFrame(prediction_table)
   print(prediction_table_df)
   ```

## Results

The `prediction_table_df` DataFrame displays the predictions made by different models. Review the predictions to evaluate model performance.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
