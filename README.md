# Credit Card Fraud Detection Using Machine Learning

## Overview

Credit card fraud is a major challenge in digital financial transactions. This project uses Machine Learning (ML) techniques to identify potentially fraudulent credit card transactions and compare different models based on their performance.

The project aims to provide a simple, interactive dashboard using **Streamlit** to present fraud detection results and make the system easier to explore.

## Objectives

- Detect potentially fraudulent credit card transactions using Machine Learning.
- Train and compare different ML classification algorithms.
- Evaluate models using appropriate performance metrics.
- Handle the class imbalance commonly found in fraud detection datasets.
- Provide an interactive dashboard using Streamlit.

## Dataset

This project is designed for the Kaggle Credit Card Fraud Detection dataset.

**Dataset:** [Credit Card Fraud Detection — Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

The standard dataset contains:
- 284,807 credit card transactions.
- 492 fraudulent transactions.
- 30 input features and one target column.

### Dataset Features

- `Time`: Time elapsed between a transaction and the first transaction in the dataset.
- `Amount`: Transaction amount.
- `V1` to `V28`: Anonymized transaction features.
- `Class`: Target variable indicating whether a transaction is fraudulent.

Target labels:
- `0` — Legitimate transaction.
- `1` — Fraudulent transaction.

**Note:** Download `creditcard.csv` from Kaggle and place it in the location expected by the project code. The dataset is not included in this repository by default.

## Machine Learning Algorithms

The project may compare the following algorithms, depending on which have been implemented in the source code:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Support Vector Machine (SVM)
5. K-Nearest Neighbors (KNN)
6. Naive Bayes
7. Extra Trees Classifier
8. Gradient Boosting
9. HistGradientBoosting
10. XGBoost
11. Isolation Forest for anomaly detection

## Evaluation Metrics

The models can be evaluated using:

- **Precision:** Measures how many transactions flagged as fraudulent are actually fraudulent.
- **Recall:** Measures how many actual fraudulent transactions are detected.
- **F1-score:** Balances precision and recall.
- **ROC-AUC:** Measures the model's ability to distinguish between classes across thresholds.
- **PR-AUC:** Evaluates precision-recall performance, particularly useful for imbalanced datasets.
- **Confusion Matrix:** Summarizes correct and incorrect predictions.

Accuracy alone is not sufficient because fraudulent transactions represent only a very small fraction of the dataset.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Streamlit
- Matplotlib
- Seaborn
- Joblib

The final list of dependencies depends on the libraries actually used by the implementation.

## Project Structure

```text
CreditCardFraudDetection/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── model_comparison.csv
├── models/
│   └── saved model files
└── notebooks/
    └── fraud_detection.ipynb
```

This is an illustrative structure. Keep the filenames and folders that actually exist in your project. The dashboard must be able to locate all required datasets and model artifacts.

## Installation and Setup

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd CreditCardFraudDetection
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with your actual GitHub repository URL.

### 2. Create a Virtual Environment

On Windows:

```bash
python -m venv .venv
```

Activate it:

```bash
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Add the Dataset and Model Files

Download `creditcard.csv` from Kaggle and place it in the location expected by the application.

If the application loads previously trained models or a model comparison CSV file, make sure those files are also available in their expected locations.

### 5. Run the Streamlit Application

From the project directory, execute:

```bash
python -m streamlit run app.py
```

Streamlit will display a local URL in the terminal. Open that URL in your browser to access the dashboard.

## Troubleshooting

### Error: `No module named streamlit`

Install the dependencies in the same Python environment used to run the application:

```bash
python -m pip install -r requirements.txt
```

Then run:

```bash
python -m streamlit run app.py
```

If you are using an existing virtual environment, activate it before running these commands.

### Error: Dataset or model file not found

Check that the required file exists and that its path matches the path used in `app.py` or the relevant training script.

### Error: Missing Python library

Install the missing library or add the appropriate dependency to `requirements.txt`, then reinstall the dependencies.

## Limitations

- Fraudulent transactions are highly underrepresented in the dataset.
- Anonymized features limit direct interpretation of transaction characteristics.
- Model performance depends on preprocessing, data splitting, feature selection and classification thresholds.
- Results on a historical dataset do not guarantee performance on real-world transactions.
- This project is intended for educational and demonstration purposes, not as a production financial security system.

## Future Enhancements

- Improve the Streamlit dashboard with interactive model comparisons.
- Display confusion matrices and performance metrics for each model.
- Allow users to explore classification thresholds.
- Improve handling of imbalanced data.
- Add input validation and informative error messages.
- Explore deployment through Streamlit Community Cloud or another hosting service.

## Disclaimer

This project is developed for educational purposes and demonstrates the application of Machine Learning to credit card fraud detection. It should not be used as the sole basis for real-world financial decisions.

## Author

**Project:** Credit Card Fraud Detection Using Machine Learning

Developed as a Machine Learning project to explore fraud detection, model evaluation and interactive dashboard development.

## License

A license has not yet been specified. Add a suitable license before permitting others to reuse or distribute the project.
