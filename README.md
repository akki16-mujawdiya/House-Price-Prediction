# House Price Prediction using Machine Learning

## Project Overview

This project predicts house prices using Machine Learning techniques in Python. The dataset is preprocessed using Scikit-Learn pipelines, and multiple regression algorithms are trained and evaluated to determine the best-performing model.

The project demonstrates the complete Machine Learning workflow, including data preprocessing, feature engineering, model training, and performance evaluation.

---

## Dataset

The project uses the **Housing.csv** dataset containing various housing-related features such as:

* Longitude
* Latitude
* Housing Median Age
* Total Rooms
* Total Bedrooms
* Population
* Households
* Median Income
* Ocean Proximity

### Target Variable

* `median_house_value`

---

## Technologies Used

* Python
* NumPy
* Pandas
* Scikit-Learn

---

## Machine Learning Algorithms

The following regression models are implemented and compared:

### 1. Linear Regression

A simple regression model used as a baseline for house price prediction.

### 2. Decision Tree Regressor

A tree-based model capable of capturing non-linear relationships in the data.

### 3. Random Forest Regressor

An ensemble learning algorithm that combines multiple decision trees to improve prediction accuracy and reduce overfitting.

---

## Data Preprocessing

### Stratified Sampling

The dataset is split into training and testing sets using **StratifiedShuffleSplit** based on income categories to maintain representative data distribution.

### Numerical Features Processing

* Missing values are handled using Median Imputation.
* Features are standardized using StandardScaler.

### Categorical Features Processing

* Ocean proximity categories are transformed using One-Hot Encoding.

### Pipeline

Scikit-Learn Pipelines and ColumnTransformer are used to automate the preprocessing workflow.

---

## Model Evaluation

The models are evaluated using:

* 10-Fold Cross Validation
* Root Mean Squared Error (RMSE)

Lower RMSE values indicate better model performance.

---

## Project Structure

House-Price-Prediction/

├── Housing.csv

├── main.py

├── pipeline.pkl

├── requirements.txt

├── README.md

└── input.csv

---

## Installation

Clone the repository:

```bash
git clone https://github.com/akki16-mujawdiya/House-Price-Prediction.git
```

Move into the project directory:

```bash
cd House-Price-Prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run the Project

Execute the Python script:

```bash
python main.py
```

The program will:

1. Load the housing dataset.
2. Perform preprocessing.
3. Train multiple regression models.
4. Evaluate models using cross-validation.
5. Display RMSE statistics for each model.

---

## Features

* Data preprocessing using pipelines
* Stratified train-test splitting
* Numerical and categorical feature handling
* Multiple regression model comparison
* Cross-validation performance evaluation
* Scalable and reusable machine learning workflow

---

## Future Improvements

* Hyperparameter tuning using GridSearchCV
* Feature engineering
* Model deployment using Flask or Streamlit
* Interactive web interface for predictions

---

## Author

Akki Mujawdiya

Machine Learning Project using Python and Scikit-Learn.
