# Rossmann Store Sales Forecasting

## Internship Project – Machine Learning & Streamlit Deployment

### Project Overview

This project predicts daily sales for Rossmann stores using machine learning models trained on historical sales data. The workflow includes data preprocessing, exploratory data analysis (EDA), feature engineering, model comparison, and deployment of the best-performing model through a Streamlit web application.

---

## Project Structure

```text
Rossmann Project/
│
├── rossmann_analysis.ipynb          # Complete EDA, feature engineering & model training
├── Rossmann_Project_Summary.pptx    # Project presentation
├── README.md                        # Project documentation
│
├── train.csv                        # Training dataset
├── test.csv                         # Test dataset
├── store.csv                        # Store information
│
├── models/
│   └── sales_model_<timestamp>.pkl  # Trained XGBoost model
│
├── rossmann_pipeline.log            # Execution log
│
├── app.py                           # Streamlit application
└── requirements.txt                 # Required Python packages
```

---

## Technologies Used

* Python 3.12
* Pandas & NumPy
* Scikit-learn
* XGBoost
* TensorFlow / Keras (LSTM comparison)
* Matplotlib & Seaborn
* Streamlit

---

## Models Compared

The notebook trains and evaluates the following models:

| Model                   | Purpose                          |
| ----------------------- | -------------------------------- |
| Linear Regression       | Baseline model                   |
| Random Forest Regressor | Ensemble learning                |
| XGBoost Regressor       | Best performing production model |
| LSTM Neural Network     | Deep learning comparison         |

**Selected Production Model:** XGBoost Regressor

---

## How to Run in VS Code (Without Virtual Environment)

### Step 1: Open the Project

* Open **VS Code**
* Select **File → Open Folder**
* Open the **Rossmann Project** folder.

### Step 2: Install Dependencies

Open the integrated terminal and run:

```bash
"C:\Users\rhndu\anaconda3\python.exe" -m pip install -r requirements.txt
```

If required, install additional packages:

```bash
"C:\Users\rhndu\anaconda3\python.exe" -m pip install jupyter tensorflow xgboost matplotlib seaborn statsmodels
```

### Step 3: Run the Notebook

1. Open `rossmann_analysis.ipynb`
2. Select the **Anaconda Python** kernel.
3. Click **Run All**.

The notebook will:

* Load and preprocess the datasets
* Perform exploratory data analysis
* Engineer features
* Train and compare multiple models
* Save the best XGBoost model in the `models/` folder

### Step 4: Launch the Streamlit Application

From the project directory, run:

```bash
"C:\Users\rhndu\anaconda3\python.exe" -m streamlit run app.py
```

The application will open in your browser at:

```text
http://localhost:8501
```

---

## Streamlit Features

* Predict sales for a single store and date
* Upload a CSV for bulk sales prediction
* Interactive prediction dashboard
* Estimated customer count
* Download predictions as CSV

---

## Dataset

The project uses the Rossmann Store Sales dataset:

* `train.csv` – Historical daily sales
* `test.csv` – Future prediction data
* `store.csv` – Store metadata

---

## Output

The trained XGBoost model is automatically saved as:

```text
models/sales_model_<timestamp>.pkl
```

This model is loaded by the Streamlit application for real-time predictions.

---

## Author

**Rossmann Store Sales Forecasting – Internship Project**

Developed using Python, XGBoost, TensorFlow, and Streamlit.
