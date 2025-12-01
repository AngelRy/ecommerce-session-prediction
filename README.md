```
🛒 Ecommerce Session Prediction

Predicting purchase intent from large-scale ecommerce user behavior data

📘 Overview

This project focuses on predicting whether an ecommerce browsing session will result in a purchase, using large-scale clickstream behavioral data. It covers the entire pipeline:

Data ingestion & preprocessing

Efficient storage and querying under hardware constraints

Feature engineering

Machine learning modeling

Evaluation and insights

The goal is to build a reproducible ML workflow able to handle millions of events without requiring high-end compute resources.

📦 Datasets

The project uses public datasets sourced from Kaggle:

1. Ecommerce Behavior Data from Multi-Category Store

Contains:

user behavior logs

event types (view, cart, purchase)

timestamps

device/browser info

product metadata

2. Ecommerce Purchases Dataset

Contains:

purchase events

session-level linking

cart and checkout behaviors

Both datasets can be found on Kaggle by searching their names.
(Direct links can be added if desired.)

⚙️ Project Motivation & Constraints

The entire pipeline was developed on modest and outdated hardware, far below ideal for handling large ecommerce datasets.
This limitation shaped many engineering decisions:

🧩 The Constraints

Limited RAM

Slow disk I/O

No GPU

No ability to run Spark or large distributed systems

🛠️ Workarounds & Engineering Solutions
✔ Parquet Instead of CSV

Used to reduce memory usage and speed up processing due to:

compression

columnar access

faster scanning

✔ DuckDB Instead of Pandas-Only or Spark

DuckDB was used as the in-process analytical engine because it can:

execute SQL queries directly on parquet files

handle large datasets on a laptop

avoid full dataframe loading into RAM

serve as a “poor man’s OLAP engine”

✔ Chunked & Streaming Processing

For steps requiring pandas, chunk-based loading prevented memory overflow.

✔ Model Training with Memory Awareness

Feature selection and dimensionality reduction focused on runtime performance.

This approach makes the repository highly useful for anyone working under similar hardware limitations.

🧠 Machine Learning Approach

The modeling part explores:

classification models: Logistic Regression, Random Forest, XGBoost

session-level aggregation

event frequency patterns

time-based behavior

Evaluation includes:

ROC-AUC

precision/recall

confusion matrix

feature contribution insights

(Consider adding model results if applicable.)
```
```
📂 Repository Structure
ecommerce-session-prediction/
│
├── data/                       # Parquet/duckdb intermediate files (ignored)
├── notebooks/                  # Exploratory analysis and modeling notebooks
├── src/                        # Data processing & model training scripts
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
└── .gitignore                  # Ignoring datasets, temp files, envs
```
🚀 How to Run
git clone https://github.com/AngelRy/ecommerce-session-prediction.git
cd ecommerce-session-prediction

# Create conda environment
conda create -n ecom_env_new python=3.11 -y
conda activate ecom_env_new

pip install -r requirements.txt

# Start exploration or run scripts
```
jupyter lab

🧰 Tech Stack

Python 3.11

DuckDB (core analytical engine)

Pandas (chunked processing only)

NumPy

Scikit-learn

Parquet (PyArrow backend)

Matplotlib / Seaborn for visualizations

🏗️ Key Strength: Working with Limited Hardware

This project demonstrates how to build large-scale ML pipelines on everyday consumer hardware using smart data-engineering practices.

It is a practical, real-world example for:

students

laptop-only data scientists

Kaggle competitors

learners without access to cloud compute

📜 License

MIT License (or whatever you will choose).
```
