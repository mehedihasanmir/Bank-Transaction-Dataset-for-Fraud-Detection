# 🕵️ Fraud Detection Using Machine Learning

This project focuses on detecting fraudulent transactions using machine learning techniques. It leverages a synthetic financial transactions dataset from Kaggle, containing detailed behavioral, transactional, and customer attributes, with the goal of building models that can effectively distinguish between **fraudulent** and **non-fraudulent** activities.

---

## 📊 Dataset Overview

dataset:- https://www.kaggle.com/datasets/valakhorasani/bank-transaction-dataset-for-fraud-detection/data

The dataset contains **2,512** transaction records with rich feature information such as:

- `TransactionID`: Unique ID for each transaction.
- `AccountID`: Account identifier.
- `TransactionAmount`: Amount involved in the transaction.
- `TransactionDate` & `PreviousTransactionDate`: Used to calculate transaction frequency.
- `TransactionType`: Credit or Debit.
- `Location`, `DeviceID`, `IP Address`, `MerchantID`: Behavioral data.
- `AccountBalance`: Post-transaction balance.
- `Channel`: Online, ATM, or Branch.
- `CustomerAge`, `CustomerOccupation`: Demographic information.
- `TransactionDuration`: Duration of the transaction.
- `LoginAttempts`: Number of login attempts before the transaction.

---

## 🎯 Project Objective

The primary objective is to **detect fraudulent transactions** using various machine learning techniques. The task includes:

- Data preprocessing and feature engineering.
- Anomaly detection using **Isolation Forest**.
- Feature creation like `IsFraud` from anomaly scores.
- Supervised classification using traditional machine learning algorithms.

---

## 💠 Technologies Used

- Python 🐍
- Pandas, NumPy, Matplotlib, Seaborn
- Scikit-learn
- Imbalanced-learn (SMOTE)
- Jupyter Notebook

---

## 🔍 Feature Engineering

New features were created to prediction Fraud or not:

- **TimeDelta**: Time between `TransactionDate` and `PreviousTransactionDate`
- **IsFraud**: Binary label derived using Isolation Forest for anomaly detection

---

## 🤖 Models Trained

Several supervised learning algorithms were trained and evaluated, including:

- Logistic Regression
- Random Forest
- Decision Tree
- Gradient Boosting
- AdaBoost
- Support Vector Machine (SVM)
- K-Nearest Neighbors
- Naive Bayes
- Neural Network (MLPClassifier)
- LDA, QDA
- Extra Trees
- Bagging Classifier

**Imbalanced data** was handled using **SMOTE** (Synthetic Minority Oversampling Technique).

---

## 📈 Evaluation Metrics

- Accuracy Score
- Confusion Matrix
- (Optional): F1 Score, Precision, Recall, ROC-AUC

> Since this is a fraud detection problem, further focus can be given to **recall** and **precision** rather than just accuracy.

---

## 📁 Project Structure

```bash
🔹 dataset/                     # Dataset files (not included in repo)
🔹 notebook.ipynb              # Main Jupyter notebook
🔹 README.md                   # Project overview
```

---

## 🚀 How to Run

1. Clone this repository:

```bash
git clone https://github.com/mehedihasanmir/Bank-Transaction-Dataset-for-Fraud-Detection.git
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Open the notebook:

```bash
jupyter notebook notebook.ipynb
```

4. Download the dataset from Kaggle and place it in the `dataset/` folder.

---

## 📌 Future Work

- Use deep learning models (LSTM/Autoencoders)
- Use explainability tools like SHAP
- Deploy model using Streamlit or Flask
- Apply real-time fraud detection pipeline

---

## 🙋‍♂️ Author

**Mehedi Hasan Mir**\
Final Year CSE Student | AI & Machine Learning Enthusiast\
Email: [mdmehedihassanmir@gmail.com(mailto\:mdmehedihassanmir@gmail.com)\
GitHub: [https://github.com/mehedihasanmir](https://github.com/mehedihasanmir)

---

## 📄 License

This project is for educational purposes only.

