# Chronic Kidney Disease Prediction 🧠💉

A Machine Learning-based web application for predicting the presence of Chronic Kidney Disease (CKD) based on clinical parameters. This project utilizes data preprocessing, model training, evaluation, and deployment using modern tools and techniques.

## 🔬 Problem Statement

Chronic Kidney Disease (CKD) is a serious public health issue that, if not detected early, can lead to permanent kidney failure. This project aims to automate the prediction process using machine learning to help doctors and patients make better decisions.

---

## 🧰 Tech Stack Used

| Layer        | Tools/Technologies                                 |
|--------------|----------------------------------------------------|
| 👩‍💻 Language   | Python                                            |
| 📊 ML Models | Random Forest, Logistic Regression, SVM, etc.      |
| 📦 Libraries  | Pandas, NumPy, Matplotlib, Seaborn, scikit-learn  |
| 🌐 Deployment | Flask, Streamlit (optional), GitHub               |
| 🧪 Dataset    | UCI CKD Dataset (via Kaggle or public source)     |

---

## 📁 Project Structure

```
├── Python Notebooks/
│   ├── Main-Code.ipynb                  # Final model training and prediction
│   ├── Kidney_Disease_Prediction.ipynb  # Exploratory data analysis
├── models
│   ├── kidney.pkl                       # Trained model file
├── templates/                           # HTML files for Flask or web integration
├── README.md
├── app.py                               # Flask application (if applicable)
├── requirements.txt                     # Python dependencies
└── model.pkl                            # Saved ML model
```

---

## 📊 Dataset Info

- **Source**: UCI Machine Learning Repository / Kaggle
- **Features**:  
  `age`, `bp`, `sg`, `al`, `su`, `rbc`, `pc`, `pcc`, `ba`, `bgr`, `bu`, `sc`, `sod`, `pot`, `hemo`, `pcv`, `wc`, `rc`, `htn`, `dm`, `cad`, `appet`, `pe`, `ane`, and `classification`.

- **Target**: `classification` → whether the patient has CKD or not

---

## 🧠 Machine Learning Workflow

1. **Data Cleaning**: Handling missing values, encoding categorical data
2. **Exploratory Data Analysis**: Visualizing distributions and relationships
3. **Feature Selection**: Choosing relevant features based on correlation
4. **Model Building**: Training and comparing multiple classifiers
5. **Evaluation**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
6. **Saving Model**: Using `joblib` or `pickle` for deployment
7. **Deployment**: Web interface using Flask (optional)

---

## ✅ Model Accuracy (Example)

| Model               | Accuracy |
|--------------------|----------|
| Random Forest       | 98%      |
| Logistic Regression | 95%      |
| SVM                 | 94%      |

---

## 💻 How to Run Locally

1. **Clone the repository**
   ```bash
   git clone https://github.com/supriya46788/Chronic-Kidney-Disease-Prediction.git
   cd Chronic-Kidney-Disease-Prediction
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Jupyter Notebook**
   ```bash
   jupyter notebook
   ```

4. *(Optional)* **Run Flask App**
   ```bash
   python app.py
   ```

