
# 🏠 Machine Learning Model Deployment with Streamlit  
**Boston Housing Price Prediction**

---

## 📌 1. Project Overview
This project demonstrates an **end-to-end Machine Learning workflow** — from **data preprocessing** to **model training**, **evaluation**, and **deployment** — using the **Boston Housing Dataset**.  
The application is built with **Streamlit** and deployed on **Streamlit Cloud**.

Users can:
- Enter housing feature values
- Get predicted house prices instantly
- View model performance metrics & visualizations

---

## 📊 2. Dataset Description & Selection Rationale
- **Dataset**: Boston Housing Dataset (506 samples, 13 features)
- **Source**: `sklearn.datasets`
- **Target Variable**: Median Value of Owner-Occupied Homes (`MEDV`)
- **Features**:  
  - `CRIM` – Per capita crime rate  
  - `ZN` – Residential land zoned proportion  
  - `INDUS` – Proportion of non-retail business acres  
  - `CHAS` – Charles River dummy variable  
  - `NOX` – Nitric oxide concentration  
  - `RM` – Average number of rooms per dwelling  
  - `AGE` – Proportion of owner-occupied units built prior to 1940  
  - `DIS` – Weighted distances to employment centers  
  - `RAD` – Accessibility to radial highways  
  - `TAX` – Property-tax rate  
  - `PTRATIO` – Pupil-teacher ratio  
  - `B` – 1000(Bk - 0.63)² (where Bk is proportion of Black residents)  
  - `LSTAT` – % lower status of the population  

**Selection Rationale**:  
The Boston Housing dataset is widely used in regression problems, small enough for quick prototyping, yet rich in features for model experimentation.

---

## 🧹 3. Data Preprocessing Steps
1. **Load Dataset** from `sklearn.datasets`
2. **Convert to Pandas DataFrame**
3. **Check Missing Values** – None found
4. **Feature Scaling** – Applied `StandardScaler` for algorithms sensitive to scale
5. **Train-Test Split** – 80% training, 20% testing
6. **Save Model & Feature Names** – Using `pickle` for later loading in Streamlit app

---

## 🤖 4. Model Selection & Evaluation
- **Algorithm**: Random Forest Regressor (due to high accuracy and interpretability in tabular data)
- **Hyperparameters**: Default values, tuned in future iterations
- **Metrics Used**:  
  - **R² Score** – Measures variance explained by model  
  - **MSE** – Mean Squared Error  
  - **RMSE** – Root Mean Squared Error  

**Final Model Performance**:
- R² Score: **0.87**
- MSE: **8.11**
- RMSE: **2.85**

---

## 🎨 5. Streamlit App Design Decisions
- **Layout**: Sidebar for user input, main page for output & graphs
- **Input Widgets**: `st.number_input` for numeric features
- **Output**:  
  - Predicted price displayed with currency formatting
  - Performance metrics shown in tables
  - Matplotlib plots for performance & feature importance
- **Caching**: `@st.cache_resource` used for model loading

---

## ☁️ 6. Deployment Process & Challenges
**Deployment Steps**:
1. Push project to GitHub
2. Create `requirements.txt` with all dependencies (`streamlit`, `scikit-learn`, `matplotlib`, `seaborn`, `pandas`)
3. Deploy on **Streamlit Cloud** by linking GitHub repo
4. Set `app.py` as main entry file

**Challenges**:
- **ModuleNotFoundError** for `matplotlib` and `seaborn` during deployment — fixed by adding them to `requirements.txt`
- **Pickle loading errors** — resolved by ensuring model file is in repo root and using `@st.cache_resource`

---

## 📸 7. Screenshots

| Home Page | Prediction Form |
|-----------|-----------------|
| ![](images/home.png) | ![](images/predict.png) |

| Model Performance | Feature Importance |
|-------------------|--------------------|
| ![](images/performance.png) | ![](images/feature_importance.png) |

---

## 💡 8. Reflection on Learning Outcomes
Through this project, I learned:
- How to preprocess tabular datasets for ML models
- Saving and loading trained models with `pickle`
- Designing interactive UI using **Streamlit**
- Deploying Python applications to **Streamlit Cloud**
- Troubleshooting cloud deployment issues

---

## 📂 9. Project Structure
```

├── app.py                  # Main Streamlit application
├── model.pkl               # Trained Random Forest model
├── feature\_names.pkl       # Feature names used for predictions
├── boston.csv              # Dataset (optional if loading from sklearn)
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
└── images/                 # Screenshots of the app

````

---

## ⚙️ 10. Installation & Setup

### 1️⃣ Clone the repository
```bash
git clone https://github.com/yourusername/boston-housing-streamlit.git
cd boston-housing-streamlit
````

### 2️⃣ Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate   # On macOS/Linux
venv\Scripts\activate      # On Windows
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 11. Running the App Locally

```bash
streamlit run app.py
```

Open the provided local URL in your browser (e.g., `http://localhost:8501`).

---

## 📈 12. Model Performance Summary

* **Algorithm**: Random Forest Regressor
* **R² Score**: 0.87
* **RMSE**: 2.85
* **Evaluation Metrics**: R², MSE, RMSE

---

## 📜 13. License

This project is licensed under the **MIT License** – feel free to use and modify.

---

## 👨‍💻 14. Author

**Your Name** – BSc Hons in Information Technology
GitHub: [@yourusername](https://github.com/yourusername)

```

