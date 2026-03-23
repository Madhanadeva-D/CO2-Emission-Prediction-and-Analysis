# 🌿 CO₂ Emission Prediction and Analysis

An interactive web application for predicting vehicle CO₂ emissions using a **Gradient Boosting Regressor** model, deployed with Streamlit.

[🚀 Open Live App](https://co2-emission-prediction-and-analysis.streamlit.app/)

---

## 📌 Overview

This project analyzes a Canadian vehicle dataset (7,385 records) and predicts CO₂ emissions (g/km) based on three key vehicle parameters:

- **Engine Size (L)**
- **Number of Cylinders**
- **Combined Fuel Consumption (L/100 km)**

The best-performing model — Gradient Boosting — was selected after comparing four ML models (Linear Regression, Random Forest, KNN, Gradient Boosting), achieving an **R² score of 0.9612**.

---

## 🚀 Features

- Real-time CO₂ prediction using interactive sliders
- Color-coded emission level badge (🟢 Low / 🟡 Medium / 🔴 High)
- Emission gauge showing prediction relative to dataset range
- Model stats (R², MAE) displayed in sidebar
- Dark green themed UI

---

## 🗂️ Project Structure

```
├── app.py                          # Streamlit web application
├── co2_emission_prediction.ipynb   # EDA, preprocessing & model notebook
├── co2 Emissions.csv               # Dataset
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation
```

---

## ⚙️ Setup & Run

**1. Clone the repository**
```bash
git clone https://github.com/your-username/co2-emission-prediction.git
cd co2-emission-prediction
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the app**
```bash
streamlit run app.py
```

---

## 🤖 Model Details

| Model              | R² Score |
|--------------------|----------|
| Gradient Boosting  | **0.9612** ✅ |
| Random Forest      | 0.9609   |
| KNN                | 0.9561   |
| Linear Regression  | 0.9074   |

**Gradient Boosting settings:** `n_estimators=200`, `learning_rate=0.1`, `max_depth=4`

---

## 📊 Dataset

- **Source:** Canadian Vehicle Fuel Consumption Ratings
- **Records:** 7,385 vehicles
- **Fuel types:** Premium Gasoline, Regular Gasoline, Diesel, Ethanol (E85)
- **Preprocessing:** Natural Gas vehicles removed, outliers filtered using Z-score (threshold 1.9)

---

## 🛠️ Tech Stack

- Python 3.10+
- Streamlit
- Scikit-learn
- Pandas / NumPy
- Matplotlib / Seaborn
- SciPy
