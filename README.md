# 🌸 SheSync — Multimodal AI System for Menstrual Health & PCOS Risk Monitoring

> A personalised period-tracking and wellness companion that learns **your** cycle (not a fixed 28-day assumption), flags early PCOS risk signals using machine learning, and supports women at every life stage — from first period to perimenopause.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-REST%20API-000000?logo=flask&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Model-1f77b4)
![Status](https://img.shields.io/badge/Status-Academic%20Prototype-purple)

> ⚠️ **Disclaimer:** SheSync is an academic prototype built for awareness and education. It is **not a medical device** and does **not** diagnose PCOS or any other condition. Always consult a qualified gynaecologist for medical decisions.

---

## 📖 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Why SheSync Is Different](#-why-shesync-is-different)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Machine Learning Pipeline](#-machine-learning-pipeline)
- [Getting Started](#-getting-started)
- [Email Reminder Setup](#-email-reminder-setup-optional)
- [REST API Reference](#-rest-api-reference)
- [Wearable Integration](#-wearable-integration)
- [Limitations & Future Work](#-limitations--future-work)
- [Team](#-team)
- [License](#-license)

---

## 🔍 Overview

Most period trackers assume a 28-day cycle, only log bleeding dates, and ignore conditions like PCOS until expensive tests are done. **SheSync** takes a different approach:

- It **predicts your next period from your own cycle history**.
- It lets you **log mood, sleep, pain and symptoms** in about 30 seconds a day.
- It runs an **AI-assisted PCOS risk check** trained on real clinical data (541 patients across 10 hospitals in Kerala, India).
- It adapts its tone and guidance to your **life stage** (first period, teen, 20s–30s, 40s+/perimenopause).
- It includes a prototype **ESP32-based wearable** pipeline for heart rate, temperature, steps and skin-response (stress) data.

The project ships in three parts: a **Streamlit app**, a **Flask REST API**, and a **standalone HTML website**, backed by notebooks and trained models.

---

## ✨ Key Features

| Module | What it does |
|---|---|
| 🏠 **Life-stage Home** | Personalised welcome and guidance for first period (9–14), teens, 20s–30s, and 40s+ (perimenopause) |
| 📓 **Daily Log** | Track period status, flow, pain, sleep, mood, 8 common symptoms, and free-text notes |
| 📅 **My Cycle** | Predicts next period, fertile window, and current cycle phase (menstrual / follicular / ovulation / luteal) from as few as 1–2 dates; flags irregular cycles (std. dev. > 7 days) |
| 🔬 **PCOS Risk Check** | Plain-language questionnaire (symptoms, lifestyle, optional hormone and ultrasound values) → risk score, contributing-factor chart, and next-step advice |
| 📡 **Wearable Data** | Simulated live sensor stream and 7-day trends (heart rate, basal body temperature) with ovulation-marker detection |
| 👩‍⚕️ **Find a Doctor** | Directory of gynaecologists in Bengaluru + downloadable **Health Summary** to share with your doctor |
| 🔔 **Reminders** | In-app period alerts and real email reminders via Gmail SMTP |

---

## 💡 Why SheSync Is Different

| Typical apps | SheSync |
|---|---|
| Assume a 28-day cycle for everyone | Learns **your** cycle length from your history (21 days or 45 — it adapts) |
| Track only period dates | Tracks mood, sleep, pain and symptoms daily so patterns emerge naturally |
| PCOS detected only after costly tests | Flags early risk signals using ML trained on real clinical data |
| Ignore first-period and perimenopause stages | Supports every life stage with tailored guidance |

---

## 🛠 Tech Stack

- **Frontend / App:** Streamlit, custom CSS, standalone HTML/CSS/JS website
- **Backend:** Flask, Flask-CORS
- **ML / Data:** pandas, NumPy, scikit-learn (Random Forest), XGBoost, imbalanced-learn (SMOTE), SHAP, Matplotlib, Seaborn
- **Model persistence:** joblib (`.pkl`)
- **Notifications:** `smtplib` (Gmail SMTP over SSL)
- **Hardware prototype:** ESP32 (heart rate, temperature, step count, GSR)

---

## 📁 Project Structure

```
SheSync/
├── app/
│   └── app.py                       # Streamlit application (7 pages)
├── backend_api.py                   # Flask REST API
├── data/
│   ├── PCOS_data_without_infertility.xlsx   # Clinical dataset (sheet: Full_new)
│   └── PCOS_infertility.csv
├── models/
│   ├── pcos_rf_model.pkl            # Trained Random Forest
│   ├── pcos_xgb_model.pkl           # Trained XGBoost
│   └── feature_names.pkl            # Feature list used for training (41 features)
├── notebooks/
│   └── 01_EDA_PCOS.ipynb            # EDA, preprocessing, training, SHAP analysis
├── SheSync_Website.html             # Standalone website (v1)
├── SheSync_Website_FIXED_Chart_Reminder_v2.html   # Standalone website (v2: chart + reminder fixes)
├── .gitignore
└── README.md
```

---

## 🧠 Machine Learning Pipeline

**Dataset:** 541 patients × 41 clinical features (after cleaning), collected from 10 hospitals in Kerala, India.
Class balance: 364 non-PCOS / 177 PCOS.

**Pipeline (see `notebooks/01_EDA_PCOS.ipynb`):**

1. Load the `Full_new` sheet and inspect missing values.
2. Drop the junk column; impute missing values (median / mode).
3. Coerce all features to numeric and fill any remaining NaNs with column medians.
4. 80/20 train–test split (`random_state=42`).
5. Handle class imbalance with **SMOTE** (applied to the training set only).
6. Train **Random Forest** and **XGBoost** classifiers.
7. Interpret feature importance with **SHAP**.
8. Export models and feature names with `joblib`.

**Results on the held-out test set (109 samples):**

| Model | Accuracy | F1 (PCOS class) |
|---|---|---|
| Random Forest | 0.826 | 0.708 |
| XGBoost | **0.835** | **0.727** |

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/SheSync.git
cd SheSync
```

### 2. Create a virtual environment

```bash
python -m venv venv
# macOS / Linux
source venv/bin/activate
# Windows
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install streamlit pandas numpy matplotlib flask flask-cors joblib \
            scikit-learn xgboost imbalanced-learn shap seaborn openpyxl jupyter
```

> 💡 Tip: pin your versions with `pip freeze > requirements.txt` so the pickled models load reliably (scikit-learn and XGBoost pickles are version-sensitive).

### 4. Run the Streamlit app

```bash
streamlit run app/app.py
```

Opens at `http://localhost:8501`.

### 5. Run the Flask backend (optional)

```bash
python backend_api.py
```

API runs at `http://localhost:5000`.

### 6. Open the standalone website

Just open `SheSync_Website_FIXED_Chart_Reminder_v2.html` in any modern browser — no server required.

### 7. Re-train the models (optional)

```bash
jupyter notebook notebooks/01_EDA_PCOS.ipynb
```

Run all cells to regenerate the `.pkl` files in `models/`. Update any hard-coded file paths to match your machine.

---

## 📧 Email Reminder Setup (optional)

Real email reminders use Gmail SMTP and Streamlit secrets. **Never commit credentials** — `secrets.toml` is already in `.gitignore`.

1. Enable 2-Step Verification on your Google account.
2. Generate a **Gmail App Password** (Google Account → Security → App passwords).
3. Create `.streamlit/secrets.toml` in the project root:

```toml
SENDER_EMAIL = "your_email@gmail.com"
SENDER_PASSWORD = "your_16_char_app_password"
```

4. Restart the Streamlit app and use the **🔔 Set Reminder** page.

---

## 🔌 REST API Reference

Base URL: `http://localhost:5000`

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check — status, model-load state, endpoint list |
| `POST` | `/api/pcos-risk` | PCOS risk score, risk level, and recommendation |
| `POST` | `/api/cycle-predict` | Next period, fertile window and current phase |
| `POST` | `/api/daily-log` | Save a daily log entry (demo — echoes data back) |
| `GET` | `/api/phase/<last_period>/<cycle_length>` | Quick current-phase lookup |
| `POST` | `/api/wearable` | Receive ESP32 sensor payloads and flag ovulation signal (temp ≥ 36.9 °C) |

### Example — PCOS risk

```bash
curl -X POST http://localhost:5000/api/pcos-risk \
  -H "Content-Type: application/json" \
  -d '{
        "age": 25, "weight": 60, "height": 160,
        "follicle_r": 9, "follicle_l": 8,
        "amh": 4.2, "cycle_type": "irregular",
        "weight_gain": true, "fast_food": true
      }'
```

**Response (abridged):**

```json
{
  "success": true,
  "risk_score": 42,
  "risk_level": "moderate",
  "bmi": 23.4,
  "recommendation": "Monitor your symptoms for 2-3 months. Consider a hormonal blood test.",
  "model_prediction": 1,
  "disclaimer": "This is an AI-generated risk assessment for awareness only..."
}
```

### Example — Cycle prediction

```bash
curl -X POST http://localhost:5000/api/cycle-predict \
  -H "Content-Type: application/json" \
  -d '{"last_period": "2026-04-18", "cycle_length": 30, "period_duration": 5}'
```

### Example — Quick phase check

```bash
curl http://localhost:5000/api/phase/2026-04-18/30
```

---

## 📡 Wearable Integration

SheSync is designed around a low-cost **ESP32 wrist wearable** that streams data every 60 seconds:

| Signal | Used for |
|---|---|
| ❤️ Heart rate / HRV | Hormone-linked variability |
| 🌡️ Body temperature | Ovulation detection via temperature rise |
| 🏃 Steps | Activity tracking |
| ⚡ Stress (GSR / skin response) | Pre-period stress patterns |

In the Streamlit app, the **Wearable Data** page currently shows a **simulated** live stream. The device posts JSON to `/api/wearable`:

```json
{
  "device_id": "ESP32_001",
  "heart_rate": 72,
  "temperature": 36.6,
  "steps": 4821,
  "gsr": 0.34,
  "timestamp": "2026-05-17T10:30:00"
}
```

---

## ⚠️ Limitations & Future Work

Being upfront about the current state of the prototype:

- **PCOS risk score in the app is rule-based.** The Streamlit risk check uses a weighted checklist; the Flask API combines the same rule-based score with an XGBoost prediction, but the model is fed only a subset of features (the rest default to zero). Wiring the full 41-feature trained model into the questionnaire is the top next step.
- **Dataset size & geography:** 541 patients from Kerala — results may not generalise to other populations.
- **Cycle prediction** uses the mean of recorded cycle lengths; more advanced time-series methods would improve accuracy for irregular cycles.
- **Storage:** daily logs live in Streamlit session state (reset on refresh); the API's log endpoint is a demo and does not persist data.
- **Wearable:** the live stream in the app is simulated; hardware integration is a prototype.
- **Doctor directory:** static list for Bengaluru — verify contact details and ratings before use.

**Roadmap ideas**

- [ ] Persistent storage (SQLite / PostgreSQL) with user accounts
- [ ] Use the full trained model in the PCOS questionnaire with SHAP-based explanations
- [ ] Time-series cycle forecasting
- [ ] Real Bluetooth / Wi-Fi wearable pipeline
- [ ] Perimenopause-specific symptom tracking and insights
- [ ] Multilingual support
- [ ] `requirements.txt` and Docker setup

---

## 👥 Team

**Department of Artificial Intelligence & Machine Learning — Global Academy of Technology, Bengaluru**

- Aditi S Yaranal — [@Aditisy](https://github.com/Aditisy)
- Sunidhi S Mayekar — [@sunidhi76](https://github.com/sunidhi76)
- Sai Sirisha K S — [@saisirish891](https://github.com/saisirish891)

**Guide:** Prof. Vijaya Dalawai

---

## 📄 License

No license has been specified yet. Add a `LICENSE` file (e.g., MIT) before accepting contributions or public reuse.

---

<p align="center">Made with 💜 for women's health awareness</p>
