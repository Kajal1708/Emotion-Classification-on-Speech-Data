# Emotion-Classification-on-Speech-Data

This project presents a real-time speech emotion recognition system using machine learning techniques, particularly XGBoost. The application allows users to upload `.wav` files and receive instant emotion predictions. It extracts rich audio features and delivers results via a lightweight Streamlit interface.

---

##  Key Features

- **Fast and Accurate Emotion Detection** powered by XGBoost  
- **User-Friendly Interface** with support for real-time predictions  
- **Robust Audio Feature Extraction** using MFCC, Chroma, and Mel Spectrogram  
- **Supports Standard Audio Format:** `.wav` files  
- **Web Deployed:** Access the model anytime through a hosted Streamlit application  

---

##  Recognized Emotions

The system identifies 8 key emotional states:

| Emotion   | Symbol |
|-----------|--------|
| Angry     | 😠     |
| Calm      | 😌     |
| Disgust   | 🤢     |
| Fearful   | 😨     |
| Happy     | 😄     |
| Neutral   | 😐     |
| Sad       | 😢     |
| Surprise  | 😲     |

---

##  Model Overview

- **Classifier:** XGBoost (`XGBClassifier`)  
- **Sampling Rate:** 16 kHz for uniform audio analysis  
- **Input:** Preprocessed `.wav` files  
- **Features Used:**
  - MFCC (Mel-Frequency Cepstral Coefficients)
  - Chroma Features
  - Mel Spectrogram  
- **Model File:** `xgb_model.json`



## Project Structure

README.md
app.py
emotion_classification.ipynb
requirements.txt
xgb_model.json


## 📊 Performance Summary

**XGBoost Classification Report**

| Class     | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| Angry     | 0.855     | 0.891  | 0.87     | 38      |
| Calm      | 0.949     | 0.969  | 0.96     | 38      |
| Disgust   | 0.749     | 0.943  | 0.83     | 38      |
| Fearful   | 0.880     | 0.898  | 0.89     | 39      |
| Happy     | 0.827     | 0.744  | 0.78     | 39      |
| Neutral   | 0.966     | 0.759  | 0.85     | 19      |
| Sad       | 0.739     | 0.680  | 0.71     | 38      |
| Surprise  | 0.763     | 0.718  | 0.74     | 39      |
|           |           |        |          |         |
| **Accuracy**     |        |        | **0.83**  |         |
| **Macro Avg**    | 0.841  | 0.825  | 0.83     | 288     |
| **Weighted Avg** | 0.833  | 0.829  | 0.83     | 288     |

---

## Visualization

![Coefficient Matrix](https://github.com/user-attachments/assets/f168cd39-6732-4acc-a3d8-2ef6269bd671)

---

## Live Demo

Try out the live emotion recognition app here:  
👉 [Streamlit Web Application](https://emotion-classification-on-speech-data-hkmoc3ngvfhkk47jsmxu2u.streamlit.app/)

---
