import streamlit as st
import librosa
import numpy as np
from xgboost import XGBClassifier
import tempfile
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Load the pre-trained model
@st.cache_resource
def get_model(model_path="xgb_model.json"):
    classifier = XGBClassifier()
    classifier.load_model(model_path)
    return classifier

emotion_model = get_model()

# Define emotion categories
emotion_classes = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprise']

# Feature extraction using MFCC, Chroma, and Mel-spectrogram
def generate_features(audio_file, mfcc_count=40, mel_count=128):
    signal, rate = librosa.load(audio_file, sr=None)
    
    # Normalize to 16kHz
    if rate != 16000:
        signal = librosa.resample(signal, orig_sr=rate, target_sr=16000)
        rate = 16000

    mfcc_feat = np.mean(librosa.feature.mfcc(y=signal, sr=rate, n_mfcc=mfcc_count).T, axis=0)
    chroma_feat = np.mean(librosa.feature.chroma_stft(y=signal, sr=rate).T, axis=0)
    mel_feat = np.mean(librosa.feature.melspectrogram(y=signal, sr=rate, n_mels=mel_count).T, axis=0)

    return np.hstack([mfcc_feat, chroma_feat, mel_feat])

# Perform prediction
def detect_emotion(audio_file_path):
    audio_features = generate_features(audio_file_path)
    audio_features = audio_features.reshape(1, -1)
    predicted_index = emotion_model.predict(audio_features)[0]
    return emotion_classes[predicted_index]

# ---------------------------
# Streamlit Interface
# ---------------------------
st.set_page_config(page_title="🎧 XGBoost Audio Emotion Classifier", layout="centered")
st.title("🔊 Voice Emotion Recognition using XGBoost")
st.write("Upload a `.wav` audio file to classify the emotional tone.")

# Upload section
audio_input = st.file_uploader("Choose a .wav file", type=["wav"])

if audio_input:
    st.audio(audio_input, format='audio/wav')

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(audio_input.read())
        audio_path = temp_audio.name

    with st.spinner("Processing audio and predicting emotion..."):
        try:
            detected_emotion = detect_emotion(audio_path)
            st.success(f"🧠 Predicted Emotion: **{detected_emotion.upper()}**")
        except Exception as err:
            st.error(f"⚠️ An error occurred: {err}")
