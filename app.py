import streamlit as st
import librosa
import numpy as np
from tensorflow.keras.models import load_model
import os
import glob
from sklearn.preprocessing import LabelEncoder

# Load the model
MODEL_PATH = 'model.keras'
model = load_model(MODEL_PATH)

# Load and encode labels
data = []
labels = []
dataset_path = r"TESS Toronto emotional speech set data"

for folder_name in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, folder_name)
    for file_path in glob.glob(os.path.join(folder_path, '*.wav')):
        labels.append(folder_name)

label_encoder = LabelEncoder()
label_encoder.fit(labels)

# Emotion mapping for TESS dataset
emotion_mapping = {
    'YAF_angry': 'ANGRY',
    'YAF_disgust': 'DISGUST',
    'YAF_fear': 'FEAR',
    'YAF_happy': 'HAPPY',
    'YAF_neutral': 'NEUTRAL',
    'YAF_pleasant_surprised': 'SURPRISED',
    'YAF_sad': 'SAD',
    'OAF_angry': 'ANGRY',
    'OAF_disgust': 'DISGUST',
    'OAF_Fear': 'FEAR',
    'OAF_happy': 'HAPPY',
    'OAF_neutral': 'NEUTRAL',
    'OAF_Pleasant_surprise': 'SURPRISED',
    'OAF_Sad': 'SAD',
}

# Feature extraction
def extract_features(file_path):
    audio, sr = librosa.load(file_path, res_type='kaiser_fast')
    features = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13).T, axis=0)
    return features

# Emotion prediction
def predict_emotion(audio_file):
    # Load audio and extract features
    audio, sr = librosa.load(audio_file, sr=None)
    features = extract_features(audio_file)
    features = features[np.newaxis, np.newaxis, :]  # Reshape for model

    # Predict probabilities
    predicted_probabilities = model.predict(features)
    predicted_probabilities_nn=label_encoder.classes_[range(len(predicted_probabilities[0]))]
    predicted_probabilities_n=[]
    for i in predicted_probabilities_nn :
        p=emotion_mapping.get(i)
        predicted_probabilities_n.append(p)
    for i in range(len(predicted_probabilities_n)):
        for j in range(i+1,len(predicted_probabilities_n)):
            if predicted_probabilities_n[i]==predicted_probabilities_n[j] :
                predicted_probabilities[0][i]=predicted_probabilities[0][i]+predicted_probabilities[0][j]
                predicted_probabilities[0][j]=predicted_probabilities[0][i]
    dictionnary = dict(zip(predicted_probabilities_n,predicted_probabilities[0]))
    predicted_label_index = np.argmax(predicted_probabilities)
    predicted_emotion = label_encoder.classes_[predicted_label_index]
    recognizable_emotion = emotion_mapping.get(predicted_emotion, "UNKNOWN")
    
    return recognizable_emotion, dictionnary

# Streamlit App
st.title("Analyse des émotions à partir d'un fichier audio 🎵")
st.write("Chargez un fichier audio pour prédire l'émotion.")

uploaded_file = st.file_uploader("Choisir un fichier audio", type=["wav", "mp3"])

if uploaded_file is not None:
    # Save temporary file
    with open("temp_audio.wav", "wb") as f:
        f.write(uploaded_file.read())
    
    st.audio("temp_audio.wav", format="audio/wav")

    # Predict emotion
    emotion, probabilities = predict_emotion("temp_audio.wav")

    # Display results
    st.write(f"L'émotion prédite est : **{emotion}**")
    st.bar_chart(probabilities)
