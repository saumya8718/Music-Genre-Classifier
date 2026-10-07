import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Music Genre Classifier", page_icon="🎵")

@st.cache_resource
def load_model():
    return joblib.load("models/genre_model.pkl")

@st.cache_data
def load_data():
    return pd.read_csv("data/features_30_sec.csv")

saved = load_model()
model, columns = saved["model"], saved["columns"]
df = load_data()

st.title("🎵 Music Genre Classifier")
st.write("Pick a song clip from the dataset. The model looks only at its audio "
         "measurements (tempo, brightness, loudness...) and guesses the genre.")

song = st.selectbox("Choose a song", df["filename"].tolist())
row = df[df["filename"] == song]

if st.button("Predict genre"):
    features = row[columns]
    probs = model.predict_proba(features)[0]
    prediction = model.classes_[probs.argmax()]
    actual = row["label"].iloc[0]

    st.subheader(f"Predicted: {prediction}")
    st.write(f"Actual genre: **{actual}**")
    if prediction == actual:
        st.success("Correct!")
    else:
        st.error("The model got this one wrong.")

    st.bar_chart(pd.Series(probs, index=model.classes_, name="Confidence"))