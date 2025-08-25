 

import streamlit as st
import cv2
import numpy as np
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# ===============================
# Spotify API setup
# ===============================
client_id = "6c09886ab4164d2bb19694bc91394028"
client_secret = "d2d7c46d0f54575ac441b92075a960e"
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id=client_id, client_secret=client_secret
))

# ===============================
# Dummy Emotion Detector (replace with your model later)
# ===============================
def analyze_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml").detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    if len(faces) == 0:
        return None, []

    # Example: Always return "happy" (replace with ML model output)
    emotion = "happy"

    # Get songs from Spotify
    results = sp.search(q=emotion, limit=5, type="track")
    tracks = [(t["name"], t["artists"][0]["name"], t["external_urls"]["spotify"]) for t in results["tracks"]["items"]]

    return emotion, tracks


# ===============================
# Streamlit App UI
# ===============================
st.title("🎵 Emotion-Based Music Recommendation")

mode = st.radio("Choose input method:", ["Upload Image (Cloud Safe)", "Use Webcam (Local Only)"])

# --- File Upload (Cloud Safe) ---
if mode == "Upload Image (Cloud Safe)":
    uploaded_file = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)

        emotion, tracks = analyze_frame(frame)

        if emotion:
            st.success(f"Detected Emotion: {emotion}")
            for i, (song, artist, url) in enumerate(tracks, 1):
                st.write(f"{i}. **{song}** by {artist}")
                st.markdown(f"[▶️ Listen on Spotify]({url})")
        else:
            st.warning("No face detected. Try another image.")

# --- Webcam (Local Only) ---
elif mode == "Use Webcam (Local Only)":
    st.warning("⚠️ Webcam only works on your computer, not on Streamlit Cloud.")
    run = st.checkbox("Start Webcam")

    if run:
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            st.error("Could not access webcam. Please run locally with a camera.")
        else:
            ret, frame = cap.read()
            cap.release()

            if ret:
                st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

                emotion, tracks = analyze_frame(frame)

                if emotion:
                    st.success(f"Detected Emotion: {emotion}")
                    for i, (song, artist, url) in enumerate(tracks, 1):
                        st.write(f"{i}. **{song}** by {artist}")
                        st.markdown(f"[▶️ Listen on Spotify]({url})")
                else:
                    st.warning("No face detected. Try again.")
