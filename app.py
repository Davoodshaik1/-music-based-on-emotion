import streamlit as st
import cv2
import numpy as np
from deepface import DeepFace
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# Initialize Spotify API client
client_credentials_manager = SpotifyClientCredentials(client_id='6c09886ab4164d2bb19694bc91394028', client_secret='3d2d7c46d0f54575ac441b92075a960e')
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# Function to fetch tracks based on emotion and language
def fetch_tracks(emotion, language):
    # Modify the search query to include both emotion and language
    query = f'emotion:{emotion} language:{language}'
    results = sp.search(q=query, type='track', limit=50)
    tracks = [(track['name'], track['artists'][0]['name'], track['external_urls']['spotify']) for track in results['tracks']['items']]
    return tracks

def analyze_frame(frame):
    # Analyze facial emotion
    result = DeepFace.analyze(img_path=frame, actions=['emotion'], enforce_detection=False)
    
    # Get the first result (assuming only one face is detected)
    first_result = result[0] if result else None
    
    if first_result:
        emotion = first_result['dominant_emotion']
        
        # Fetch tracks based on detected emotion and language
        language = st.session_state.language_selection  # Get user-selected language
        tracks = fetch_tracks(emotion, language)
        
        return emotion, tracks
    else:
        return None, None

# Initialize an empty list to store played tracks
played_tracks = []

def facesentiment():
    cap = cv2.VideoCapture(0)
    stframe = st.image([])

    while True:
        ret, frame = cap.read()

        emotion, tracks = analyze_frame(frame)
        
        if emotion:
            cv2.putText(frame, f"Emotion: {emotion}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame_rgb, channels="RGB")

        
        if tracks:
            # Sort tracks alphabetically by track name
            tracks = sorted(tracks, key=lambda x: x[0])
            for i, track in enumerate(tracks):
                # Check if the track has been played before
                if track[0] not in played_tracks:
                    st.write(f"{i+1}. **{track[0]}** by {track[1]}")
                    st.write(f"Listen on Spotify: [{track[0]}]({track[2]})")
                    # Add the track to the played list
                    played_tracks.append(track[0])
                

    cap.release()
    cv2.destroyAllWindows()

def main():
    st.title("Emotion-Based Music Recommendation")

    # Select language
    st.session_state.language_selection = st.selectbox("Select your preferred language:", ["English", "Telugu", "Tamil"])

    activities = ["Webcam Face Detection", "About"]
    choice = st.sidebar.selectbox("Select Activity", activities)
    st.sidebar.markdown(
        """ Developed by Davood Shaik 
            Email : davoodshaik43@gamil.com
        """)
    if choice == "Webcam Face Detection":
        html_temp_home1 = """<div style="background-color:#6D7B8D;padding:10px">
                                            <h4 style="color:white;text-align:center;">
                                            Real time face emotion recognition of webcam feed using OpenCV, DeepFace and Streamlit.</h4>
                                            </div>
                                            </br>"""
        st.markdown(html_temp_home1, unsafe_allow_html=True)
        facesentiment()

    elif choice == "About":
        st.subheader("About this app")

        html_temp4 = """
                                     		<div style="background-color:#98AFC7;padding:10px">
                                     		<h4 style="color:white;text-align:center;">This Application is developed by Shrimanta Satpati. </h4>
                                     		<h4 style="color:white;text-align:center;">Thanks for Visiting</h4>
                                     		</div>
                                     		<br></br>
                                     		<br></br>"""

        st.markdown(html_temp4, unsafe_allow_html=True)

    else:
        pass

if __name__ == "__main__":
    main()
