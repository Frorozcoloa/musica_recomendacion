import streamlit as st
import librosa
from src.preprosecing import preprosecing
st.write("Clasificación de canciones")
#st.set_page_config(page_title="Upload Music", page_icon=":musical_note:", layout="wide")

def main():
    uploaded_file = st.file_uploader("Choose a music file", type=["mp3", "wav"])

    if uploaded_file is not None:
        uploaded_file, features = preprosecing(uploaded_file)
        st.audio(uploaded_file, format='audio/wav')
        st.success("30 secs audio snippet")
        st.success("File uploaded successfully")
        st.write("This is the features from the audio")
        st.write(features)
    else:
        st.warning("Please upload a file of type: mp3, wav")

if __name__ == "__main__":
    main()