import streamlit as st
import base64
import torch
from transformers import pipeline
import scipy.io.wavfile
import os
from io import BytesIO

# Set page configuration
st.set_page_config(page_title="Symphony Smith", layout="wide")

# Cache the model using @st.cache_resource
@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return pipeline("text-to-audio", model="facebook/musicgen-small", device=device)

# Load the model only once
model = load_model()

# Initialize session state
if "page" not in st.session_state:
    st.session_state.update({
        "page": "WelcomePage",
        "audio_data": None,
        "sampling_rate": None,
        "prompt_history": []
    })

# Function to handle image encoding
@st.cache_data
def get_encoded_image(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    else:
        st.error(f"Image file '{image_path}' not found.")
        return None

# Function to set background image
def set_background(image_path):
    encoded_image = get_encoded_image(image_path)
    if encoded_image:
        st.markdown(f"""
            <style>
                .stApp {{
                    background-image: url("data:image/jpeg;base64,{encoded_image}");
                    background-size: cover;
                    background-repeat: no-repeat;
                    background-position: center;
                }}
            </style>
        """, unsafe_allow_html=True)

# Function to generate music
@st.cache_data
def generate_music(prompt, duration=5, temperature=0.8):
    try:
        music = model(prompt, forward_params={"do_sample": True, "max_new_tokens": duration * 50, "temperature": temperature})
        return music["audio"], music["sampling_rate"]
    except Exception as e:
        st.error(f"An error occurred while generating music: {str(e)}")
        return None, None

# Welcome page function
def welcome_page():
    set_background('ssbg.jpg')
    st.markdown("<h1 style='color: white;'>Welcome to Symphony Smith</h1>", unsafe_allow_html=True)
    if st.button("Go Ahead"):
        st.session_state.page = "MainPage"

# Main page function
def main_page():
    set_background('ssbg.jpg')
    st.title("Symphony Smith")
    
    prompt = st.text_input("Enter your music prompt:")
    duration = st.slider("Duration (seconds)", 5, 30, 10)
    temperature = st.slider("Creativity (temperature)", 0.1, 1.0, 0.8)

    if st.button("Generate Music"):
        if prompt:
            with st.spinner("Generating music..."):
                audio_data, sampling_rate = generate_music(prompt, duration, temperature)
                if audio_data is not None:
                    st.session_state.audio_data = audio_data
                    st.session_state.sampling_rate = sampling_rate
                    st.session_state.prompt_history.append(prompt)
                    st.success("Music generated successfully!")
        else:
            st.warning("Please enter a prompt to generate music.")

    # Display the audio player if audio data is available
    if st.session_state.audio_data is not None and st.session_state.sampling_rate is not None:
        # Create a BytesIO object to store the WAV file
        wav_bytes = BytesIO()
        scipy.io.wavfile.write(wav_bytes, rate=st.session_state.sampling_rate, data=st.session_state.audio_data)
        wav_bytes.seek(0)
        
        st.audio(wav_bytes, format="audio/wav")
        st.download_button(
            label="Download Music",
            data=wav_bytes,
            file_name="generated_music.wav",
            mime="audio/wav"
        )

    if st.button("Back"):
        st.session_state.page = "WelcomePage"

    if st.button("Thank You"):
        st.session_state.page = "ThankYouPage"

# Thank you page function
def thank_you_page():
    set_background('ssbg.jpg')
    st.markdown("<h1 style='color: white;'>Thank You for Using Symphony Smith!</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='color: white;'>Your Music Journey:</h2>", unsafe_allow_html=True)
    for i, prompt in enumerate(st.session_state.prompt_history, 1):
        st.markdown(f"<p style='color: white;'>{i}. {prompt}</p>", unsafe_allow_html=True)
    
    if st.button("Start Over"):
        st.session_state.page = "WelcomePage"
        st.session_state.prompt_history = []

# Page navigation
pages = {
    "WelcomePage": welcome_page,
    "MainPage": main_page,
    "ThankYouPage": thank_you_page
}

# Run the appropriate page function
pages[st.session_state.page]()