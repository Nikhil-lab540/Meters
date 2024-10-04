import streamlit as st
import os
import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()

# Set up API
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

def upload_to_gemini(path, mime_type=None):
    """Uploads the given file to Gemini."""
    file = genai.upload_file(path, mime_type=mime_type)
    return file

# Create the model configuration with explicit JSON response schema
generation_config = {
    "temperature": 0,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "application/json",
    "response_schema": {
        "type": "object",
        "properties": {
            "measurement": {
                "type": "number"
            },
            "units": {
                "type": "string"
            },
            "is_meter": {  
                "type": "boolean"
            }
        },
        "required": ["measurement", "units", "is_meter"]
    }
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)

st.title("Digital Meters Image Analyzer(Gemini 1.5 Flash)")

# File uploader
uploaded_file = st.file_uploader("Choose an image file", type=['jpg', 'jpeg', 'png'])
with st.spinner():
    if uploaded_file is not None:
        # Save the uploaded file locally
        with open("temp_image.jpg", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Upload to Gemini and start chat
        user_input = """You are an assistant tasked with analyzing images specifically of digital and mechanical meters to 
        identify and record the primary measurement displayed. You should only analyze and return measurements if the image 
        contains a meter. If the user sends images of other objects, such as animals, airplanes, or anything unrelated to meters, 
        no measurements should be returned."""
        gemini_file = upload_to_gemini("temp_image.jpg", mime_type="image/jpeg")
        chat_session = model.start_chat(
            history=[
                {"role": "user", "parts": [gemini_file, user_input]},
            ]
        )

        response = chat_session.send_message(user_input)
        
        import json
        response_data = json.loads(response.text)
        
        if response_data.get("is_meter"):  # Check if the image contains a meter
            measurement = response_data.get("measurement")
            units = response_data.get("units")
            
            # Display extracted information in the specified format
            st.subheader("Extracted Information")
            st.write(f"Measurement: {measurement}")
            st.write(f"Units: {units}")
        else:
            st.warning("The uploaded image does not contain a digital or mechanical meter. Please upload an appropriate image.")
