import streamlit as st
import base64
import os
from langchain_openai import ChatOpenAI
import cv2
from io import BytesIO
from PIL import Image
import numpy as np
from langchain_core.messages import HumanMessage
import json
from dotenv import load_dotenv
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


# Function to preprocess and encode images
def preprocess_image(image):
    """Apply preprocessing that includes adaptive thresholding and contour detection."""
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Find contours and filter based on area size, then draw rectangles
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) > 100:  # adjust this threshold as needed
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # draws a rectangle on original image

    # Convert cv2 image back to PIL image for consistent encoding
    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    buffer = BytesIO()
    img_pil.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

# Function to summarize the image (this should ideally make a call to an API, here it's mocked)
def image_summarize(img_base64, prompt):
    """Make image summary using a chat model and invoke function definition."""

    # Define the function that will be called
    function_definition = {
        "name": "extract_meter_measurement",
        "description": "Extracts the flow meter reading and units from an image.",
        "parameters": {
            "type": "object",
            "properties": {
                "measurement": {
                    "type": "string",
                    "description": "The numeric reading from the flow meter."
                },
                "units": {
                    "type": "string",
                    "description": "The units of measurement (e.g., L/min, GPM)."
                }
            },
            "required": ["measurement", "units"]
        }
    }

    # Initialize the chat model (ensure ChatOpenAI and related classes are properly imported)
    chat = ChatOpenAI(model="gpt-4o-mini", max_tokens=1024)

    # Define the human message with both prompt and the base64 image
    msg = chat.invoke(
        [
            HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
                ]
            )
        ],
        functions=[function_definition]
    )
    
    if msg and msg.additional_kwargs and msg.additional_kwargs.get("function_call"):
      function_call = msg.additional_kwargs["function_call"]
      arguments = function_call.get("arguments")
    else:
        return "The uploaded image does not contain a digital or mechanical meter. Please upload an appropriate image."
    if arguments:
      import json
      arguments_dict = json.loads(arguments)
    else:
        return "The uploaded image does not contain a digital or mechanical meter. Please upload an appropriate image." 
    return arguments_dict


# Streamlit app
st.title("Digital Meters Image Analyzer(gpt-4o-mini)")

# Prompt the user to upload one or more images
uploaded_files = st.file_uploader("Upload flow meter images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

# If files are uploaded, process and show results
if uploaded_files:
    prompt = """You are an assistant tasked with analyzing images of various digital and mechanical meters to 
    identify and record the main measurement displayed."""
    
    for uploaded_file in uploaded_files:
        # Open the uploaded image file
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)  # Convert to cv2 image
        
        # Show the original image
        #st.image(image, caption=f"Uploaded Image: {uploaded_file.name}", use_column_width=True)
        
        # Preprocess and encode image
        with st.spinner():
            img_base64 = preprocess_image(image)
            
            # Call the image summarization function
            result = image_summarize(img_base64, prompt)
            
            # Check if the result contains a measurement or if it's a string message
            if isinstance(result, dict) and 'measurement' in result:
                # Display the results if measurement-related data is found
                st.write("### Extracted Information")
                st.write(f"**Measurement**: {result['measurement']}")
                st.write(f"**Units**: {result['units']}")
                # st.write(f"**Confidence Score**: {result['confidence_score']}")
            elif isinstance(result, str):
                # Display a warning if the result is a string (likely an error message)
                st.warning(result)
            else:
                # Handle other cases where the result is neither a dict nor a string
                st.warning("No valid data found. Please check the image or try again.")
