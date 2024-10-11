import os
import json
import cv2
from PIL import Image
import io
import tempfile
from PIL import Image, ImageDraw
import soundfile as sf  # For reading audio files
import numpy as np  # For audio processing
from streamlit_option_menu import option_menu

import google.generativeai as genai

# working directory path
working_dir = os.path.dirname(os.path.abspath(__file__))

# path of config_data file
config_file_path = f"{working_dir}/config.json"
config_data = json.load(open("config.json"))

# loading the GOOGLE_API_KEY
GOOGLE_API_KEY = config_data["GOOGLE_API_KEY"]

# configuring google.generativeai with API key
genai.configure(api_key = GOOGLE_API_KEY)


def load_gemini_pro_model():
    gemini_pro_model = genai.GenerativeModel("gemini-pro")
    return gemini_pro_model


# get response from Gemini-Pro-Vision model - image/text to text
# def gemini_pro_vision_response(prompt, image):
#     gemini_pro_vision_model = genai.GenerativeModel("gemini-pro-vision")
#     response = gemini_pro_vision_model.generate_content([prompt, image])
#     result = response.text
#     return result
def generate_gemini_image_caption_response(image_file):
    """Generates a caption for the given image using the gemini-1.5-flash model.

    Args:
        image_file: An uploaded image file (in-memory).

    Returns:
        str: The generated caption for the image.
    """
    try:
        # Open and convert the image to the appropriate format
        image = Image.open(image_file)

        # Use gemini-1.5-flash model for image captioning
        gemini_flash_model = genai.GenerativeModel("models/gemini-1.5-flash")

        # Create a placeholder description prompt for image captioning
        description_prompt = "Generate a descriptive caption for this image."

        # Send the image and prompt to the model for caption generation
        response = gemini_flash_model.generate_content([description_prompt, image])

        # Access the text from the response
        caption = response.text  # Changed this line to access the correct attribute

        return caption

    except Exception as e:
        return f"Error generating caption: {e}"



# get response from embeddings model - text to embeddings
def embeddings_model_response(input_text):
    embedding_model = "models/embedding-001"
    embedding = genai.embed_content(model=embedding_model,
                                    content=input_text,
                                    task_type="retrieval_document")
    embedding_list = embedding["embedding"]
    return embedding_list


# get response from Gemini-Pro model - text to text
def gemini_pro_response(user_prompt):
    gemini_pro_model = genai.GenerativeModel("gemini-pro")
    response = gemini_pro_model.generate_content(user_prompt)
    result = response.text
    return result

#translation response
def translate_text(input_text, target_language):
    """Translates text from English to the specified target language.

    Args:
        input_text: The text to be translated.
        target_language: The target language for translation.

    Returns:
        str: The translated text.
    """
    try:
        # Create the prompt for translation
        translation_prompt = f"Translate the following text from English to {target_language}: {input_text}"

        # Use the appropriate Gemini model for text translation
        gemini_model = genai.GenerativeModel("models/gemini-1.5-flash")  # Replace with the correct translation model

        # Call the model for translation
        response = gemini_model.generate_content(translation_prompt)

        if hasattr(response, 'text'):
            return response.text.strip()  # Return the translated text
        else:
            return None

    except Exception as e:
        return f"Error during translation: {e}"


# result = gemini_pro_response("What is Machine Learning")
# print(result)
# print("-"*50)
#
#
# image = Image.open("test_image.png")
# result = gemini_pro_vision_response("Write a short caption for this image", image)
# print(result)
# print("-"*50)
#
#
# result = embeddings_model_response("Machine Learning is a subset of Artificial Intelligence")
# print(result)