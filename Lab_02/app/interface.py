import streamlit as st
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
from keras.models import load_model
from keras.preprocessing.image import img_to_array


# Load and detect animals
# def detect_animals(image):
#     model = YOLO("runs/detect/train7/weights/best.pt")
#     results = model.predict(image)
#     result = results[0]
#     output = []
#     for box in result.boxes:
#         x1, y1, x2, y2 = [
#           round(x) for x in box.xyxy[0].tolist()
#         ]
#         class_id = box.cls[0].item()
#         prob = round(box.conf[0].item(), 2)
#         output.append([
#           x1, y1, x2, y2, result.names[class_id], prob
#         ])
#     return output

def detect_animals(image):
    model = YOLO("runs/detect/train7/weights/best.pt")
    results = model.predict(image)
    result = results[0]
    
    # Convert PIL Image to draw context
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    # Define color map
    color_map = {0: "red", 1: "green", 2: "blue"} 

    for box in result.boxes:
        x1, y1, x2, y2 = [round(x) for x in box.xyxy[0].tolist()]
        class_id = box.cls[0].item()
        prob = round(box.conf[0].item(), 2)
        label = f"{result.names[class_id]}: {prob}"

        # Get color for class
        color = color_map.get(class_id, "white")  # Default to white if class ID not in color map

        # Draw bounding box and label
        draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=2)
        draw.text((x1, y1 - 10), label, fill=color, font=font)

    return image


st.title('Animal Detection App')

st.write("This app can detect monkeys, cats, and dogs.")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

# if uploaded_file is not None:
#     image = Image.open(uploaded_file)
#     st.image(image, caption='Uploaded Image.', use_column_width=True)
#     st.write("")
#     # st.write("Classifying...")
#     processed_image = detect_animals(image)
#     st.image(processed_image, caption='Processed Image.', use_column_width=True)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.write("Detection result.")

    processed_image = image.copy()  # Create a copy of the original image
    processed_image = detect_animals(processed_image)

    # Create two columns
    col1, col2 = st.columns(2)

    # Display the original image in the first column
    col1.image(image, caption='Uploaded Image.', use_column_width=True)

    # Display the processed image in the second column
    col2.image(processed_image, caption='Processed Image.', use_column_width=True)