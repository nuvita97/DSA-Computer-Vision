import streamlit as st
from PIL import Image

# Load your trained model
# model = load_model('path_to_your_model')

def detect_animals(image):
    # Preprocess your image and make predictions
    # return model.predict(image)

    # Dummy function for demonstration - replace with your own prediction function
    return "Prediction result"

st.title('Animal Detection App')

st.write("This app can detect monkeys, cats, and dogs.")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    label = detect_animals(image)
    st.write('%s' % label)