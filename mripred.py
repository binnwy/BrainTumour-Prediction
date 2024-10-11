import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

# Load the pre-trained model
model = load_model('my_model.keras')

# Function to preprocess the uploaded image
def preprocess_image(image, target_size=(231, 283)):
    img_array = img_to_array(image) / 255.0  # Convert the image to array and normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Set page config
st.set_page_config(page_title="Brain Tumor Detection App", page_icon="🧠", layout="centered")

# App title
st.title("Brain Tumor Detection App 🧠")
st.markdown("""
Upload a brain MRI scan, and the model will predict whether a tumor is present.
""")

# Image upload section
uploaded_file = st.file_uploader("Upload a brain MRI image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Load and display the uploaded image
    image = load_img(uploaded_file, target_size=(231, 283))  # Resize to model input size
    st.image(image, caption='Uploaded Brain MRI.', use_column_width=True)

    # Add a button for classification
    if st.button("Classify"):
        with st.spinner("Classifying... Please wait."):
            preprocessed_image = preprocess_image(image)
            prediction = model.predict(preprocessed_image)

            # Print the raw prediction for debugging
            st.write(f"Raw prediction output: {prediction}")

            # Determine prediction threshold
            if prediction[0][0] > 0.388:
                st.error("The model predicts: Brain Tumor Detected")
            else:
                st.success("The model predicts: No Brain Tumor Detected")

    # Plotting the image
    st.markdown("### Image Preview")
    plt.imshow(image)
    plt.axis('off')  # Hide axes
    st.pyplot(plt)

# Disclaimer
st.markdown("""
### Disclaimer:
This application is for educational purposes only. Please consult a healthcare professional for medical advice.
""")
