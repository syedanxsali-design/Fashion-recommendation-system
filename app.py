import streamlit as st
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import os
from PIL import Image

Image.init()
SUPPORTED_IMAGE_EXTENSIONS = sorted(
    ext
    for ext in {ext.lstrip(".").lower() for ext in Image.registered_extensions()}
    if ext
)

# Hosted data sources
METADADOS_URL = "https://github.com/syedanxsali-design/Fashion-recommendation-system/releases/download/metadados.csv/metadados.csv"
EMBEDDINGS_URL = "https://github.com/syedanxsali-design/Fashion-recommendation-system/releases/download/embeddings.csv/embeddings.csv"
DF_SAMPLE_URL = "https://github.com/syedanxsali-design/Fashion-recommendation-system/releases/download/df_sample.csv/df_sample.csv"

# Load data from hosted CSV files
metadados = pd.read_csv(METADADOS_URL)
embeddings_df = pd.read_csv(EMBEDDINGS_URL)
df_sample = pd.read_csv(DF_SAMPLE_URL)

# Keep only numeric embedding columns and ignore common index-like columns.
embeddings_numeric = embeddings_df.select_dtypes(include=[np.number]).drop(
    columns=["Unnamed: 0", "index", "id"],
    errors="ignore",
)
embeddings_array = embeddings_numeric.to_numpy(dtype=np.float32)

# Local image folder relative to this app file.
image_directory = os.path.join(os.path.dirname(__file__), "images")

# Load ResNet50 model for feature extraction
model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

# Function to extract features for the uploaded image
def extract_image_features(img):
    img = img.convert("RGB")
    img = img.resize((224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    features = model.predict(img_data)
    return features.flatten()

# Function to get top-N similar items
def get_recommendations(uploaded_image, top_n=5):
    uploaded_image_features = extract_image_features(uploaded_image)
    similarity_scores = cosine_similarity(uploaded_image_features.reshape(1, -1), embeddings_array).flatten()
    similar_indices = np.argsort(similarity_scores)[-top_n:][::-1]
    recommended_items = df_sample.iloc[similar_indices]
    return recommended_items

# Streamlit app
st.title("Fashion Recommendation System")
st.write("Upload an image, and we'll show you similar items from our collection!")

uploaded_file = st.file_uploader("Choose an image...", type=SUPPORTED_IMAGE_EXTENSIONS)

if uploaded_file is not None:
    try:
        uploaded_image = Image.open(uploaded_file)
        uploaded_image.load()
    except (OSError, ValueError):
        uploaded_image = None
        st.error("Could not read that file as an image. Try a different format.")

    if uploaded_image is not None:
        # Display the uploaded image
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

        # Get recommendations
        with st.spinner("Finding similar items..."):
            recommendations = get_recommendations(uploaded_image)

        # Display recommendations
        st.write("### Recommended Items:")
        cols = st.columns(3)  # Create 5 columns for the recommendations

        for i, (_, row) in enumerate(recommendations.iterrows()):
            img_path = os.path.join(image_directory, row["image"])
            recommended_image = Image.open(img_path)

            with cols[i % len(cols)]:  # Place the image in the appropriate column
                st.image(
                    recommended_image,
                    caption=row["productDisplayName"],
                    use_column_width="auto",
                )  # Smaller images side by side
