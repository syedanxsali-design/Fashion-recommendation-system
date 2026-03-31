# Fashion Recommendation System

An intelligent, image-based fashion recommendation web application built with Streamlit, TensorFlow (ResNet50), and Scikit-Learn. 

Simply upload a picture of a fashion product, and the application will act as a visual search engine, retrieving visually similar items from a catalog using precomputed deep learning embeddings.

## ⚠️ Important Note on Large Files (GitHub Releases)

Due to GitHub's repository size limits, the large data files required to run this project are **not** stored directly in the main repository tree. Instead, they are hosted in the **[Releases](../../releases)** section of this repository.

* **Dynamically Fetched Files:** The core dataset files (`metadados.csv`, `embeddings.csv`, and `df_sample.csv`) are automatically downloaded at runtime by `app.py` via their direct release URLs. You do not need to download these manually.
* **Local Images Directory:** To display the recommended items, the app requires a local `images` folder. Because of the large volume of product images, you must manually download the images archive from the Releases section and extract it into the project root before running the app.

## Overview & Architecture

This project performs a visual similarity search for fashion products.

- **Query Input:** User-uploaded image (Supports JPG, PNG, WEBP, JPEG, etc.).
- **Feature Extractor:** Pre-trained ResNet50 model (ImageNet weights) with global average pooling.
- **Retrieval Engine:** Cosine similarity scoring against precomputed catalog embeddings.
- **Output:** Top-N visually similar products displayed with their product names and thumbnail images.

### How It Works Behind the Scenes
1. **Data Initialization:** `app.py` fetches the catalog metadata and the embedding matrix directly from the hosted CSV files in GitHub Releases.
2. **Matrix Prep:** It filters the `embeddings.csv` to keep only numeric columns, forming a 2D float32 embedding matrix.
3. **Model Loading:** Initializes ResNet50 to extract a fixed-length feature vector.
4. **Image Processing:** When a user uploads an image, it is resized to 224x224 and preprocessed according to ResNet50 standards.
5. **Similarity Search:** The app computes the cosine similarity between the uploaded image's feature vector and all embeddings in the catalog.
6. **Result Rendering:** The top matches are selected, cross-referenced with `df_sample.csv`, and displayed using the local `images` directory.

## Project Structure

```text
.
|-- app.py                # Main Streamlit application script
|-- requirements.txt      # Python dependencies
|-- README.md             # Project documentation
`-- images/               # MUST BE DOWNLOADED FROM RELEASES (Contains product thumbnails)
```

## Requirements

- Python 3.9+ recommended
- Stable Internet connection (required to dynamically fetch CSVs and download ResNet50 weights)

Dependencies:
- `streamlit`
- `tensorflow`
- `scikit-learn`
- `pandas`
- `numpy`
- `Pillow`

## Installation & Setup

**1. Clone the repository**
```bash
git clone <your-repo-url>
cd Fashion-recommendation-system
```

**2. Create and activate a virtual environment**
```bash
python -m venv .venv

# On Windows PowerShell:
.venv\Scripts\Activate.ps1

# On Linux/macOS:
source .venv/bin/activate
```

**3. Install required dependencies**
```bash
pip install -r requirements.txt
```

**4. Download the Images Dataset (Crucial Step)**
* Go to the **Releases** section of this GitHub repository.
* Download the zipped images archive (e.g., `images.zip`).
* Extract the contents directly into the root folder of this project so that the path looks exactly like this: `Fashion-recommendation-system/images/`.

## Running the Application

Once your dependencies are installed and your `images` folder is securely in place, start the Streamlit server:

```bash
streamlit run app.py
```

Open the local network URL provided in your terminal (usually `http://localhost:8501`). Upload an image and wait for the ResNet50 model to extract features and fetch your recommendations!

## Troubleshooting

- **Slow First Run:** The very first time you process an image, TensorFlow will download the ResNet50 pretrained weights. This may take a minute depending on your internet speed.
- **Data Loading Failures:** Because `app.py` fetches CSVs from GitHub Releases (`metadados.csv`, `embeddings.csv`, `df_sample.csv`), ensure your network does not block GitHub URLs. 
- **Missing Recommended Images:** If the app runs but displays errors where the images should be, ensure your `images` folder is correctly extracted in the same directory as `app.py` and that the filenames map correctly to the entries in `df_sample.csv`.

## Suggested Future Enhancements

- **Streamlit Caching:** Implement `@st.cache_data` for the remote CSV downloads and `@st.cache_resource` for the ResNet50 model initialization to drastically improve reload speeds.
- **Format Migration:** Migrate the hosted embeddings from CSV to `.npy` format for faster network transmission and parsing.
- **Metadata Filtering:** Utilize the currently unused `metadados.csv` to allow users to filter recommendations by gender, category, or season.
```
