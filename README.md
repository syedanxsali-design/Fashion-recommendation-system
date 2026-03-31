# Fashion Recommendation System

An image-based fashion recommendation web app built with Streamlit, TensorFlow (ResNet50), and cosine similarity.

Upload a product image, and the app retrieves visually similar items from a catalog using precomputed embeddings.

## Overview

This project performs visual similarity search for fashion products.

- Query input: user-uploaded image
- Feature extractor: pre-trained ResNet50 (ImageNet)
- Retrieval engine: cosine similarity against catalog embeddings
- Output: top similar products with image + product name

## Core Features

- Streamlit web interface for quick interactive use
- ResNet50-based feature extraction for uploaded images
- Top-N nearest-neighbor recommendation using cosine similarity
- Hosted dataset loading from GitHub release assets
- Local image rendering for recommended items from images folder

## Project Structure

```text
.
|-- app.py
|-- requirements.txt
|-- README.md
|-- df_sample.csv
|-- metadados.csv
|-- embeddings.csv
|-- embeddings.npy
|-- model_embeddings.npy
`-- resnet_embeddings.npy
```

Notes:
- The running app currently reads CSV data from hosted URLs configured inside app.py.
- Local CSV/NPY files may be historical or alternative artifacts.

## How It Works

1. Load catalog metadata and embedding matrix from hosted CSV files.
2. Keep only numeric columns from embeddings.csv to form a 2D embedding matrix.
3. Initialize ResNet50 with global average pooling to get a fixed-length feature vector.
4. User uploads a JPG image in the Streamlit UI.
5. App preprocesses the image (resize to 224x224 + ResNet preprocessing).
6. App extracts query embedding and computes cosine similarity to all catalog embeddings.
7. Top matches are selected and displayed with product names and images.

## Data Sources (Configured in app.py)

- metadados.csv
	- https://github.com/syedanxsali-design/Fashion-recommendation-system/releases/download/metadados.csv/metadados.csv
- embeddings.csv
	- https://github.com/syedanxsali-design/Fashion-recommendation-system/releases/download/embeddings.csv/embeddings.csv
- df_sample.csv
	- https://github.com/syedanxsali-design/Fashion-recommendation-system/releases/download/df_sample.csv/df_sample.csv

## Data Contract

### df_sample.csv

Expected minimum columns used by the app:
- image
- productDisplayName

For local image mode:
- image values should match filenames present in images directory (for example 15970.jpg).

Other metadata columns (for example gender, category, color, season) can exist and are useful for analysis.

### embeddings.csv
Source: https://github.com/syedanxsali-design/Fashion-recommendation-system/releases/download/model_embeddings.npy/model_embeddings.npy

Expected format:
- One row per catalog item
- Numeric embedding columns (for example 0..2047)
- Optional extra non-numeric/index columns are ignored by the app

Important:
- Row order in embeddings.csv must align exactly with row order in df_sample.csv.

### metadados.csv

Currently loaded but not directly used in recommendation scoring.
It can be used in future for filtering, faceted search, reranking, or UI enrichment.

## Requirements

- Python 3.9+ recommended
- Internet connection (to download hosted CSV files and pretrained ResNet50 weights)

Python dependencies are listed in requirements.txt:
- streamlit
- tensorflow
- scikit-learn
- pandas
- numpy
- Pillow

## Installation

```bash
# 1) Clone and enter project
git clone <your-repo-url>
cd FRS

# 2) Create virtual environment
python -m venv .venv

# 3) Activate virtual environment
# Windows PowerShell
.venv\Scripts\Activate.ps1

# 4) Install dependencies
pip install -r requirements.txt
```

## Run The App

```bash
streamlit run app.py
```

Then open the local URL shown in your terminal (typically http://localhost:8501).

## Usage

1. Launch the Streamlit app.
2. Upload a JPG PNG JPEG WEBP image.
3. Wait for feature extraction and similarity search.
4. Review recommended products shown in the grid.

## Troubleshooting

### 1) TensorFlow installation issues

- Update pip first:

```bash
python -m pip install --upgrade pip
```

- Ensure your Python version is supported by the TensorFlow version resolved by pip.

### 2) Slow first run

- The first run may take longer because ResNet50 pretrained weights are downloaded.

### 3) Network/data loading failures

- The app depends on external CSV URLs; if offline or blocked, loading will fail.
- Confirm URLs are reachable from your network.

### 4) Shape/alignment mismatch

If you see similarity computation errors, verify:
- embeddings.csv has numeric vectors with consistent length
- number/order of rows in embeddings.csv matches df_sample.csv

### 5) Recommended image not found

- The app loads recommendations from local images directory.
- Ensure image files exist under images and match names in df_sample.csv image column.

## Known Limitations

- Upload type is currently restricted to JPG only
- metadados.csv is loaded but not yet used in ranking/filtering
- No caching for dataset/model loading in current script
- Local images directory is required for recommendation thumbnails

## Suggested Enhancements

- Add Streamlit caching:
	- st.cache_data for CSV loads
	- st.cache_resource for model initialization
- Add UI controls for top-N recommendations
- Support PNG and JPEG file uploads
- Add metadata filters (gender/category/usage/season)
- Add robust handling for missing/corrupt local image files
- Optionally migrate from CSV embeddings to NPY for faster startup

## License

No license file is currently included in this repository.
Add a LICENSE file if you want to define usage and distribution rights.
