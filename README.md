# Text-based Image Retrieval 

HCMAI Challenge 2025

## 📋 Project Structure

```
project_root/
│
├── data_preparation.py        # Functions to build and save FAISS index from images/keyframes
├── faiss_index.py             # FaissIndex class for search operations
├── translation.py             # Translation module (Vietnamese -> English)
├── vlm.py                     # Abstract BaseVLM class and model implementations (CLIP, BLIP)
├── app.py                     # FastAPI application
│
├── data/                      # Data directory
│   └── keyframes/             # Directory for extracted keyframes
│
├── faiss_index/               # Directory for FAISS index files
│
├── requirements.txt           # Project dependencies
└── README.md                  # This file
```

## 🔧 Installation

1. Clone the repository:
```bash
git clone https://github.com/QuocThinh73/HCMAI2025.git
cd HCMAI2025
```

2. Install the requirements:
```bash
pip install -r requirements.txt
```

3. Create necessary directories:
```bash
mkdir -p data/keyframes faiss_index
```

## 🚀 Usage

### 1. Extract keyframes and build index

```python

```

### 2. Run the API

Start the FastAPI application:

```bash
python app.py
```

The API will be available at http://localhost:8000, with interactive documentation at http://localhost:8000/docs.

## 📦 Extending with New Models

To add a new Vision-Language Model, simply create a new class in `vlm.py` that inherits from `BaseVLM`:

```python
class NewModel(BaseVLM):
    def __init__(self, #TODO):
        # Initialize your model
        
    def encode_text(self, text):
        # Implement text encoding
        
    def encode_image(self, image):
        # Implement image encoding
```