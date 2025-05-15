# Text-to-Video Search with Vision-Language Models

This project allows you to search videos based on text descriptions using Vision-Language Models (VLMs) like CLIP. It supports both Vietnamese and English queries through an integrated translation module.

## 🌟 Features

- **Cross-modal search**: Search for videos using either text descriptions or example images
- **Multilingual support**: Automatic translation from Vietnamese to English
- **Modular design**: Easily extendable with new VLM models
- **Efficient retrieval**: Fast similarity search using FAISS indices
- **API-based interface**: Simple REST API built with FastAPI
- **Keyframe extraction**: Automatic extraction of keyframes from videos

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
git clone https://github.com/username/text-to-video-search.git
cd text-to-video-search
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

To extract keyframes from your videos and build a FAISS index:

```python
from vlm import CLIPModel
from data_preparation import build_index_from_videos

# Initialize the model
model = CLIPModel()

# Extract keyframes and build the index
build_index_from_videos(
    video_directory="path/to/your/videos",  
    model=model,
    output_dir="faiss_index",
    keyframes_dir="data/keyframes"
)
```

### 2. Run the API

Start the FastAPI application:

```bash
python app.py
```

The API will be available at http://localhost:8000, with interactive documentation at http://localhost:8000/docs.

### 3. Search for videos

#### Using text:

```
GET /search/text?query=người đang đi dạo trong công viên&top_k=5
```

#### Using an image:

```
POST /search/image
```
With form data:
- file: [image file]
- top_k: 5 (optional)

## 📦 Extending with New Models

To add a new Vision-Language Model, simply create a new class in `vlm.py` that inherits from `BaseVLM`:

```python
class NewModel(BaseVLM):
    def __init__(self, model_name: str = "path/to/model"):
        # Initialize your model
        
    def encode_text(self, text: str) -> np.ndarray:
        # Implement text encoding
        
    def encode_image(self, image: Image.Image) -> np.ndarray:
        # Implement image encoding
```

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

## 👏 Acknowledgements

- OpenAI for the CLIP model
- Facebook Research for FAISS
- Hugging Face for the transformers library
