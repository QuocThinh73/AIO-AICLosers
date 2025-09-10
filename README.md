# HCMAI2025 - Video Retrieval Application

## Project Structure

```
├── backend/                 # FastAPI backend server
│   ├── app/
│   │   ├── api/            # API endpoints
│   │   ├── core/           # Configuration and logging
│   │   ├── database/       # Database connections
│   │   ├── ml/             # Machine learning models
│   │   ├── models/         # Data schemas
│   │   ├── services/       # Business logic services
│   │   └── utils/          # Utility functions
│   ├── .env
│   └── requirements.txt
├── frontend/                # React frontend application
│   ├── src/
│   │   ├── components/     # React components
│   │   ├── services/       # API service calls
│   │   └── assets/         # Static assets
│   └── package.json
├── preprocess/              # Video preprocessing pipeline
│   ├── {task_name}.py      # Preprocessing task
│   └── README.md           # Detailed preprocessing guide
└── models/                  # Machine learning models
```

## Quick Start


### 1. Clone Repository

```bash
git clone https://github.com/QuocThinh73/AIO-AICLosers
cd HCMAI2025
```

### 2. Backend

Open new terminal for backend

```bash
cd backend
```

#### 2.1. Pull Qdrant Docker

```bash
docker run -d --name qdrant-db -p 6333:6333 -p 6334:6334 qdrant/qdrant:latest
```

#### 2.2. Set up environment

```bash
conda create --name backend python=3.10
conda activate backend
pip install -r requirements.txt
```

#### 2.3. Add .env

```
API_V1_STR=/api
PROJECT_NAME=AIO-AIClosers

QDRANT_HOST=localhost
QDRANT_PORT=6333

CAPTION_COLLECTION_NAME=caption
OPENCLIP_COLLECTION_NAME=openclip

UUID_NAMESPACE=11111111-2222-3333-4444-555555555555

MEDIA_VIDEO_DIR=<your-video-dir>
MEDIA_KEYFRAME_DIR=<your-keyframe-dir>
MEDIA_URL_PREFIX=/media

FPS_PATH=<your-video-fps-file>
```

Copy to .env

#### 2.4. Run backend

```bash
uvicorn app.main:create_app --host 0.0.0.0 --port 8000 --factory
```

### 3. Frontend

Open new terminal for frontend

```bash
cd frontend
```

#### 3.1. Set up environment

```bash
npm install
```

#### 3.2. Run frontend

```bash
npm run dev
```
