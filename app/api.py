import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from fastapi import FastAPI
from pydantic import BaseModel
from faiss_index import FaissIndex
from models.clip import CLIP
# from models.blip2 import BLIP2
from models.openclip import OpenCLIP

app = FastAPI()
clip_model = CLIP()
# blip2_model = BLIP2()
openclip_model = OpenCLIP(backbone="ViT-B-32", pretrained="laion2b_s34b_b79k")
clip_faiss = FaissIndex(clip_model)
clip_faiss.load("database/clip_faiss.bin", "database/clip_id2path.pkl")
# blip2_faiss = FaissIndex(blip2_model)
# blip2_faiss.load("database/blip2_faiss.bin", "database/blip2_id2path.pkl")
openclip_faiss = FaissIndex(openclip_model)
openclip_faiss.load("database/openclip_faiss.bin", "database/openclip_id2path.pkl")


class TextSearchRequest(BaseModel):
    query: str
    top_k: int
    model_name: str


@app.post("/text-search/")
def text_search(request: TextSearchRequest):
    if request.model_name == "clip":
        scores, indices, paths = clip_faiss.text_search(request.query, top_k=request.top_k)
    # elif request.model_name == "blip2":
    #     scores, indices, paths = blip2_faiss.text_search(request.query, top_k=request.top_k)
    elif request.model_name == "openclip":
        scores, indices, paths = openclip_faiss.text_search(request.query, top_k=request.top_k)
    
    return {"scores": scores, "indices": indices, "paths": paths}


    
