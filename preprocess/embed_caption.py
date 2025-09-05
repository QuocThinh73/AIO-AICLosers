from FlagEmbedding import BGEM3FlagModel
import torch


def get_caption_embedder():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return BGEM3FlagModel('BAAI/bge-m3', use_fp16=True, device=device)

def process_video(keyframes_video_dir, output_embedded_vector_path):    
    caption_embedder = get_caption_embedder()

    embedded_vectors = []
    for keyframes in sorted(os.listdir(keyframes_video_dir)):
        for keyframe in keyframes:
            keyframe_name = keyframe["keyframe"]
            caption = keyframe["caption"]
            
            keyframe_path = os.path.join(keyframes_video_dir, keyframe_name)
            
            embedding_output = caption_embedder.encode(
                    [caption], 
                    return_dense=True,
                    return_sparse=True,
                    return_colbert_vecs=True
                )

            embedded_vectors.append({
                "keyframe": keyframe_name,
                "dense_vector": embedding_output["dense_vecs"][0],
                "colbert_vector": embedding_output["colbert_vecs"][0],
                "sparse_weight": embedding_output["lexical_weights"][0]
            })
    
    with open(output_embedded_vector_path, "w", encoding="utf-8") as f:
        json.dump(embedded_vectors, f)

def embed_caption(input_caption_dir, output_embedded_vector_dir, mode, lesson_name=None):
    pass