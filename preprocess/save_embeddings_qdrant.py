from backend.app.database.qdrant import Qdrant


def get_qdrant_client():
    qdrant_client = Qdrant(host="localhost", port=6333)
    return qdrant_client

def save_embeddings_qdrant(embedded_vector_file, collection_name):
    qdrant_client = get_qdrant_client()