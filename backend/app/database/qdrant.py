import os
import torch
from qdrant_client import QdrantClient, models
from FlagEmbedding import BGEM3FlagModel
from app.ml.openclip import OpenCLIP
from app.utils.generate_id import generate_id


class Qdrant:
    def __init__(self, host, port, caption_model=None, openclip_model=None):
        self.client = QdrantClient(
            host=host, port=port, grpc_port=6334, prefer_grpc=True)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.caption_model = caption_model or BGEM3FlagModel(
            'BAAI/bge-m3', use_fp16=True, device=self.device)
        self.openclip_model = openclip_model or OpenCLIP(
            'ViT-B-16', pretrained='dfn2b', device=self.device)

    def _chunk(self, xs, n):
        for i in range(0, len(xs), n):
            yield xs[i:i+n]

    def _create_sparse_vector(self, sparse_data):
        sparse_indices = []
        sparse_values = []

        for key, value in sparse_data.items():
            if float(value) > 0:
                if isinstance(key, str):
                    if key.isdigit():
                        key = int(key)
                    else:
                        continue

                sparse_indices.append(key)
                sparse_values.append(float(value))

        return models.SparseVector(
            indices=sparse_indices,
            values=sparse_values
        )

    def _create_filter(self, include_batch_ids=None, exclude_batch_ids=None, include_video_ids=None, exclude_video_ids=None, ocr=None):
        must, must_not = [], []

        if include_batch_ids:
            must.append(models.FieldCondition(key="batch_id",
                                              match=models.MatchAny(any=include_batch_ids)))

        if include_video_ids:
            must.append(models.FieldCondition(key="video_id",
                                              match=models.MatchAny(any=include_video_ids)))

        if exclude_batch_ids:
            must_not.append(models.FieldCondition(
                key="batch_id", match=models.MatchAny(any=exclude_batch_ids)))

        if exclude_video_ids:
            must_not.append(models.FieldCondition(
                key="video_id", match=models.MatchAny(any=exclude_video_ids)))

        if ocr:
            should_terms = [models.FieldCondition(
                key="text", match=models.MatchText(text=t.lower())) for t in ocr]
            must.append(models.NestedCondition(nested=models.Nested(
                key="ocr", filter=models.Filter(should=should_terms))))

        if not must and not must_not:
            return None
        return models.Filter(must=must, must_not=must_not)

    def is_collection_exists(self, collection_name):
        return self.client.collection_exists(collection_name)

    def generate_caption_embeddings(self, captions):
        return self.caption_model.encode(
            captions,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False
        )

    def generate_openclip_embeddings(self, text, image):
        if text:
            return self.openclip_model.encode_text(text)
        elif image:
            return self.openclip_model.encode_image(image)

    def create_collection(self, collection_name):
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "openclip_dense": models.VectorParams(
                    size=512,
                    distance=models.Distance.COSINE
                ),
                "caption_dense": models.VectorParams(
                    size=1024,
                    distance=models.Distance.COSINE
                )
            },
            sparse_vectors_config={
                "caption_sparse": models.SparseVectorParams(
                    index=models.SparseIndexParams(
                        on_disk=True
                    )
                )
            },
        )

    def insert_to_collection(self, items, collection_name, batch_size):
        from tqdm import tqdm
        batches = list(self._chunk(items, batch_size))
        for batch in tqdm(batches, total=len(batches)):
            captions = [(item["caption"]) for item in batch]
            caption_embeddings = self.generate_caption_embeddings(captions)
            captions_dense = caption_embeddings["dense_vecs"]
            captions_sparse = [self._create_sparse_vector(
                sparse_vec) for sparse_vec in caption_embeddings["lexical_weights"]]

            points = []
            for item, caption_dense, caption_sparse in zip(batch, captions_dense, captions_sparse):
                keyframe = item["keyframe"]
                openclip_dense = item["embedded_vector"]
                ocr = [
                    {"text": r["text"].lower(), "box": r["box"]}
                    for r in item["ocr"]
                ]

                base = os.path.basename(keyframe)
                stem = os.path.splitext(base)[0]
                batch_id, video_id, frame_id = stem.split("_")

                points.append(models.PointStruct(
                    id=generate_id(keyframe),
                    payload={
                        "keyframe": keyframe,
                        "batch_id": batch_id,
                        "video_id": video_id,
                        "frame_id": int(frame_id),
                        "caption": item["caption"],
                        "ocr": ocr
                    },
                    vector={
                        "openclip_dense": openclip_dense,
                        "caption_dense": caption_dense,
                        "caption_sparse": caption_sparse,
                    }
                ))

            self.client.upsert(collection_name=collection_name,
                               points=points, wait=False)

    def search_caption(self, search_query, collection_name, limit, include_batch_ids=None, exclude_batch_ids=None, include_video_ids=None, exclude_video_ids=None, ocr=None):
        caption_embeddings = self.generate_caption_embeddings([search_query])

        caption_dense = caption_embeddings["dense_vecs"][0]
        caption_sparse = self._create_sparse_vector(
            caption_embeddings["lexical_weights"][0])

        filter = self._create_filter(
            include_batch_ids=include_batch_ids, exclude_batch_ids=exclude_batch_ids, include_video_ids=include_video_ids, exclude_video_ids=exclude_video_ids, ocr=ocr)

        points = self.client.query_points(
            collection_name,
            prefetch=[
                models.Prefetch(query=caption_dense,
                                using="caption_dense", limit=limit*2),
                models.Prefetch(query=caption_sparse,
                                using="caption_sparse", limit=limit*2),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            with_payload=True,
            query_filter=filter,
            limit=limit,
        ).points

        keyframes = [point.payload["keyframe"] for point in points]

        return keyframes

    def search_openclip(self, text, image, collection_name, limit, include_batch_ids=None, exclude_batch_ids=None, include_video_ids=None, exclude_video_ids=None, ocr=None):
        openclip_dense = self.generate_openclip_embeddings(
            text=text, image=image)

        filter = self._create_filter(
            include_batch_ids=include_batch_ids, exclude_batch_ids=exclude_batch_ids, include_video_ids=include_video_ids, exclude_video_ids=exclude_video_ids, ocr=ocr)

        points = self.client.query_points(
            collection_name,
            query=openclip_dense,
            using="openclip_dense",
            with_payload=True,
            query_filter=filter,
            limit=limit,
        ).points

        keyframes = [point.payload["keyframe"] for point in points]
        return keyframes
