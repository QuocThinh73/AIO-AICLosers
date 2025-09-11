import os
from qdrant_client import QdrantClient, models
from FlagEmbedding import BGEM3FlagModel
from backend.app.ml.openclip import OpenCLIP
from backend.app.utils.generate_id import generate_id


class Qdrant:
    def __init__(self, host, port, caption_model=BGEM3FlagModel('BAAI/bge-m3', use_fp16=True), openclip_model=OpenCLIP('ViT-B-16', pretrained='dfn2b')):
        self.client = QdrantClient(
            host=host,
            port=port,           # REST
            grpc_port=6334,      # gRPC phải mở
            prefer_grpc=True,    # ưu tiên gRPC, không fallback nếu gRPC sẵn sàng
            timeout=60.0,
        )
        self.caption_model = caption_model
        self.openclip_model = openclip_model

    def is_collection_exists(self, collection_name):
        cols = self.client.get_collections().collections
        return any(c.name == collection_name for c in cols)

    def generate_caption_embeddings(self, text):
        return self.caption_model.encode(
            [text],
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=True
        )

    def generate_caption_embeddings_batch(self, texts):
        return self.caption_model.encode(
            texts,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=True
        )

    def generate_openclip_embeddings(self, text, image):
        if text:
            return self.openclip_model.encode_text(text)
        elif image:
            return self.openclip_model.encode_image(image)

    def _create_sparse_vector(self, sparse_data):
        """Convert BGE-M3 sparse output to Qdrant sparse vector format"""
        sparse_indices = []
        sparse_values = []

        for key, value in sparse_data.items():
            # Only process positive values
            if float(value) > 0:
                # Handle string keys
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
                ),
                "caption_colbert": models.VectorParams(
                    size=1024,
                    distance=models.Distance.COSINE,
                    multivector_config=models.MultiVectorConfig(
                        comparator=models.MultiVectorComparator.MAX_SIM
                    ),
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

    def _chunk(self, xs, n):
        for i in range(0, len(xs), n):
            yield xs[i:i+n]

    def insert_to_collection(self, items, collection_name, batch_points):
        print(
            f"Inserting {len(items)} items to collection {collection_name} in batches of {batch_points}")
        batches = list(self._chunk(items, batch_points))
        for i, batch in enumerate(batches, start=1):
            print(f"Inserting batch {i} of {len(batches)}")
            texts = [(it.get("caption") or "") for it in batch]
            enc = self.generate_caption_embeddings_batch(texts)
            dense_list = enc["dense_vecs"]
            colbert_list = enc["colbert_vecs"]
            sparse_list = enc["lexical_weights"]

            points = []
            for it, c_dense, c_colbert, c_sparse in zip(batch, dense_list, colbert_list, sparse_list):
                kf = it["keyframe"]
                vec = it.get("embedded_vector") or it.get(
                    "embed_vector") or it.get("embedd_vector")
                base = os.path.basename(kf)
                stem = os.path.splitext(base)[0]
                b, v, f = stem.split("_")
                points.append(models.PointStruct(
                    id=generate_id(kf),
                    payload={"keyframe": kf, "batch_id": b,
                             "video_id": v, "frame_id": int(f)},
                    vector={
                        "openclip_dense": vec,
                        "caption_dense": c_dense,
                        "caption_colbert": c_colbert,
                        "caption_sparse": self._create_sparse_vector(c_sparse)
                    }
                ))
            self.client.upsert(collection_name=collection_name,
                               points=points, wait=(i == len(batches)))

    def search_caption(self, search_query, collection_name, limit=100, prefetch_limit=300):
        # Generate caption embeddings for the query
        caption_embeddings = self.generate_caption_embeddings(search_query)

        caption_dense = caption_embeddings["dense_vecs"][0]
        caption_sparse = caption_embeddings["lexical_weights"][0]
        caption_colbert = caption_embeddings["colbert_vecs"][0]

        # Set up prefetch for hybrid search
        prefetch = [
            models.Prefetch(
                query=caption_sparse,
                using="sparse",
                limit=prefetch_limit),
            models.Prefetch(
                query=caption_dense,
                using="dense",
                limit=prefetch_limit)
        ]

        # Perform reranking with ColBERT
        points = self.client.query_points(
            collection_name,
            prefetch=prefetch,
            query=caption_colbert,
            using="colbert",
            with_payload=True,
            limit=limit,
        ).points

        keyframes = [point.payload["keyframe"] for point in points]

        return keyframes

    def _create_filter(self, include_batch_ids=None, exclude_batch_ids=None):
        should, must_not = [], []

        if include_batch_ids:
            should.append(models.FieldCondition(key="batch_id",
                          match=models.MatchAny(any=include_batch_ids)))
        if exclude_batch_ids:
            must_not.append(models.FieldCondition(
                key="batch_id", match=models.MatchAny(any=exclude_batch_ids)))

        if not should and not must_not:
            return None
        return models.Filter(should=should, must_not=must_not)

    def search_openclip(self, text, image, collection_name, limit=100, include_batch_ids=None, exclude_batch_ids=None):
        dense_vec = self.generate_openclip_embeddings(text=text, image=image)

        points = self.client.query_points(
            collection_name,
            query=dense_vec,
            using="dense",
            with_payload=True,
            query_filter=self._create_filter(
                include_batch_ids=include_batch_ids, exclude_batch_ids=exclude_batch_ids),
            limit=limit,
        ).points

        keyframes = [point.payload["keyframe"] for point in points]
        return keyframes
