from backend.app.database.qdrant import Qdrant
from backend.app.database.elasticsearch import Elasticsearch


class DatabaseService:
    def __init__(self, qdrant_host, qdrant_port, elasticsearch_host, elasticsearch_port):
        self.qdrant = Qdrant(host=qdrant_host, port=qdrant_port)
        self.elasticsearch = Elasticsearch(host=elasticsearch_host, port=elasticsearch_port)