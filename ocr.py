import os
import json
import logging
from datetime import datetime, timezone
from elasticsearch import Elasticsearch, helpers

OCR_RESULTS_ROOT = "/Volumes/Transcend/AIC/AIO-AIClosers/PublicData/result"
ES_HOST = "127.0.0.1"
ES_PORT = 9200
ES_INDEX = "ocr_extractions"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_es_client():
    es = Elasticsearch([{
        "host": ES_HOST,
        "port": ES_PORT,
        "scheme": "http"
    }])
    if not es.ping():
        raise RuntimeError("Cannot connect to Elasticsearch")
    logger.info("✅ Connected to Elasticsearch at %s:%s", ES_HOST, ES_PORT)
    return es


def create_ocr_index(es_client):
    if es_client.indices.exists(index=ES_INDEX):
        logger.info(f"Index {ES_INDEX} already exists.")
        return
    mapping = {
        "mappings": {
            "properties": {
                "image_filename": {"type": "keyword"},
                "image_path": {"type": "keyword"},
                "processing_timestamp": {"type": "date"},
                "ocr_results": {
                    "type": "nested",
                    "properties": {
                        "text": {"type": "text"},
                        "confidence": {"type": "float"},
                        "bbox": {
                            "properties": {
                                "x1": {"type": "integer"},
                                "y1": {"type": "integer"},
                                "x2": {"type": "integer"},
                                "y2": {"type": "integer"}
                            }
                        }
                    }
                },
                "extracted_text_full": {"type": "text"},
                "total_confidence": {"type": "float"},
                "processing_status": {"type": "keyword"},
                "error_message": {"type": "text"}
            }
        }
    }
    es_client.indices.create(index=ES_INDEX, body=mapping)
    logger.info(f"✅ Created index: {ES_INDEX}")


def load_valid_json(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if not content or content == "{}":
                return None
            return json.loads(content)
    except Exception as e:
        logger.warning(f"⚠️ Error reading {file_path}: {e}")
        return None


def index_json_files(es_client):
    for level in range(1, 13):
        dir_name = f"ocr_results_L{level:02}"
        full_path = os.path.join(OCR_RESULTS_ROOT, dir_name)
        if not os.path.isdir(full_path):
            logger.warning(f"Ignore missing dir: {full_path}")
            continue

        logger.info(f"📂 Processing {dir_name}")
        actions = []

        for file_name in os.listdir(full_path):
            if not file_name.endswith(".json"):
                continue
            file_path = os.path.join(full_path, file_name)
            data = load_valid_json(file_path)
            if not data:
                continue

            ocr_results = []
            for frame in data:
                frame_image = frame.get("image")  
                results = frame.get("results", [])
                for item in results:
                    text = item.get("text")
                    conf = item.get("confidence")
                    box = item.get("box")
                    if not (isinstance(text, str) and isinstance(conf, (int, float)) and isinstance(box, list) and len(box) == 4):
                        continue
                    xs = [p[0] for p in box]
                    ys = [p[1] for p in box]
                    bbox = {
                        "x1": int(min(xs)),
                        "y1": int(min(ys)),
                        "x2": int(max(xs)),
                        "y2": int(max(ys)),
                    }
                    ocr_results.append({
                        "image": frame_image,
                        "text": text,
                        "confidence": float(conf),
                        "bbox": bbox
                    })

            extracted_text_full = " ".join(item["text"] for item in ocr_results)
            total_confidence = sum(item["confidence"] for item in ocr_results) / len(ocr_results) if ocr_results else 0
            processing_status = "success"
            error_message = None

            logger.info(f"  📄 {file_name}: indexed {len(ocr_results)} OCR entries")

            doc = {
                "_index": ES_INDEX,
                "_source": {
                    "image_filename": file_name.replace(".json", ""),
                    "image_path": file_path,
                    "processing_timestamp": datetime.now(timezone.utc),
                    "ocr_results": ocr_results,
                    "extracted_text_full": extracted_text_full,
                    "total_confidence": total_confidence,
                    "processing_status": processing_status,
                    "error_message": error_message
                }
            }
            actions.append(doc)

        if actions:
            helpers.bulk(es_client, actions)
            logger.info(f"✅ Indexed {len(actions)} docs from {dir_name}")
        else:
            logger.info(f"⚠️ No valid JSONs in {dir_name}")


if __name__ == "__main__":
    es = create_es_client()
    create_ocr_index(es)
    index_json_files(es)
    logger.info("🎉 All done!")    
