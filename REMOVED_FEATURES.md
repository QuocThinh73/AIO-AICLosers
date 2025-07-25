# Removed Features Documentation

## YOLOv8 and Object Filtering

As of July 25, 2025, YOLOv8 object detection and filtering functionality has been removed from the AIO-AIClosers pipeline to streamline the application and reduce memory requirements.

### Components Removed:

1. **Code Files:**
   - `app/filters/yolo_filter.py` - YOLOv8 filtering module
   - Related YOLOv8 detection data in `database/detections/yolov8-seg/`

2. **Configuration:**
   - YOLOv8-specific config in `app/config.py`:
     - `YOLO_COCO_CLASSES`
     - `YOLOV8_INDEX`
     - `YOLOV8_FILTER_ENABLED`
     - `YOLOV8_CONFIDENCE_THRESHOLD`

3. **Pipeline Integration:**
   - YOLOv8 filter import and usage from `search_handler.py`
   - YOLOv8 filtering code in unified search pipeline

4. **UI Components:**
   - Object selection UI remains visible but is disabled
   - A notification has been added to indicate removal of the feature

### Current Search Pipeline:

The search pipeline now uses:
1. **OpenCLIP** - For embedding-based semantic search
2. **GroundingDINO** - For object detection search

The previous object filtering functionality that used YOLOv8 to filter search results by detected objects has been removed. Any code attempting to filter results using `filter_by_objects()` will now simply return the unfiltered results.

### Elasticsearch:

The YOLOv8 index (`yolov8`) in Elasticsearch is no longer referenced or used by the application. Only the `groundingdino` index is used for object detection search.

### User Interface:

The object selection UI components in the web interface have been disabled with a notification that the feature has been removed. The components remain in the UI for potential future re-implementation but are non-functional.
