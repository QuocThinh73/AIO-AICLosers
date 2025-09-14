from fastapi import APIRouter, UploadFile, File, Depends
from typing import Optional
from app.models.schemas import BaseSearchRequest, TemporalSearchRequest
from app.utils.rerank import rrf
from app.utils.translate import translate_text
from io import BytesIO
from PIL import Image
from app.services.search_service import get_search_service

router = APIRouter()


def to_list(ids):
    """Convert comma-separated string to list"""
    return [id.strip() for id in ids.split(",")] if ids else None

@router.post("/base_search")
async def base_search(base_search_request: BaseSearchRequest = Depends(BaseSearchRequest.as_form), embedding_image: Optional[UploadFile] = File(None), search_service=Depends(get_search_service)):

    # Convert string to list of batch ids
    include_batch_ids = to_list(base_search_request.include_batch_ids)
    exclude_batch_ids = to_list(base_search_request.exclude_batch_ids)
    include_video_ids = to_list(base_search_request.include_video_ids)
    exclude_video_ids = to_list(base_search_request.exclude_video_ids)

    # Translate text from Vietnamese to English if needed
    if base_search_request.use_translation:
        base_search_request.embedding_text = await translate_text(text=base_search_request.embedding_text, src_lang="vi", dest_lang="en") if base_search_request.embedding_text else base_search_request.embedding_text
        base_search_request.captioning_text = await translate_text(text=base_search_request.captioning_text, src_lang="vi", dest_lang="en") if base_search_request.captioning_text else base_search_request.captioning_text
        base_search_request.object_detection_text = await translate_text(text=base_search_request.object_detection_text, src_lang="vi", dest_lang="en") if base_search_request.object_detection_text else base_search_request.object_detection_text

    # Search
    results = []
    if base_search_request.embedding_text:
        embedding_text_result = search_service.search_openclip(
            text=base_search_request.embedding_text, image=None, top_k=base_search_request.top_k*2, include_batch_ids=include_batch_ids, exclude_batch_ids=exclude_batch_ids, include_video_ids=include_video_ids, exclude_video_ids=exclude_video_ids)
        results.append(embedding_text_result)
    if embedding_image:
        embedding_image = Image.open(BytesIO(await embedding_image.read())).convert("RGB")
        embedding_image_result = search_service.search_openclip(
            image=embedding_image, text=None, top_k=base_search_request.top_k*2, include_batch_ids=include_batch_ids, exclude_batch_ids=exclude_batch_ids, include_video_ids=include_video_ids, exclude_video_ids=exclude_video_ids)
        results.append(embedding_image_result)
    if base_search_request.captioning_text:
        captioning_result = search_service.search_caption(
            base_search_request.captioning_text, base_search_request.top_k*2, include_batch_ids=include_batch_ids, exclude_batch_ids=exclude_batch_ids, include_video_ids=include_video_ids, exclude_video_ids=exclude_video_ids)
        results.append(captioning_result)
    if base_search_request.ocr_text:
        ocr_result = search_service.search_ocr(
            base_search_request.ocr_text, base_search_request.top_k*2, include_batch_ids=include_batch_ids, exclude_batch_ids=exclude_batch_ids, include_video_ids=include_video_ids, exclude_video_ids=exclude_video_ids)
        results.append(ocr_result)
    if base_search_request.object_detection_text:
        object_detection_result = search_service.search_object(
            base_search_request.object_detection_text, base_search_request.top_k*2, include_batch_ids=include_batch_ids, exclude_batch_ids=exclude_batch_ids, include_video_ids=include_video_ids, exclude_video_ids=exclude_video_ids)
        results.append(object_detection_result)

    # Rerank
    results = rrf(results, base_search_request.top_k)

    # Response
    response = {
        "results": results
    }

    if base_search_request.use_translation:
        response["english_query"] = base_search_request.embedding_text

    # Return results
    return response



def group_temporal_results(event_results, events):
    """Group search results temporally based on batch_id and video_id"""
    # Làm phẳng tất cả kết quả và thêm event_id
    all_keyframes = []
    for i, results in enumerate(event_results):
        for result in results:
            result["event_id"] = events[i].event_id
            all_keyframes.append(result)
    
    # Nhóm theo batch_id và video_id
    groups = {}
    for keyframe in all_keyframes:
        # Tạo combined_id kết hợp batch_id và video_id
        batch_id = keyframe.get("batch_id", "unknown")
        video_id = keyframe.get("video_id", "unknown")
        combined_id = f"{batch_id}_{video_id}"
        
        if combined_id not in groups:
            groups[combined_id] = []
        groups[combined_id].append(keyframe)
    
    # Xử lý từng nhóm để đảm bảo thứ tự thời gian
    valid_sequences = []
    for combined_id, group in groups.items():
        # Sắp xếp theo event_id trong nhóm
        events_in_group = {}
        for keyframe in group:
            event_id = keyframe["event_id"]
            if event_id not in events_in_group:
                events_in_group[event_id] = []
            events_in_group[event_id].append(keyframe)
        
        # Kiểm tra xem có đủ các event trong nhóm không
        expected_event_ids = [event.event_id for event in events]
        if not all(event_id in events_in_group for event_id in expected_event_ids):
            continue
        
        # Với mỗi event, sắp xếp các keyframe theo timestamp/frame_index
        for event_id in events_in_group:
            events_in_group[event_id].sort(key=lambda x: x.get("frame_index", 0))
        
        # Tạo tất cả các chuỗi thời gian hợp lệ (E1 < E2 < E3 < E4)
        valid_sequences_in_group = validate_temporal_order(events_in_group, expected_event_ids, combined_id)
        valid_sequences.extend(valid_sequences_in_group)
    
    # Sắp xếp các chuỗi theo số lượng sự kiện và điểm số
    valid_sequences.sort(key=lambda x: len(x["keyframes"]), reverse=True)
    
    return valid_sequences

def validate_temporal_order(events_in_group, expected_event_ids, video_id):
    """Tìm tất cả chuỗi thỏa mãn thứ tự thời gian và định dạng theo yêu cầu"""
    valid_sequences = []
    
    # Sử dụng thuật toán đệ quy để tìm chuỗi hợp lệ
    def find_valid_sequences(current_keyframes, remaining_events, last_timestamp=None):
        # Nếu đã xử lý hết các sự kiện, thêm chuỗi vào kết quả
        if not remaining_events:
            # Định dạng theo yêu cầu: chỉ lấy tên keyframe
            keyframe_names = [kf["keyframe"] for kf in current_keyframes]
            valid_sequences.append({
                "video_id": video_id,
                "keyframes": keyframe_names
            })
            return
        
        # Lấy event_id tiếp theo cần xử lý
        next_event_id = remaining_events[0]
        
        # Duyệt qua tất cả keyframes của event này
        for keyframe in events_in_group[next_event_id]:
            timestamp = keyframe.get("frame_index", 0)
            
            # Kiểm tra thứ tự thời gian
            if last_timestamp is None or timestamp > last_timestamp:
                # Thêm keyframe vào chuỗi và tiếp tục đệ quy
                current_keyframes.append(keyframe)
                find_valid_sequences(current_keyframes, remaining_events[1:], timestamp)
                current_keyframes.pop()  # Quay lui
    
    # Bắt đầu tìm chuỗi hợp lệ
    find_valid_sequences([], expected_event_ids)
    return valid_sequences

@router.post("/temporal_search")
async def temporal_search(temporal_search_request: TemporalSearchRequest, search_service=Depends(get_search_service)):
    # 1. Xử lý các tham số và chuyển đổi (tương tự base_search)
    include_batch_ids = to_list(temporal_search_request.include_batch_ids)
    exclude_batch_ids = to_list(temporal_search_request.exclude_batch_ids)
    include_video_ids = to_list(temporal_search_request.include_video_ids)
    exclude_video_ids = to_list(temporal_search_request.exclude_video_ids)
    
    # 2. Dịch các truy vấn nếu cần
    if temporal_search_request.use_translation:
        for event in temporal_search_request.events:
            event.query = await translate_text(text=event.query, src_lang="vi", dest_lang="en")
    
    # 3. Tìm kiếm top_k kết quả cho mỗi sự kiện
    event_results = []
    for event in temporal_search_request.events:
        # Thực hiện tìm kiếm
        results = search_service.search_openclip(
            text=event.query, image=None, 
            top_k=temporal_search_request.top_k, 
            include_batch_ids=include_batch_ids, 
            exclude_batch_ids=exclude_batch_ids,
            include_video_ids=include_video_ids,
            exclude_video_ids=exclude_video_ids
        )
        
        event_results.append(results)
    
    # 4. Gọi hàm xử lý nhóm theo phương pháp 2
    grouped_results = group_temporal_results(event_results, temporal_search_request.events)
    
    # 5. Trả về kết quả
    return {"results": grouped_results}
