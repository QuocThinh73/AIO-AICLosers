from fastapi import APIRouter

router = APIRouter()

@router.post("/base_search")
def base_search(base_search_request: BaseSearchRequest):
    pass

@router.post("/temporal_search")
def temporal_search(temporal_search_request: TemporalSearchRequest):
    pass