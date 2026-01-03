from pydantic import BaseModel
from typing import Optional, List

class SearchResult(BaseModel):
    file: Optional[str]
    start_page: Optional[int]
    end_page: Optional[int]
    header: Optional[str]
    header_type: Optional[str]
    part: Optional[str]
    chapter: Optional[str]
    section: Optional[str]
    subchunk_index: Optional[int]
    article_index: Optional[int]
    similarity: float
    content: str

class SearchResponse(BaseModel):
    results: List[SearchResult]