from typing import Optional
from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel
from sqlmodel import SQLModel, Field

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
    results: list[SearchResult]

class LLMRagResponse(BaseModel):
    answer: str
    file: Optional[str] = None
    part: Optional[str] = None
    chapter: Optional[str] = None
    section: Optional[str] = None
    article: Optional[str] = None
    start_page: Optional[int] = None

class LLMChatResponse(BaseModel):
    response: str

class MessageRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"

class Message(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    conversation_id: str
    role: MessageRole
    content: str
    created_at: datetime = datetime.now()