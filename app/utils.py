import re
import uuid
from typing import Literal

from app.models import SearchResult, Message

def format_context(data: list[SearchResult], max_chars_per_doc: int = 1200) -> str:
    if not data:
        return "(未找到相關法條或段落)"
    
    formatted_parts = []
    for i, result in enumerate(data, start=1):
        # Build source line: 【來源 N】 filename (pages X - Y)  標題: header
        source_parts = []
        
        # File name
        if result.file:
            source_parts.append(result.file)
        else:
            source_parts.append("unknown")
        
        # Pages
        if result.start_page:
            page_info = f"(pages {result.start_page} - {result.end_page if result.end_page else result.start_page})"
            source_parts.append(page_info)
        
        # Header/Title
        if result.header:
            source_parts.append(f"標題: {result.header}")
        else:
            source_parts.append("標題: (無標題)")
        
        # Combine source line
        source_line = f"【來源 {i}】 {' '.join(source_parts)}"
        
        # Truncate content if needed
        content = result.content.strip()
        if len(content) > max_chars_per_doc:
            content = content[:max_chars_per_doc] + "..."
        
        # Format entry: source line + newline + content
        formatted_entry = f"{source_line}\n{content}"
        formatted_parts.append(formatted_entry)
    
    return "\n\n---\n\n".join(formatted_parts)


def format_answer(answer: str) -> str:
    if not answer:
        return ""
    
    # Remove assistant marker and everything before it
    if match := re.search(r'\bassistant\b', answer, re.IGNORECASE):
        answer = answer[match.end():].lstrip()
    
    # Remove prefix and everything before it (handles both : and ：)
    prefix_pattern = r'(?:悟空|孫悟空|猴子|齊天大聖|孫|wuluo)\s*[:：]\s*'
    if match := re.search(prefix_pattern, answer, re.IGNORECASE):
        answer = answer[match.end():].lstrip()
    
    # Remove code patterns and everything after them (like ://wuluo:)
    if match := re.search(r'[:：]{1,2}[/\\/]{2,}', answer):
        answer = answer[:match.start()]
    
    # Filter valid characters to remove garbled text
    answer = re.sub(
        r'[^\u4e00-\u9fff\u3000-\u303fa-zA-Z0-9\s.,!?;:：；，。！？、（）()\[\]「」『』""\'\'…—\-\\/]', '', answer)
    
    # Normalize whitespace
    answer = ' '.join(answer.split())
    
    # Keep only complete sentences
    sentence_endings = '。！？.!?'
    last_end = max((i for i, c in enumerate(answer) if c in sentence_endings), default=-1)
    if last_end > 0:
        answer = answer[:last_end + 1]
    
    return answer.strip()


def format_chromadb_results(docs, metas, dists) -> list[SearchResult]:
    results = []
    for doc, meta, dist in zip(docs, metas, dists):
        results.append(
            SearchResult(
                file=meta.get("file"),
                start_page=meta.get("start_page"),
                end_page=meta.get("end_page"),
                header=meta.get("header"),
                header_type=meta.get("header_type"),
                part=meta.get("part"),
                chapter=meta.get("chapter"),
                section=meta.get("section"),
                subchunk_index=meta.get("subchunk_index"),
                article_index=meta.get("article_index"),
                similarity=round(dist, 4),
                content=doc.strip().replace("\n", " ")
            )
        )
    return results

def get_data(embed_model, collection, user_input: str):

    query_embedding = embed_model.encode(
        [user_input], 
        convert_to_numpy=True, 
        show_progress_bar=False, 
        normalize_embeddings=True)[0]
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5,
        include=['documents', 'metadatas', 'distances']
)
    
    docs = results.get('documents', [[]])[0]
    metas = results.get('metadatas', [[]])[0]
    dists = results.get('distances', [[]])[0]

    return docs, metas, dists

def save(db_session, role: list[Literal["user", "assistant"]], content: list[str]) -> None:

    conversation_id = str(uuid.uuid4())

    message1 = Message(
        conversation_id=conversation_id,
        role=role[0],
        content=content[0]
    )

    message2 = Message(
        conversation_id=conversation_id,
        role=role[1],
        content=content[1]
    )
    with db_session:
        db_session.add(message1)
        db_session.add(message2)
        db_session.commit()