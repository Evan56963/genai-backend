from fastapi import APIRouter, Request

from app.models import SearchResponse
from app.utils import format_chromadb_results

router = APIRouter(prefix="/rag", tags=["RAG"])

@router.post("/search", response_model=SearchResponse)
async def search_legal_provisions(query: str, request: Request):

    query_embedding = request.app.state.embed_model.encode(
        [query], 
        convert_to_numpy=True, 
        show_progress_bar=False, 
        normalize_embeddings=True)[0]
    
    results = request.app.state.collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
        include=['documents', 'metadatas', 'distances']
)
    
    docs = results.get('documents', [[]])[0]
    metas = results.get('metadatas', [[]])[0]
    dists = results.get('distances', [[]])[0]

    return SearchResponse(results=format_chromadb_results(docs, metas, dists))

@router.post("/llm/answer")
async def answer_throgh_llm(query: str):
    return {"answer": f"Answering query: {query}"}