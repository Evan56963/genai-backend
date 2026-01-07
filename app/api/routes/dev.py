# These endpoints are for development and testing purposes only.
from fastapi import APIRouter, Request

from app.models import LLMChatResponse, SearchResponse
from app.utils import format_chromadb_results, get_data

router = APIRouter(prefix="/dev", tags=["Dev"])

@router.post("/chat", response_model=LLMChatResponse)
async def send_message_to_llm(message: str, request: Request):

    state = request.app.state
    
    # Use chat template without fine-tuned role
    messages = [{"role": "user", "content": message}]
    formatted_prompt = state.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = state.tokenizer(text=formatted_prompt, images=None, return_tensors="pt").to(state.model.device)
    outputs = state.model.generate(**inputs, max_new_tokens=128)
    
    # Decode only the newly generated tokens (exclude the input prompt)
    input_length = inputs.input_ids.shape[1]
    generated_tokens = outputs[0][input_length:]
    response = state.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    
    return LLMChatResponse(response=response)

@router.post("/search", response_model=SearchResponse)
async def search_legal_provisions(query: str, request: Request):

    state = request.app.state
    
    docs, metas, dists = get_data(state.embed_model, state.collection, query)
    return SearchResponse(results=format_chromadb_results(docs, metas, dists))