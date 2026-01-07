import textwrap

import torch
from fastapi import APIRouter, Request

from app.models import LLMRagResponse
from app.utils import format_chromadb_results, format_answer, get_data, format_context, save

router = APIRouter(prefix="/rag", tags=["RAG"])

@router.post("/ask", response_model=LLMRagResponse)
async def answer_throgh_llm(user_input: str, request: Request):

    state = request.app.state

    docs, metas, dists = get_data(state.embed_model, state.collection, user_input)
    preprocess_context = format_chromadb_results(docs, metas, dists)

    context = format_context(preprocess_context)

    # Prepare the prompt with correct format from training
    # prompt = f"Human: {user_input}\n 悟空:"
    prompt = textwrap.dedent(f"""
    Human: 以下是從法規資料庫檢索到的相關內容，請參考並且根據這些內容回答下面的問題；回答後請簡短列出你引用的來源 (檔名、起訖頁、標題)。

    {context}

    問題: {user_input}

    悟空:
    """)

    inputs = state.tokenizer(text=prompt, images=None, truncation=True, return_tensors="pt", max_length=4032).to(state.model.device)
    with torch.no_grad():
        outputs = state.model.generate(**inputs,
                        max_new_tokens=256,
                        temperature=0.7,
                        do_sample=True,
                        top_p=0.9,
                        repetition_penalty=1.1,
                        eos_token_id=state.tokenizer.eos_token_id,)
    
    input_length = inputs.input_ids.shape[1]
    answer = state.tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()
    
    save(state.session, role=["user", "assistant"], content=[user_input, answer])

    top = preprocess_context[0] if preprocess_context else None
    return LLMRagResponse(
        answer=format_answer(answer),
        file=top.file.replace(".pdf", "") if top else None,
        part=top.part if top else None,
        chapter=top.chapter if top else None,
        section=top.section if top else None,
        article=top.header if top else None,
        start_page=top.start_page if top else None
    )