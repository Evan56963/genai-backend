from fastapi import APIRouter, Request
from sqlmodel import select

from app.models import Message

router = APIRouter(prefix="/manage", tags=["Manage"])

@router.get("/conversations", response_model=list[Message])
async def get_conversations(request: Request):

    state = request.app.state
    
    statement = select(Message).order_by(Message.created_at)
    messages = state.session.exec(statement).all()
    
    return messages

@router.delete("/conversations/delete")
async def delete_conversations(conversation_id: str, request: Request):

    state = request.app.state
    
    statement = select(Message).where(Message.conversation_id == conversation_id)
    messages = state.session.exec(statement).all()
    
    for message in messages:
        state.session.delete(message)
    
    state.session.commit()