from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from typing import Dict, List
from uuid import uuid4

MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16,
    device_map="auto")

class Message(BaseModel):
    role: str
    content: str

class ChatResponse(BaseModel):
    reply: str

class SimpleChatRequest(BaseModel):
    prompt: str

class CreateChatRequest(BaseModel):
    message: str

class AddMessageRequest(BaseModel):
    message: str

class ChatResource(BaseModel):
    id: str
    messages: List[Message]

class ReplaceMessageRequest(BaseModel):
    content: str

class EditLatestRequest(BaseModel):
    content: str

def build_prompt(messages: list[Message]) -> str:
    """
    Build a Phi-3 chat prompt from Message objects.
    """
    chat_format = [{"role": m.role, "content": m.content} for m in messages]
    prompt = tokenizer.apply_chat_template(
        chat_format,
        tokenize=False,
        add_generation_prompt=True)
    return prompt

def run_model(prompt: str, max_new_tokens: int = 256) -> str:
    """
    Takes a prompt (string), runs the model and returns the reply text.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)

    # Cut off the input, so we only keep the generated reply
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    reply_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return reply_text

def chat_from_messages(messages: list[Message]) -> str:
    """
    From Message objects to a reply string.
    """
    prompt = build_prompt(messages)
    reply = run_model(prompt)
    return reply

def chat_from_prompt(prompt: str) -> str:
    """
    Simple helper for a single user prompt.
    """
    user_message = Message(role="user", content=prompt)
    return chat_from_messages([user_message])

DATABASE: Dict[str, List[Message]] = {}

app = FastAPI(title="Phi-3 mini API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"])

@app.post("/v1/models/sentiment", response_model=ChatResponse)
def sentiment_explainer(req: SimpleChatRequest) -> ChatResponse:
    """
    Sentiment endpoint: LLM returns a short explanation sentence
    starting with Positive/Negative/Neutral/Mixed.
    """
    system_prompt = (
        "You are a sentiment analysis assistant. "
        "Given a text, decide if the overall sentiment is "
        "Positive, Negative, Neutral, or Mixed. "
        "Answer in a short sentence starting with the label, e.g.: "
        "\"Positive: ...\" or \"Negative: ...\".")
    full_prompt = f"{system_prompt}\n\nText:\n{req.prompt}"
    reply_text = chat_from_prompt(full_prompt)
    return ChatResponse(reply=reply_text)

@app.post("/v1/models/chat", response_model=ChatResponse)
def simple_chat_model(req: SimpleChatRequest) -> ChatResponse:
    """
    Simple chat model endpoint: one prompt in, one reply out.
    """
    reply_text = chat_from_prompt(req.prompt)
    return ChatResponse(reply=reply_text)

@app.get("/v1/models/chat")
def chat_model_info():
    return {
        "id": "phi-3-mini-chat",
        "type": "chat-completion",
        "description": "General-purpose chat model (Phi-3 mini).",
        "usage": {
            "endpoint": "POST /v1/models/chat",
            "request_body_example": {
                "prompt": "Explain containerization"
            },
            "response_example": {
                "reply": "Containerization is a way to wrap an application..."
            }
        }
    }

@app.get("/v1/models/sentiment")
def sentiment_model_info():
    return {
        "id": "phi-3-mini-sentiment",
        "type": "sentiment-analysis",
        "description": (
            "Use Phi-3 to make sentiment-analysis "
            "(Positive/Negative/Neutral/Mixed) of a text."
        ),
        "usage": {
            "endpoint": "POST /v1/models/sentiment",
            "request_body_example": {
                "prompt": "I love containers, but I hate debugging"
            },
            "response_example": {
                "reply": (
                    "Mixed: The text expresses something positive (love) "
                    "and something negative (hate debugging)."
                )
            }
        }
    }

@app.get("/v1/health")
def health():
    return {"status": "ok"}

@app.get("/v1/info")
def info():
    return {
        "message": "Phi-3 API running",
        "models": [
            {
                "id": "phi-3-mini-chat",
                "endpoint": "POST /v1/models/chat",
                "description": "Chat with Phi-3"
            },
            {
                "id": "phi-3-mini-sentiment",
                "endpoint": "POST /v1/models/sentiment",
                "description": "Sentiment-analysis of a text"
            },
        ],
        "helper_endpoints": {
            "GET /v1/health": "Returns {'status': 'ok'} if the server is running",
            "GET /v1/models/chat": "Info and example of usage for chat",
            "GET /v1/models/sentiment": "Info and example of usage for sentiment-analysis"
        }
    }

# CREATE a chat session
@app.post("/chats", response_model=ChatResource)
def create_chat(req: CreateChatRequest):
    chat_id = str(uuid4())

    user_msg = Message(role="user", content=req.message)
    assistant_reply = Message(
        role="assistant",
        content=chat_from_messages([user_msg]),
    )

    DATABASE[chat_id] = [user_msg, assistant_reply]

    return ChatResource(id=chat_id, messages=DATABASE[chat_id])

# READ a chat session
@app.get("/chats/{chat_id}", response_model=ChatResource)
def get_chat(chat_id: str):
    if chat_id not in DATABASE:
        raise HTTPException(status_code=404, detail="Chat not found")
    return ChatResource(id=chat_id, messages=DATABASE[chat_id])

# UPDATE: add message to chat and get new reply
@app.post("/chats/{chat_id}/messages", response_model=ChatResource)
def add_message(chat_id: str, req: AddMessageRequest):
    if chat_id not in DATABASE:
        raise HTTPException(status_code=404, detail="Chat not found")

    # 1. Add user message
    user_msg = Message(role="user", content=req.message)
    DATABASE[chat_id].append(user_msg)

    # 2. Generate assistant reply using entire conversation
    assistant_reply = Message(
        role="assistant",
        content=chat_from_messages(DATABASE[chat_id]),
    )
    DATABASE[chat_id].append(assistant_reply)

    return ChatResource(id=chat_id, messages=DATABASE[chat_id])

@app.put("/chats/{chat_id}/messages/latest", response_model=ChatResource)
def edit_latest_user_message(chat_id: str, req: EditLatestRequest):
    if chat_id not in DATABASE:
        raise HTTPException(404, "Chat not found")

    messages = DATABASE[chat_id]

    # Find index of the latest user message
    last_user_idx = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].role == "user":
            last_user_idx = i
            break

    if last_user_idx is None:
        raise HTTPException(400, "No user messages to edit")

    # Keep everything up to and including that user message
    messages = messages[: last_user_idx + 1]

    # Update that user message
    messages[last_user_idx].content = req.content

    # Regenerate assistant reply based on all messages up to that user msg
    new_reply = Message(
        role="assistant",
        content=chat_from_messages(messages),
    )
    messages.append(new_reply)

    # Save back to DB
    DATABASE[chat_id] = messages

    return ChatResource(id=chat_id, messages=messages)

@app.put("/chats/{chat_id}/messages/{index}", response_model=ChatResource)
def edit_message(chat_id: str, index: int, req: ReplaceMessageRequest):
    if chat_id not in DATABASE:
        raise HTTPException(404, "Chat not found")

    messages = DATABASE[chat_id]

    # Build a list of indices of user messages only
    user_indices = [i for i, m in enumerate(messages) if m.role == "user"]

    # Validate "user index"
    if index < 0 or index >= len(user_indices):
        raise HTTPException(404, "User message index out of range")

    # Map user index -> real index in messages list
    msg_idx = user_indices[index]

    # Update that user message
    messages[msg_idx].content = req.content

    # Cut off everything after this user message
    messages = messages[: msg_idx + 1]

    # Regenerate next assistant reply
    new_reply = Message(
        role="assistant",
        content=chat_from_messages(messages),
    )
    messages.append(new_reply)

    # Save back to DB
    DATABASE[chat_id] = messages

    return ChatResource(id=chat_id, messages=messages)

# DELETE a chat session
@app.delete("/chats/{chat_id}")
def delete_chat(chat_id: str):
    if chat_id not in DATABASE:
        raise HTTPException(status_code=404, detail="Chat not found")

    del DATABASE[chat_id]
    return {"status": "deleted", "chat_id": chat_id}