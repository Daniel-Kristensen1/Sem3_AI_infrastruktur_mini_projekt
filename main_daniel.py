from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from typing import Dict, List
from uuid import uuid4

# LLM setup
MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16,
    device_map="auto",
)


# Datamodeller
class Message(BaseModel):
    role: str       # "system", "user" eller "assistant"
    content: str

class ChatRequest(BaseModel):
    messages: list[Message]

class ChatResponse(BaseModel):
    reply: str

class SimpleChatRequest(BaseModel):
    prompt: str


# ------------------------------------------------ 
# REST API for chat management
class CreateChatRequest(BaseModel):
    message: str     # initial user message


class AddMessageRequest(BaseModel):
    message: str     # user message being added


class ChatResource(BaseModel):
    id: str
    messages: List[Message]

class ReplaceMessageRequest(BaseModel):
    content: str   # new content for the message

class EditLatestRequest(BaseModel):
    content: str   # new content for the latest user message

class SentimentRequest(BaseModel):
    text: str

class SentimentResponse(BaseModel):
    sentiment: str
# ------------------------------------------------

def build_prompt(messages: list[Message]) -> str:
    """
    Bygger en Phi-3 chat-prompt ud fra Message-objekter.
    """
    chat_format = [
        {"role": m.role, "content": m.content}
        for m in messages
    ]

    prompt = tokenizer.apply_chat_template(
        chat_format,
        tokenize=False,
        add_generation_prompt=True,
    )
    return prompt


def run_model(prompt: str, max_new_tokens: int = 256) -> str:
    """
    Tager en prompt (string), kører modellen og returnerer svar-teksten.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
        )

    # Skær input fra, så vi kun har det genererede svar
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    reply_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return reply_text


def chat_from_messages(messages: list[Message]) -> str:
    """
    Fra Message-objekter til svartekst.
    """
    prompt = build_prompt(messages)
    reply = run_model(prompt)
    return reply


def chat_from_prompt(prompt: str) -> str:
    """
    Simpel helper til én user-prompt.
    """
    user_message = Message(role="user", content=prompt)
    return chat_from_messages([user_message])

# In-Memory database til chats

DATABASE: Dict[str, List[Message]] = {}   # chat_id -> messages list


# FastAPI app + endpoints
app = FastAPI(title="Phi-3 mini API")


@app.post("/v1/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    """
    Fuld chat-endpoint: tager en liste af beskeder (system/user/assistant).
    """
    reply_text = chat_from_messages(req.messages)
    return ChatResponse(reply=reply_text)


@app.post("/v1/simple-chat", response_model=ChatResponse)
def simple_chat(req: SimpleChatRequest) -> ChatResponse:
    """
    Simpelt endpoint: én prompt ind, ét svar ud.
    """
    reply_text = chat_from_prompt(req.prompt)
    return ChatResponse(reply=reply_text)


@app.get("/")
def root():
    return {
        "message": "Phi-3 API kører ✓",
        "usage": "Send POST til /v1/simple-chat med {'prompt': 'din tekst'}",
    }


# ------------------------------------------------ 
# REST 

###############
#### CHAT #####
###############

# CREATE a chat session
@app.post("/chats", response_model=ChatResource)
def create_chat(req: CreateChatRequest):
    chat_id = str(uuid4())

    user_msg = Message(role="user", content=req.message)
    assistant_reply = Message(
        role="assistant",
        content=chat_from_messages([user_msg])
    )

    DATABASE[chat_id] = [user_msg, assistant_reply]

    return ChatResource(id=chat_id, messages=DATABASE[chat_id])

# READ a chat session
@app.get("/chats/{chat_id}", response_model=ChatResource)
def get_chat(chat_id: str):
    if chat_id not in DATABASE:
        raise HTTPException(status_code=404, detail="Chat not found")

    return ChatResource(id=chat_id, messages=DATABASE[chat_id])


# UPDATE a chat session 

@app.post("/chats/{chat_id}/messages", response_model=ChatResource)
def add_message(chat_id: str, req: AddMessageRequest):
    if chat_id not in DATABASE:
        raise HTTPException(status_code=404, detail="Chat not found")

    # 1. Add user message
    user_msg = Message(role="user", content=req.message)
    DATABASE[chat_id].append(user_msg)

    # 2. Generate assistant reply using the *entire conversation*
    assistant_reply = Message(
        role="assistant",
        content=chat_from_messages(DATABASE[chat_id])
    )
    DATABASE[chat_id].append(assistant_reply)

    return ChatResource(id=chat_id, messages=DATABASE[chat_id])


# Update the latest user message in a chat session (and regenerate new response)
@app.put("/chats/{chat_id}/messages/latest", response_model=ChatResource)
def edit_latest_user_message(chat_id: str, req: EditLatestRequest):
    if chat_id not in DATABASE:
        raise HTTPException(404, "Chat not found")

    messages = DATABASE[chat_id]

    # last message must be user
    if messages[-1].role != "user":
        raise HTTPException(400, "Only latest user message can be edited")

    # update latest user msg
    messages[-1].content = req.content

    # delete assistant reply
    if len(messages) >= 2 and messages[-2].role == "assistant":
        messages.pop()

    # regenerate the assistant reply
    new_reply = Message(
        role="assistant",
        content=chat_from_messages(messages)
    )
    messages.append(new_reply)

    return ChatResource(id=chat_id, messages=messages)

# Update a ANY message in a chat session (Remove all messages after it and regenerate new response)
@app.put("/chats/{chat_id}/messages/{index}", response_model=ChatResource)
def edit_message(chat_id: str, index: int, req: ReplaceMessageRequest):

    if chat_id not in DATABASE:
        raise HTTPException(404, "Chat not found")

    messages = DATABASE[chat_id]

    # Validate index
    if index < 0 or index >= len(messages):
        raise HTTPException(404, "Message index out of range")

    # Only user messages can be edited
    if messages[index].role != "user":
        raise HTTPException(400, "Can only edit user messages")

    # Update the content
    messages[index].content = req.content

    # Delete everything after this message
    messages = messages[:index + 1]

    # Regenerate the next assistant message
    new_reply = Message(
        role="assistant",
        content=chat_from_messages(messages)
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


####################################
######## Sentiment Analysis ########
####################################

@app.post("/v1/sentiment", response_model=SentimentResponse)
def sentiment_analysis(req: SentimentRequest):
    system_msg = Message(role="system", content="You are a sentiment analyzer. Reply with positive, negative, or neutral.")
    user_msg = Message(role="user", content=req.text)
    
    sentiment = chat_from_messages([system_msg, user_msg])
    
    return SentimentResponse(sentiment=sentiment.strip())
# ------------------------------------------------ 


# Kør server:
# uvicorn main:app --reload --host 127.0.0.1 --port 8000
# Gå til:
# http://127.0.0.1:8000/docs