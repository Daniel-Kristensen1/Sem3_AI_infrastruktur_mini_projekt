from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


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

# Kør server:
# uvicorn main:app --reload --host 127.0.0.1 --port 8000
# Gå til:
# http://127.0.0.1:8000/docs