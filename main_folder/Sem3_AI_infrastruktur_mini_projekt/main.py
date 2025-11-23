from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

app = FastAPI(title="Miniproject")

class GenerateRequest(BaseModel):
    prompt: str

class GenerateResponse(BaseModel):
    response: str

def generate_answer(prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=150)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/model", response_model=dict)
async def model_info():
    return {"model": model_name}

@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    text = generate_answer(req.prompt)
    return GenerateResponse(response=text)