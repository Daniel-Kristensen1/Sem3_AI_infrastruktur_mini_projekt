# AutoTokenizer: Giver os Hugging Face-biblioteket, så vi kan få AutoTokenizer som laver tekst <-> tokens(tal)
# AutoModelForCausalLM: Loader en causal language model (LLM) til tekstgenerering
# Torch bruger vi til datatyper, device (GPU/CPU) der køres også en generate()-beregning med torch.
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Navnet på den model vi bruger
model_id = "microsoft/Phi-3-mini-4k-instruct" # En LLM model udviklet af microsoft. Har ca. 3.8B parametre, optimeret til inference på GPU, trænet til samtaler og instruktioner. Kommer fra Hugging Face Modelhub
# https://huggingface.co/microsoft/Phi-3-mini-4k-instruct?utm_source=chatgpt.com

#AutoTokenizer kikker på model_id og finder den rigtige tokenizertype til Phi-3
# Den ved hvordan man splitter tekst til tokens, hvilke special tokens |user| |assistant| bruges, og hvordan chat-templaten ser ud (apply_chat_template)
tok = AutoTokenizer.from_pretrained(model_id)

# Loader vægtene til selve LLM'en (Phi-3)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    # Vi gemmer model vægte som bfloat6 i stedet for 32, hvilket giver mindre VRAM, hurtigere, og stadig en fin præcision i generering. 
    torch_dtype=torch.bfloat16,
    # Kikker efter tilgængelig GPU ellers falder den tilbage på CPU
    device_map="auto",
    )

# Definere samtalen, skal laves om til at være et loop og samtale skal gemmes til at føre en konstant samtale
messages = [
    {"role": "system", "content": "You are a helpful assistant that explains things clearly."},
    {"role": "user", "content": "Hvad er en backbone i en CNN?"},
]

# Byg prompt i det rigtige Phi-3-chatformat
# Tager vores python liste af messages og oversætter dem til det rå tekst-format som Phi-3 faktisk er trænet på.
prompt = tok.apply_chat_template(
    messages,
    # Skal stadig undersøges
    tokenize=False,
    add_generation_prompt=True,  # tilføjer <|assistant|> til sidst
)
# Tokeniser prompten og flytter til samme device som modellen.
inputs = tok(prompt, return_tensors="pt").to(model.device) # Konverterer tekst -> numeriske tokens (PyTorch tensors) og flytter til GPU(eller det device som tidligere blev fundet).

#Generer tekst uden at tracke gradienter
with torch.no_grad(): # Kør kun fremad-pass ikke træning
    outputs = model.generate(
        **inputs, # Pakker dict ud
        max_new_tokens=256, # hvor mange nye tokens modellen må generere ud over prompten
        temperature=0.7, # hvor "random/creativ" den er 0.1/0.3 mere konservativ og deterministisk. Hvor at 1.0/1.5 er mere kreativ/kaotisk
        do_sample=True, # brug sampling (slump i stedet for bare greedy "tag altid det mest sandsynlige næste token"
    )# Resultatet bliver outputs som typisk er en tensor med form [batch_size, sekvenslængde]

# Decoder tokens til tekst igen som så kan printes.
text = tok.decode(outputs[0], skip_special_tokens=True)
print(text)

# OVERSIGT
#Hvad laver programmet “som helhed”?

#Loader værktøjskassen: transformers + torch.

#Vælger en LLM (Phi-3 mini instruct).

#Loader tokenizer og model til GPU med bfloat16.

#Bygger en samtale som en Python-liste af messages.

#Konverterer samtalen til det nøjagtige tekst-format, modellen er trænet på.

#Tokeniserer til tal/tensors på samme device som modellen.

#Beder modellen om at fortsætte teksten (generere svar).

#Oversætter tokens tilbage til tekst og printer det.

#Det er præcis de trin, du senere kan pakke ind i et FastAPI-endpoint som fx:

#POST /v1/chat der tager messages som JSON

#laver apply_chat_template + generate

#og returnerer assistant-svaret.