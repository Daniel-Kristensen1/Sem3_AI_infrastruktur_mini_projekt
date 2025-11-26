# Let Python-baseimage
FROM python:3.11-slim

# Logs skal komme direkte i terminalen
ENV PYTHONUNBUFFERED=1

# Arbejdsmappen inde i containeren
WORKDIR /app

# Systempakker (build tools osv. – gode til fx torch)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Kopiér requirements ind i containeren
COPY requirements.txt .

# Installer Python-pakker
# Vi tilføjer ekstra index-url til PyTorch CUDA-hjul (+cu121)
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu121 -r requirements.txt

# Kopiér resten af projektet (inkl. main.py)
COPY . .

# FastAPI/uvicorn kører på port 8000
EXPOSE 8000

# Start serveren når containeren kører
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
